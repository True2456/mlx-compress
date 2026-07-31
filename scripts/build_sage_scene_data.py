"""Build incremental scene-construction SFT examples from SAGE-10k layout JSONs.

Each scene becomes a multi-turn trajectory: turn 0 sets up the room (walls,
doors), each subsequent turn adds a batch of objects. This mirrors the
agentic/iterative style of the rest of the training mix (build up a result
step by step) rather than a single-shot "generate the whole scene" pair --
also multiplies examples per scene for free (a 59-object room yields ~6-10
turns, not 1).

Target code is a small, self-contained scene-construction API (Room/Scene)
defined here in the system prompt -- not SAGE's own toolkit -- so the model
learns a general "construct a realistic environment via code" pattern rather
than one dataset's internal API surface.

Usage:
    .venv/bin/python scripts/build_sage_scene_data.py \
        --layouts-dir data/lora_traces/sage3d/layouts \
        --out data/lora_traces/sage3d/scenes.jsonl \
        --objects-per-step 8 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

SYSTEM_PROMPT = """You are an autonomous coding agent that constructs realistic 3D \
environments using a simple scene-construction API. Build the scene \
incrementally, one call at a time, using this API:

    scene = Scene()
    room = scene.add_room(room_type: str, width: float, length: float, height: float)
    room.add_wall(start: tuple[float,float,float], end: tuple[float,float,float], \
height: float, thickness: float)
    room.add_door(wall_index: int, position_on_wall: float, width: float, \
height: float, door_type: str)
    room.add_object(obj_type: str, description: str, position: tuple[float,float,float], \
rotation: tuple[float,float,float], dimensions: tuple[float,float,float])

Positions are (x, y, z) in meters, rotation in degrees. Respond with only the \
code for the requested step -- assume prior turns' code has already run."""


def fmt_tuple(*vals):
    return "(" + ", ".join(f"{v:.4g}" for v in vals) + ")"


def room_setup_code(room):
    lines = [
        f'room = scene.add_room(room_type="{room["room_type"]}", '
        f'width={room["dimensions"]["width"]:.4g}, '
        f'length={room["dimensions"]["length"]:.4g}, '
        f'height={room["dimensions"]["height"]:.4g})'
    ]
    for w in room.get("walls", []):
        s, e = w["start_point"], w["end_point"]
        lines.append(
            f'room.add_wall(start={fmt_tuple(s["x"], s["y"], s["z"])}, '
            f'end={fmt_tuple(e["x"], e["y"], e["z"])}, '
            f'height={w["height"]:.4g}, thickness={w["thickness"]:.4g})'
        )
    for i, d in enumerate(room.get("doors", [])):
        lines.append(
            f'room.add_door(wall_index={i}, position_on_wall={d["position_on_wall"]:.4g}, '
            f'width={d["width"]:.4g}, height={d["height"]:.4g}, '
            f'door_type="{d["door_type"]}")'
        )
    return "\n".join(lines)


def object_code(obj):
    p, r, dim = obj["position"], obj["rotation"], obj["dimensions"]
    desc = obj.get("description", "").replace('"', "'")
    return (
        f'room.add_object(obj_type="{obj["type"]}", description="{desc}", '
        f'position={fmt_tuple(p["x"], p["y"], p["z"])}, '
        f'rotation={fmt_tuple(r["x"], r["y"], r["z"])}, '
        f'dimensions={fmt_tuple(dim["width"], dim["length"], dim["height"])})'
    )


def room_setup_instruction(room, rng, detailed):
    if detailed:
        return (
            f'Create a {room["room_type"].replace("_", " ")} that is '
            f'{room["dimensions"]["width"]:.3g}m x {room["dimensions"]["length"]:.3g}m x '
            f'{room["dimensions"]["height"]:.3g}m, with walls and '
            f'{len(room.get("doors", []))} door(s) as appropriate.'
        )
    return f'Set up a realistic {room["room_type"].replace("_", " ")}.'


def objects_instruction(batch, rng, detailed):
    descs = [o.get("description") or o["type"] for o in batch]
    if detailed:
        parts = []
        for o in batch:
            p = o["position"]
            parts.append(
                f'{o.get("description", o["type"])} at position '
                f'({p["x"]:.2g}, {p["y"]:.2g}, {p["z"]:.2g})'
            )
        return "Add the following: " + "; ".join(parts) + "."
    if len(descs) == 1:
        return f"Now add {descs[0]}, placed somewhere reasonable in the room."
    return "Now add: " + ", ".join(descs) + ". Place them somewhere reasonable in the room."


def build_trajectory(layout, objects_per_step, rng):
    rooms = layout.get("rooms", [])
    if not rooms:
        return None
    room = rooms[0]
    objects = room.get("objects", [])
    if not objects:
        return None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    records = []

    detailed = rng.random() < 0.4
    messages.append({"role": "user", "content": room_setup_instruction(room, rng, detailed)})
    messages.append({"role": "assistant", "content": "```python\nscene = Scene()\n" + room_setup_code(room) + "\n```"})
    records.append({"messages": list(messages)})

    batches = [objects[i:i + objects_per_step] for i in range(0, len(objects), objects_per_step)]
    for batch in batches:
        detailed = rng.random() < 0.4
        messages.append({"role": "user", "content": objects_instruction(batch, rng, detailed)})
        code = "```python\n" + "\n".join(object_code(o) for o in batch) + "\n```"
        messages.append({"role": "assistant", "content": code})
        records.append({"messages": list(messages)})

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts-dir", default="data/lora_traces/sage3d/layouts")
    ap.add_argument("--out", default="data/lora_traces/sage3d/scenes.jsonl")
    ap.add_argument("--objects-per-step", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    files = sorted(glob.glob(f"{a.layouts_dir}/*.json"))
    print(f"[sage] found {len(files)} layout files", flush=True)

    all_records = []
    n_scenes_ok = 0
    for fp in files:
        try:
            layout = json.loads(Path(fp).read_text())
        except Exception as e:
            print(f"  skip {fp}: {type(e).__name__}: {e}", flush=True)
            continue
        recs = build_trajectory(layout, a.objects_per_step, rng)
        if not recs:
            continue
        traj_id = Path(fp).stem
        for r in recs:
            r["_traj"] = traj_id
        all_records.extend(recs)
        n_scenes_ok += 1

    print(f"[sage] {n_scenes_ok}/{len(files)} scenes converted, {len(all_records)} step-records", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in all_records:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
    print(f"[sage] wrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
