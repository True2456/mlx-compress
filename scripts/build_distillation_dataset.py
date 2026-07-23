#!/usr/bin/env python3
"""
Multi-Teacher Agentic Distillation Generator (Gemini 3.6 Flash / Claude Fable 5)
Balanced post-pruning dataset generator preventing catastrophic forgetting & preserving vision/reasoning.
"""
import json
import time
import os
import requests
from pathlib import Path

def load_key(env_name: str) -> str:
    val = os.environ.get(env_name)
    if val:
        return val
    for env_path in [Path.home() / ".env", Path(".env")]:
        if env_path.exists():
            try:
                for line in env_path.read_text().splitlines():
                    if line.startswith(f"{env_name}="):
                        return line.split("=", 1)[1].strip().strip("\"'")
            except Exception:
                pass
    return ""

# Config: Set default to "gemini"
PROVIDER = os.environ.get("TEACHER_PROVIDER", "gemini").lower()
GEMINI_API_KEY = load_key("GEMINI_API_KEY")
ANTHROPIC_API_KEY = load_key("ANTHROPIC_API_KEY")

def get_gcloud_token() -> str:
    """Fetch gcloud auth token if ADC is configured."""
    gcloud_paths = [
        "gcloud",
        str(Path.home() / "google-cloud-sdk" / "bin" / "gcloud"),
        "/opt/homebrew/bin/gcloud",
        "/usr/local/bin/gcloud"
    ]
    for bin_path in gcloud_paths:
        try:
            import subprocess
            res = subprocess.run([bin_path, "auth", "print-access-token"], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
    return ""

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "calib" / "reap_balanced_recovery.json"

# Balanced Capability Buckets (Prevents Coding Overfitting & Protects Vision/Reasoning)
# Code is capped at ~15% to avoid catastrophic forgetting of general intelligence.
CAPABILITY_DOMAINS = {
    "general_reasoning": {
        "weight_share": "25%",
        "desc": "Logical deduction, multi-step word problems, decision making, common sense reasoning"
    },
    "tool_calling_api": {
        "weight_share": "25%",
        "desc": "JSON tool calls, function routing, API execution schemas, parameter validation"
    },
    "multimodal_spatial_vision": {
        "weight_share": "20%",
        "desc": "UI layout parsing, spatial coordinate grounding [ymin, xmin, ymax, xmax], OCR extraction, chart/diagram interpretation"
    },
    "coding_refactoring": {
        "weight_share": "15%",
        "desc": "Algorithmic problem solving, Python/JS refactoring, memory leak debugging"
    },
    "instruction_formatting": {
        "weight_share": "15%",
        "desc": "Strict JSON schema adherence, system prompt compliance, clean stop-tag formatting"
    }
}

def query_gemini(system_prompt: str, user_prompt: str) -> str:
    """Query Gemini API using Gemini 3.6 Flash / 3.5 Flash."""
    token = get_gcloud_token()
    models = ["models/gemini-3.6-flash", "models/gemini-3.5-flash", "models/gemini-3.1-flash-lite"]
    
    for m in models:
        if GEMINI_API_KEY:
            url = f"https://generativelanguage.googleapis.com/v1beta/{m}:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
        elif token:
            url = f"https://generativelanguage.googleapis.com/v1beta/{m}:generateContent"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        else:
            print("❌ Error: Neither GEMINI_API_KEY nor gcloud OAuth ADC token is available.")
            return ""

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4096}
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=90)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            elif resp.status_code != 404:
                print(f"❌ Gemini ({m}) Error {resp.status_code}: {resp.text[:150]}")
                return ""
        except Exception as e:
            print(f"❌ Gemini request error: {e}")
            return ""
    return ""

def query_claude(system_prompt: str, user_prompt: str) -> str:
    """Query Claude Fable 5 via Anthropic API"""
    if not ANTHROPIC_API_KEY:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment.")
        return ""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-fable-5",
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": 2048,
        "temperature": 0.7
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=90)
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        print(f"❌ Claude Error {resp.status_code}: {resp.text[:150]}")
    except Exception as e:
        print(f"❌ Claude request error: {e}")
    return ""

def generate_teacher_response(system: str, user: str) -> str:
    if PROVIDER == "gemini":
        return query_gemini(system, user)
    return query_claude(system, user)

def main():
    print("="*65)
    print(f"Balanced Post-Pruning Distillation Generator")
    print(f"Teacher Provider: {PROVIDER.upper()}")
    print("="*65)

    dataset = []

    for domain_key, info in CAPABILITY_DOMAINS.items():
        desc = info["desc"]
        share = info["weight_share"]
        print(f"\n🎯 Generating [{share}] seed tasks for domain: {domain_key}")
        
        gen_prompt = (
            f"Generate 5 challenging user task prompts focusing on: {desc}. "
            f"Keep each task prompt concise (under 25 words). "
            f"Output ONLY a raw JSON array of 5 strings, e.g. [\"task 1\", \"task 2\", ...]."
        )
        
        raw_tasks = generate_teacher_response("You are a SOTA benchmark designer. Output strictly raw JSON arrays.", gen_prompt)
        if not raw_tasks:
            continue

        try:
            clean = raw_tasks.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.split("```")[0].strip()
            # Extract bracketed JSON array if prefixed by conversational text
            if "[" in clean and "]" in clean:
                clean = clean[clean.index("["):clean.rindex("]") + 1]
            tasks = json.loads(clean)
        except Exception as e:
            print(f"   ❌ Failed to parse tasks: {e}\nRaw output: {raw_tasks[:100]}")
            continue

        solver_system = (
            "You are an expert AI assistant. First, think step-by-step inside <think> and </think> tags. "
            "After closing </think>, output your final, clean, production-ready solution."
        )

        for task in tasks:
            print(f"   Generating trace for task: {task[:55]}...")
            solution = generate_teacher_response(solver_system, task)
            if solution:
                dataset.append({
                    "domain": domain_key,
                    "target_share": share,
                    "messages": [
                        {"role": "user", "content": task},
                        {"role": "assistant", "content": solution}
                    ]
                })
            time.sleep(0.5)

    if dataset:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(dataset, f, indent=2)
        print("\n" + "="*65)
        print(f"✅ DISTILLATION DATASET GENERATED SUCCESSFULLY!")
        print(f"Total Traces: {len(dataset)}")
        print(f"Saved to: {OUTPUT_FILE}")
        print("="*65)
    else:
        print("\n❌ No dataset generated. Check API keys.")

if __name__ == "__main__":
    main()
