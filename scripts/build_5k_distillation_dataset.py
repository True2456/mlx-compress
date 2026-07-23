#!/usr/bin/env python3
"""
5,000 Sample Parallel Production Distillation Dataset Generator (Gemini 3.6 Flash)
Features: 25 Parallel Workers, Thread-Safe Deduplication, Thread-Safe Autosaving, Subtopic Randomization.
"""
import json
import time
import os
import random
import threading
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
NUM_WORKERS = 25  # High parallel throughput (25 concurrent requests)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    for env_path in [Path.home() / ".env", Path(".env")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    GEMINI_API_KEY = line.split("=", 1)[1].strip().strip("\"'")
                    break

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "calib" / "reap_5k_balanced_recovery.json"
TARGET_TOTAL = 5000

SUBTOPICS = {
    "general_reasoning": [
        "Logic Puzzles & Knights/Knaves", "Probability & Combinatorics", "Physics Motion & Hydrostatics",
        "Financial Trade-off Optimization", "Time & Rate Calculations", "Spatial Orientation & Grid Navigation",
        "Syllogisms & Formal Logic", "Game Theory & Nash Equilibrium", "Causal Chain Analysis", "Hypothetical Counterfactuals"
    ],
    "tool_calling_api": [
        "E-commerce Checkout & Refund Routing", "Cloud Infrastructure Provisioning (AWS/GCP)", "Database Querying & Schema Validation",
        "IoT Sensor Telemetry Alerting", "Healthcare Patient Record Routing", "Financial Fraud Risk Scoring",
        "Calendar & Event Scheduling Routing", "CRM Lead Allocation", "DevOps CI/CD Pipeline Triggers", "Multi-API Mashup Parsing"
    ],
    "multimodal_spatial_vision": [
        "Mobile App Screen Bounding Box Parsing [ymin, xmin, ymax, xmax]", "Web Dashboard UI Coordinate Extraction",
        "OCR Form Field Mapping", "Bar/Line Chart Data Point Extraction", "Architecture Diagram Node Linking",
        "CAD Blueprint Spatial Relations", "Infographic Statistic Mining", "Table Header/Row Grounding",
        "PDF Document Layout Inspection", "Heatmap Intensity Coordinate Resolution"
    ],
    "coding_refactoring": [
        "Python AsyncIO Concurrency & Deadlocks", "Node.js Memory Leak Resolution", "JavaScript Event Loop Optimization",
        "Tree & Graph Traversal Algorithms", "Dynamic Programming Matrix Optimization", "C++ Pointer & Reference Safety",
        "SQL Query Index Tuning & N+1 Fixes", "Rust Memory Borrowing & Lifetimes", "FastAPI Data Pipeline Bottlenecks", "PyTorch GPU Memory Optimization"
    ],
    "instruction_formatting": [
        "Nested Metadata JSON Output", "Strict XML Tag Compliance", "Regex-Validated String Patterns",
        "Multi-language Translation Constraints", "Strict Markdown Table Generation", "Role-Playing Persona Restrictions",
        "Negative Constraints (Forbidden Word Enforcement)", "Structured Output Key Ordering", "YAML Config Generator", "Delimiter Wrapped Payload"
    ]
}

DOMAIN_TARGETS = {
    "general_reasoning": 1250,      # 25%
    "tool_calling_api": 1250,       # 25%
    "multimodal_spatial_vision": 1000, # 20%
    "coding_refactoring": 750,      # 15%
    "instruction_formatting": 750   # 15%
}

lock = threading.Lock()

def query_gemini(system_prompt: str, user_prompt: str) -> str:
    """Query Gemini 3.6 Flash with exponential backoff on 429 rate limits."""
    models = ["models/gemini-3.6-flash", "models/gemini-3.5-flash", "models/gemini-3.1-flash-lite"]
    
    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/{m}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.85, "maxOutputTokens": 4096}
        }
        for attempt in range(4):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=90)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                elif resp.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                elif resp.status_code != 404:
                    break
            except Exception:
                time.sleep(1)
    return ""

def load_existing_dataset() -> tuple[list[dict], set[str], dict[str, int]]:
    dataset = []
    seen_prompts = set()
    domain_counts = {k: 0 for k in DOMAIN_TARGETS}

    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                dataset = json.load(f)
            for item in dataset:
                d = item.get("domain")
                if d in domain_counts:
                    domain_counts[d] += 1
                prompt = item["messages"][0]["content"].strip().lower()
                seen_prompts.add(prompt)
            print(f"🔄 Resuming from checkpoint with {len(dataset)} existing records.")
        except Exception as e:
            print(f"⚠️ Warning loading checkpoint: {e}")
    return dataset, seen_prompts, domain_counts

def save_dataset_threadsafe(dataset: list[dict]):
    """Thread-safe atomic save to disk."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(dataset, f, indent=2)
    tmp_path.replace(OUTPUT_FILE)

def process_single_task(domain: str, subtopic: str, seen_prompts: set[str]) -> dict | None:
    """Generate 1 prompt + solution pair."""
    seed_sys = "You are an expert AI dataset designer. Output ONLY a raw JSON array of 5 string user prompts."
    seed_prompt = (
        f"Generate 5 highly distinct, creative, challenging user prompts about '{subtopic}' for {domain}. "
        f"Each prompt must be concise (under 25 words). "
        f"Output ONLY a valid JSON array of 5 strings."
    )

    raw_tasks = query_gemini(seed_sys, seed_prompt)
    if not raw_tasks:
        return None

    try:
        clean = raw_tasks.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
            clean = clean.split("```")[0].strip()
        if "[" in clean and "]" in clean:
            clean = clean[clean.index("["):clean.rindex("]") + 1]
        tasks = json.loads(clean)
    except Exception:
        return None

    if not isinstance(tasks, list) or not tasks:
        return None

    solver_sys = (
        "You are an expert AI assistant. First, think step-by-step inside <think> and </think> tags. "
        "After closing </think>, output your final, clean, production-ready solution."
    )

    for task in tasks:
        if not isinstance(task, str) or not task.strip():
            continue
        norm_prompt = task.strip().lower()

        with lock:
            if norm_prompt in seen_prompts:
                continue
            seen_prompts.add(norm_prompt)

        solution = query_gemini(solver_sys, task)
        if not solution:
            continue

        return {
            "domain": domain,
            "subtopic": subtopic,
            "messages": [
                {"role": "user", "content": task.strip()},
                {"role": "assistant", "content": solution.strip()}
            ]
        }
    return None

def worker_job(dataset: list[dict], seen_prompts: set[str], domain_counts: dict[str, int]):
    with lock:
        active_domains = [d for d, target in DOMAIN_TARGETS.items() if domain_counts[d] < target]
        if not active_domains or len(dataset) >= TARGET_TOTAL:
            return False
        domain = random.choice(active_domains)
        subtopic = random.choice(SUBTOPICS[domain])

    res = process_single_task(domain, subtopic, seen_prompts)
    if res:
        with lock:
            if len(dataset) < TARGET_TOTAL and domain_counts[res["domain"]] < DOMAIN_TARGETS[res["domain"]]:
                res["id"] = f"sample_{len(dataset) + 1:05d}"
                dataset.append(res)
                domain_counts[res["domain"]] += 1
                save_dataset_threadsafe(dataset)
                print(f"[{len(dataset):04d}/{TARGET_TOTAL}] +1 [{res['domain']} / {res['subtopic'][:18]}] | Saved to disk.")
    return True

def main():
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found.")
        return

    print("=" * 65)
    print("Parallel 5,000 Sample Production Distillation Generator")
    print(f"Target: {TARGET_TOTAL} samples | Workers: {NUM_WORKERS} Parallel Threads")
    print("=" * 65)

    dataset, seen_prompts, domain_counts = load_existing_dataset()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        while len(dataset) < TARGET_TOTAL:
            # Maintain active worker pool
            while len(futures) < NUM_WORKERS * 2 and len(dataset) < TARGET_TOTAL:
                futures.append(executor.submit(worker_job, dataset, seen_prompts, domain_counts))

            # Prune completed futures
            done = [f for f in futures if f.done()]
            for f in done:
                futures.remove(f)
            time.sleep(0.1)

    save_dataset_threadsafe(dataset)
    print("\n" + "=" * 65)
    print("✅ 5,000 SAMPLE PARALLEL DISTILLATION COMPLETED!")
    print(f"Total Unique Samples: {len(dataset)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    main()
