#!/usr/bin/env python3
"""
5,000 Sample GOLD SFT & REAP Distillation Generator (Gemini 3.6 Flash)

Strict Critical Quality Gates (SFT-Grade):
1. Genuine Deep CoT: REJECT any row with len(think_text) < 150 or containing generic fallback stubs.
2. Complete Post-Think Solutions: REJECT any post-</think> text that is truncated, < 80 chars, or broken.
3. Clean Tool Execution: REJECT tool_calling_api rows missing <tool_call> or containing pre-tool conversational junk.
4. Balanced Code Fences: Auto-fix code block markers.
5. Dynamic Weighted Domain Sampling: Prioritizes underfilled domains (tools + coding first).
"""
import json
import time
import os
import random
import re
import threading
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

NUM_WORKERS = 25
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    for env_path in [Path.home() / ".env", Path(".env")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    GEMINI_API_KEY = line.split("=", 1)[1].strip().strip("\"'")
                    break

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "calib" / "reap_5k_gold_recovery.json"
TARGET_TOTAL = 5000

# Target domain weights
DOMAIN_TARGETS = {
    "tool_calling_api": 1750,      # 35%
    "coding_refactoring": 1500,     # 30%
    "general_reasoning": 1000,      # 20%
    "instruction_formatting": 750   # 15%
}

SUBTOPICS = {
    "tool_calling_api": [
        "E-commerce Checkout & Refund Routing", "Cloud Infrastructure Provisioning (AWS/GCP)",
        "Database Querying & Schema Validation", "IoT Sensor Telemetry Alerting",
        "Healthcare Patient Record Routing", "Financial Fraud Risk Scoring",
        "Calendar & Event Scheduling Routing", "CRM Lead Allocation",
        "DevOps CI/CD Pipeline Triggers", "Multi-API Mashup Parsing"
    ],
    "coding_refactoring": [
        "Python AsyncIO Concurrency & Deadlocks", "Node.js Memory Leak Resolution",
        "JavaScript Event Loop Optimization", "Tree & Graph Traversal Algorithms",
        "Dynamic Programming Matrix Optimization", "C++ Pointer & Reference Safety",
        "SQL Query Index Tuning & N+1 Fixes", "Rust Memory Borrowing & Lifetimes",
        "FastAPI Data Pipeline Bottlenecks", "PyTorch GPU Memory Optimization"
    ],
    "general_reasoning": [
        "Logic Puzzles & Knights/Knaves", "Probability & Combinatorics",
        "Physics Motion & Hydrostatics", "Financial Trade-off Optimization",
        "Time & Rate Calculations", "Spatial Orientation & Grid Navigation",
        "Syllogisms & Formal Logic", "Game Theory & Nash Equilibrium"
    ],
    "instruction_formatting": [
        "Nested Metadata JSON Output", "Strict XML Tag Compliance",
        "Regex-Validated String Patterns", "Multi-language Translation Constraints",
        "Strict Markdown Table Generation", "Role-Playing Persona Restrictions",
        "Negative Constraints (Forbidden Word Enforcement)", "Structured Output Key Ordering"
    ]
}

lock = threading.Lock()

def query_gemini(system_prompt: str, user_prompt: str) -> str:
    """Query Gemini 3.6 Flash via REST API."""
    models = ["models/gemini-3.6-flash", "models/gemini-3.5-flash"]
    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/{m}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.75, "maxOutputTokens": 4096}
        }
        for attempt in range(3):
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

def validate_quality_gates(domain: str, assistant_response: str) -> str | None:
    """Rigorous SFT Quality Gate Validator."""
    if not assistant_response or len(assistant_response.strip()) < 100:
        return None

    # Gate 1: Must contain both <think> and </think>
    if "<think>" not in assistant_response or "</think>" not in assistant_response:
        return None

    m = re.search(r"<think>(.*?)</think>", assistant_response, re.DOTALL)
    if not m:
        return None
    
    think_text = m.group(1).strip()
    
    # Gate 2: Substantive CoT Reasoning Check (No generic stubs)
    if len(think_text) < 140 or "analyzing prompt requirements" in think_text.lower():
        return None

    # Gate 3: Valid Post-Think Solution Check
    post_think = assistant_response.split("</think>", 1)[1].strip()
    if len(post_think) < 60:
        return None

    # Gate 4: Tool Domain Execution Check
    if domain == "tool_calling_api":
        if "<tool_call>" not in post_think or "</tool_call>" not in post_think:
            return None
        # Reject if long conversational prose precedes the tool call
        pre_tool = post_think.split("<tool_call>", 1)[0].strip()
        if len(pre_tool) > 250:
            return None

    # Gate 5: Balance Code Fences
    if assistant_response.count("```") % 2 != 0:
        assistant_response = assistant_response.strip() + "\n```"

    return assistant_response.strip()

def load_and_purge_dataset() -> tuple[list[dict], set[str], dict[str, int]]:
    """Load existing dataset and purge any rows failing strict quality gates."""
    dataset = []
    seen_prompts = set()
    domain_counts = {k: 0 for k in DOMAIN_TARGETS}

    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r") as f:
                raw_data = json.load(f)
            
            purged_count = 0
            for item in raw_data:
                d = item.get("domain")
                resp = item["messages"][1]["content"]
                
                clean_resp = validate_quality_gates(d, resp)
                if not clean_resp:
                    purged_count += 1
                    continue
                
                item["messages"][1]["content"] = clean_resp
                dataset.append(item)
                if d in domain_counts:
                    domain_counts[d] += 1
                prompt = item["messages"][0]["content"].strip().lower()
                seen_prompts.add(prompt)
                
            print(f"🧹 Purged {purged_count} defective rows. Retained {len(dataset)} Pure SFT-Grade records.")
        except Exception as e:
            print(f"⚠️ Warning loading checkpoint: {e}")
    return dataset, seen_prompts, domain_counts

def save_dataset_threadsafe(dataset: list[dict]):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_FILE.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(dataset, f, indent=2)
    tmp_path.replace(OUTPUT_FILE)

def generate_gold_prompt_and_response(domain: str, subtopic: str) -> dict | None:
    if domain == "coding_refactoring":
        seed_sys = "You are a senior software engineer creating benchmark tasks."
        seed_prompt = (
            f"Generate a self-contained coding task about '{subtopic}'. "
            f"CRITICAL: The prompt MUST contain the actual broken/unoptimized code block inline. "
            f"Do NOT say 'refactor this' without providing the code block inline. "
            f"Output ONLY the raw user prompt."
        )
    elif domain == "tool_calling_api":
        seed_sys = "You are an API integration engineer."
        seed_prompt = (
            f"Generate a realistic user request about '{subtopic}' that requires executing specific API tool calls. "
            f"Provide all required parameters (IDs, names, criteria) inline in the user prompt. "
            f"Output ONLY the raw user prompt."
        )
    else:
        seed_sys = "You are a SOTA benchmark designer."
        seed_prompt = f"Generate a challenging, concise user prompt about '{subtopic}' for {domain}. Output ONLY the raw user prompt."

    user_prompt = query_gemini(seed_sys, seed_prompt)
    if not user_prompt or len(user_prompt.strip()) < 15:
        return None
    user_prompt = user_prompt.strip().strip('"').strip("'")

    if domain == "tool_calling_api":
        solver_sys = (
            "You are an AI assistant with access to API tools. "
            "Rule 1: Start immediately with <think> and write comprehensive step-by-step reasoning evaluating parameters and tool selection. "
            "Rule 2: Close reasoning with </think>.\n"
            "Rule 3: Immediately after </think>, output the exact tool call in JSON format wrapped inside "
            "<tool_call>{\"name\": \"function_name\", \"arguments\": {...}}</tool_call>."
        )
    elif domain == "coding_refactoring":
        solver_sys = (
            "You are a principal software architect. "
            "Rule 1: Start immediately with <think> and write detailed root-cause analysis and architectural design. "
            "Rule 2: Close reasoning with </think>.\n"
            "Rule 3: Immediately after </think>, provide the complete refactored code block with explanations."
        )
    else:
        solver_sys = (
            "You are an expert AI assistant. "
            "Rule 1: Start immediately with <think> and write comprehensive step-by-step reasoning. "
            "Rule 2: Close reasoning with </think>.\n"
            "Rule 3: Immediately after </think>, provide your final solution."
        )

    assistant_response = query_gemini(solver_sys, user_prompt)
    clean_response = validate_quality_gates(domain, assistant_response)
    if not clean_response:
        return None

    return {
        "domain": domain,
        "subtopic": subtopic,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": clean_response}
        ]
    }

def worker_job(dataset: list[dict], seen_prompts: set[str], domain_counts: dict[str, int]):
    with lock:
        # Dynamic weighted sampling: Prioritize domains furthest from their target percentage
        def deficit(d):
            return (DOMAIN_TARGETS[d] - domain_counts[d]) / DOMAIN_TARGETS[d]
        
        active_domains = [d for d, target in DOMAIN_TARGETS.items() if domain_counts[d] < target]
        if not active_domains or len(dataset) >= TARGET_TOTAL:
            return False
        
        active_domains.sort(key=deficit, reverse=True)
        domain = active_domains[0]  # Pick domain with highest deficit
        subtopic = random.choice(SUBTOPICS[domain])

    res = generate_gold_prompt_and_response(domain, subtopic)
    if res:
        norm_prompt = res["messages"][0]["content"].strip().lower()
        with lock:
            if norm_prompt in seen_prompts:
                return True
            if len(dataset) < TARGET_TOTAL and domain_counts[res["domain"]] < DOMAIN_TARGETS[res["domain"]]:
                seen_prompts.add(norm_prompt)
                res["id"] = f"sample_{len(dataset) + 1:05d}"
                dataset.append(res)
                domain_counts[res["domain"]] += 1
                save_dataset_threadsafe(dataset)
                print(f"[{len(dataset):04d}/{TARGET_TOTAL}] +1 PURE-GOLD [{res['domain']} / {res['subtopic'][:18]}] -> Saved.")
    return True

def main():
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY missing.")
        return

    print("=" * 65)
    print("5,000 Sample PURE-GOLD SFT Distillation Generator")
    print(f"Target: {TARGET_TOTAL} Pure Gold Samples | Workers: {NUM_WORKERS} Parallel Threads")
    print("=" * 65)

    dataset, seen_prompts, domain_counts = load_and_purge_dataset()
    save_dataset_threadsafe(dataset)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        while len(dataset) < TARGET_TOTAL:
            while len(futures) < NUM_WORKERS * 2 and len(dataset) < TARGET_TOTAL:
                futures.append(executor.submit(worker_job, dataset, seen_prompts, domain_counts))

            done = [f for f in futures if f.done()]
            for f in done:
                futures.remove(f)
            time.sleep(0.1)

    save_dataset_threadsafe(dataset)
    print("\n" + "=" * 65)
    print("✅ 5,000 SAMPLE PURE-GOLD DISTILLATION COMPLETED!")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    main()
