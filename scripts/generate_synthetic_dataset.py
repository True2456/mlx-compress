#!/usr/bin/env python3
import json
import time
import os
import requests
from pathlib import Path

# Anthropic API details
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-fable-5"  # Can change to "claude-opus-4-8" or "claude-3-5-sonnet-20241022"
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "calib" / "claude_synthetic_dataset.json"

# List of topics/domains to seed prompt generation
TOPICS = [
    "Complex Coding (FastAPI, multi-threading, memory leaks)",
    "Algorithm Design (Trees, graphs, dynamic programming)",
    "System Design & Architecture (Scaling, caching, consensus)",
    "Logical Reasoning & Word Problems",
    "Mathematics (Calculus, linear algebra, number theory)",
    "Scientific Explanations (Physics, quantum computing, biology)"
]

def load_api_key() -> str:
    """Load Anthropic API Key safely from env or ~/.env"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        for env_path in [Path.home() / ".env", Path(".env")]:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("ANTHROPIC_API_KEY="):
                        return line.split("=", 1)[1].strip().strip("\"'")
    return api_key or ""

def query_claude(system_prompt: str, user_prompt: str, api_key: str, max_tokens: int = 1524) -> str:
    """Send message to Anthropic Messages API"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=90)
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"   ❌ Request error: {e}")
        return ""

def main():
    api_key = load_api_key()
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment or ~/.env file.")
        print("Run this to set it securely:")
        print("printf 'Enter ANTHROPIC_API_KEY (typing hidden): ' && read -s val && echo && echo \"ANTHROPIC_API_KEY=$val\" >> ~/.env")
        return

    print("="*60)
    print(f"Claude Distillation Dataset Generator")
    print(f"Model: {MODEL_NAME}")
    print("="*60)

    # 1. Ask Claude to generate distinct questions/prompts for our topics
    generated_data = []
    
    system_prompt = (
        "You are an expert AI data engineer. Your job is to output high-quality, complex, "
        "and challenging user prompts for coding, math, and logic training."
    )
    
    for topic in TOPICS:
        print(f"\n📚 Generating seed prompts for: {topic}")
        seed_prompt = (
            f"Generate a list of 5 extremely challenging, multi-step user questions or coding tasks "
            f"about {topic}. Return the questions as a raw JSON array of strings, e.g. "
            f"[\"question 1\", \"question 2\", ...]. Output ONLY the JSON array, no conversational text."
        )
        
        raw_questions = query_claude(system_prompt, seed_prompt, api_key, max_tokens=1000)
        if not raw_questions:
            continue
            
        try:
            # Clean JSON if wrapped in markdown code blocks
            clean_json = raw_questions.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
            questions = json.loads(clean_json)
        except Exception as e:
            print(f"   ❌ Failed to parse JSON questions: {e}\nRaw output: {raw_questions}")
            continue

        # 2. For each generated question, query Claude Fable to solve it with reasoning
        solver_system = (
            "You are a helpful AI assistant. For every query, you MUST think step-by-step. "
            "Enclose your complete internal reasoning process inside <think> and </think> tags. "
            "After closing </think>, output your final, clean solution."
        )

        for q in questions:
            print(f"   Generating response for task: {q[:60]}...")
            solution = query_claude(solver_system, q, api_key, max_tokens=2048)
            if not solution:
                continue

            # Append in standard ChatML training format
            generated_data.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": solution}
                ]
            })
            
            # Rate limit politeness pause
            time.sleep(1.0)

    # Save to file
    if generated_data:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(generated_data, f, indent=2)
        print("\n" + "="*60)
        print("GENERATION COMPLETED SUCCESSFULLY!")
        print(f"Generated {len(generated_data)} ChatML examples.")
        print(f"Saved to: {OUTPUT_FILE}")
        print("="*60)
    else:
        print("\n❌ No data generated.")

if __name__ == "__main__":
    main()
