"""
LLM integration for task extraction + explanation (single call, concurrency safe).
Compatible with Featherless OpenAI-style API.
"""

import json
import os
import re
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Environment Setup
# -----------------------------

load_dotenv()

API_KEY = os.getenv("FEATHERLESS_API_KEY")
BASE_URL = os.getenv("FEATHERLESS_BASE_URL")
MODEL_NAME = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2")

if not API_KEY or not BASE_URL:
    raise ValueError("Missing FEATHERLESS_API_KEY or FEATHERLESS_BASE_URL")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0,
)

# -----------------------------
# JSON Cleaning Utilities
# -----------------------------

def _extract_json_block(text: str) -> str:
    """Extract JSON block from model output safely."""
    if not text:
        return ""

    # Remove markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Extract first valid JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]

    return ""

# -----------------------------
# Main LLM Function (Single Call)
# -----------------------------

def extract_tasks_with_llm(transcript: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract tasks + explanation in ONE LLM call.
    Returns:
    {
        "tasks": [
            {
                "title": "...",
                "severity": "...",
                "urgency": true,
                "explanation": "..."
            }
        ]
    }
    """

    if not transcript.strip():
        return {"tasks": []}

    system_prompt = """
You are an Agile project assistant.

From the transcript:
1. Extract actionable software development tasks.
2. For each task provide:
   - title (short sentence)
   - severity (Low, Medium, High, Critical)
   - urgency (true or false)
   - explanation (max 80 words, professional tone, explain priority + approach)

Return ONLY valid JSON:

{
  "tasks": [
    {
      "title": "...",
      "severity": "High",
      "urgency": true,
      "explanation": "..."
    }
  ]
}

No markdown.
No extra text.
No commentary.
Only JSON.
"""

    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ],
            )

            content = (response.choices[0].message.content or "").strip()

            print("\n===== LLM RAW OUTPUT =====")
            print(content[:800])
            print("===========================\n")

            cleaned = _extract_json_block(content)

            data = json.loads(cleaned)

            if isinstance(data, dict) and "tasks" in data:
                return {"tasks": data["tasks"]}

            return {"tasks": []}

        except Exception as e:
            print(f"LLM CALL FAILED (Attempt {attempt+1}):", e)
            time.sleep(delay)
            delay *= 2  # exponential backoff

    # Final fallback
    return {"tasks": []}