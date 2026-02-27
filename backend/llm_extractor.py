import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

API_KEY = os.getenv("FEATHERLESS_API_KEY")
BASE_URL = os.getenv("FEATHERLESS_BASE_URL")

if not API_KEY or not BASE_URL:
    print("WARNING: Featherless API key or base URL not loaded.")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"


def _safe_json_parse(content: str, empty_key: str) -> Dict[str, Any]:
    """Attempt to parse JSON, falling back to an empty structure."""
    content = content.strip()

    if not content:
        return {empty_key: []}

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end])
            except Exception:
                pass

    print(f"{empty_key.capitalize()} JSON parsing failed.")
    return {empty_key: []}


# =====================================================
# 1️⃣ TASK EXTRACTION FUNCTION
# =====================================================

def extract_tasks_with_llm(transcript: str) -> Dict[str, List[Dict[str, Any]]]:
    if not API_KEY or not BASE_URL:
        print("LLM TASK CALL SKIPPED: Featherless API configuration missing.")
        return {"tasks": []}

    system_prompt = """
    You are an Agile AI assistant.

    Extract actionable software development tasks from the transcript.

    For each task return:
    - title (short actionable sentence)
    - severity (Low / Medium / High / Critical)
    - urgency (true / false)

    Return ONLY valid JSON in this format:

    {
      "tasks": [
        {
          "title": "...",
          "severity": "...",
          "urgency": true
        }
      ]
    }
    """

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
        print("\n===== LLM TASK RAW OUTPUT =====")
        print(content)
        print("================================\n")

        return _safe_json_parse(content, empty_key="tasks")

    except Exception as e:
        print("LLM TASK CALL FAILED:", str(e))
        return {"tasks": []}


# =====================================================
# 2️⃣ EXPLANATION GENERATION FUNCTION
# =====================================================

def generate_explanations(tickets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    if not API_KEY or not BASE_URL:
        print("LLM EXPLANATION CALL SKIPPED: Featherless API configuration missing.")
        return {"explanations": []}

    explanation_prompt = """
    You are an Agile AI assistant.

    For each ticket below, provide a short explanation (maximum 100 words)
    describing why it received its category and priority and tell how to approach it.

    Keep explanations concise and professional and make it sound like a human wrote it.

    Return ONLY valid JSON in this format:

    {
      "explanations": [
        {
          "title": "...",
          "explanation": "..."
        }
      ]
    }
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": explanation_prompt},
                {"role": "user", "content": json.dumps(tickets)},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        print("\n===== LLM EXPLANATION RAW OUTPUT =====")
        print(content)
        print("=======================================\n")

        return _safe_json_parse(content, empty_key="explanations")

    except Exception as e:
        print("LLM EXPLANATION CALL FAILED:", str(e))
        return {"explanations": []}