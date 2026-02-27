from openai import OpenAI
from dotenv import load_dotenv
import os
import json

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
    base_url=BASE_URL
)

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"


# =====================================================
# 1️⃣ TASK EXTRACTION FUNCTION
# =====================================================

def extract_tasks_with_llm(transcript: str):

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
                {"role": "user", "content": transcript}
            ]
        )

        content = response.choices[0].message.content
        print("\n===== LLM TASK RAW OUTPUT =====")
        print(content)
        print("================================\n")

        # Try direct JSON parse
        try:
            parsed = json.loads(content)
            return parsed
        except Exception:
            # Attempt to extract JSON block
            start = content.find("{")
            end = content.rfind("}") + 1

            if start != -1 and end != -1:
                try:
                    return json.loads(content[start:end])
                except:
                    pass

        print("Task JSON parsing failed.")
        return {"tasks": []}

    except Exception as e:
        print("LLM TASK CALL FAILED:", str(e))
        return {"tasks": []}


# =====================================================
# 2️⃣ EXPLANATION GENERATION FUNCTION
# =====================================================

def generate_explanations(tickets):

    explanation_prompt = """
    You are an Agile AI assistant.

    For each ticket below, provide a short explanation (maximum 50 words)
    describing why it received its category and priority.

    Keep explanations concise and professional.

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
                {"role": "user", "content": json.dumps(tickets)}
            ]
        )

        content = response.choices[0].message.content
        print("\n===== LLM EXPLANATION RAW OUTPUT =====")
        print(content)
        print("=======================================\n")

        try:
            parsed = json.loads(content)
            return parsed
        except Exception:
            start = content.find("{")
            end = content.rfind("}") + 1

            if start != -1 and end != -1:
                try:
                    return json.loads(content[start:end])
                except:
                    pass

        print("Explanation JSON parsing failed.")
        return {"explanations": []}

    except Exception as e:
        print("LLM EXPLANATION CALL FAILED:", str(e))
        return {"explanations": []}