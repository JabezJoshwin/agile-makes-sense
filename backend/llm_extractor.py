from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()
# -----------------------------
# Featherless Configuration
# -----------------------------

#print(os.getenv("FEATHERLESS_API_KEY"))
client = OpenAI(
    api_key=os.getenv("FEATHERLESS_API_KEY"),
    base_url=os.getenv("FEATHERLESS_BASE_URL")  # e.g. https://api.featherless.ai/v1
)

MODEL_NAME = "deepseek-ai/DeepSeek-V3.2"


# -----------------------------
# LLM Extraction Function
# -----------------------------

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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ]
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except:
        # Fallback if model adds extra text
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])