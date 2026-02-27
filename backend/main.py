import re
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from backend.llm_extractor import extract_tasks_with_llm

# -----------------------------
# Load Models
# -----------------------------

MODEL_PATH = "backend/models/"

embedder = joblib.load(MODEL_PATH + "embedder.pkl")
classifier = joblib.load(MODEL_PATH + "classifier.pkl")
regressor = joblib.load(MODEL_PATH + "regressor.pkl")
label_encoder = joblib.load(MODEL_PATH + "label_encoder.pkl")

app = FastAPI()


# -----------------------------
# Request Schema
# -----------------------------

class TextInput(BaseModel):
    text: str


# -----------------------------
# Task Extraction
# -----------------------------

ACTION_KEYWORDS = [
    # action verbs
    "fix", "add", "implement", "update",
    "migrate", "optimize", "refactor",
    "investigate", "create", "enable",
    "allow", "resolve",

    # problem indicators
    "crash", "fails", "fail", "error",
    "broken", "not working", "timeout",
    "exception", "issue"
]


def extract_tasks(transcript):
    sentences = re.split(r'[.\n]', transcript)
    tasks = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if any(keyword in sentence.lower() for keyword in ACTION_KEYWORDS):
            tasks.append(sentence)

    return tasks


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/analyze")
def analyze(input_data: TextInput):

    transcript = input_data.text

    llm_output = extract_tasks_with_llm(transcript)

    if "tasks" not in llm_output:
        return {"message": "No tasks detected."}

    results = []

    for task in llm_output["tasks"]:

        title = task["title"]
        severity = task["severity"]
        urgency_flag = task["urgency"]

        # ML embedding
        embedding = embedder.encode([title])
        class_pred = classifier.predict(embedding)
        ml_priority = float(regressor.predict(embedding)[0])
        category = label_encoder.inverse_transform(class_pred)[0]

        # Severity mapping
        severity_map = {
            "Low": 10,
            "Medium": 30,
            "High": 60,
            "Critical": 85
        }

        severity_score = severity_map.get(severity, 20)
        urgency_bonus = 15 if urgency_flag else 0

        # Hybrid fusion
        final_priority = (ml_priority * 0.6) + (severity_score * 0.3) + urgency_bonus
        final_priority = float(min(round(final_priority, 2), 100))

        results.append({
            "title": title,
            "category": category,
            "severity": severity,
            "priority": final_priority
        })

    return {
        "tickets": results,
        "total_tasks": len(results)
    }