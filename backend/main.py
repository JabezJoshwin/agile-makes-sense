import joblib
from pathlib import Path
from typing import Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ Only import the single unified function
from backend.llm_extractor import extract_tasks_with_llm


# -----------------------------
# Model Loading
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

_embedder = None
_classifier = None
_regressor = None
_label_encoder = None


def get_models() -> Tuple:
    global _embedder, _classifier, _regressor, _label_encoder

    if all([_embedder, _classifier, _regressor, _label_encoder]):
        return _embedder, _classifier, _regressor, _label_encoder

    paths = {
        "embedder": MODEL_DIR / "embedder.pkl",
        "classifier": MODEL_DIR / "classifier.pkl",
        "regressor": MODEL_DIR / "regressor.pkl",
        "label_encoder": MODEL_DIR / "label_encoder.pkl",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model files missing",
                "missing_files": missing,
            },
        )

    _embedder = joblib.load(paths["embedder"])
    _classifier = joblib.load(paths["classifier"])
    _regressor = joblib.load(paths["regressor"])
    _label_encoder = joblib.load(paths["label_encoder"])

    return _embedder, _classifier, _regressor, _label_encoder


app = FastAPI(title="Agile Makes Sense Backend")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------
# Request Schema
# -----------------------------

class TextInput(BaseModel):
    text: str


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/analyze")
def analyze(input_data: TextInput):

    transcript = input_data.text.strip()

    if not transcript:
        return {"tickets": [], "total_tasks": 0}

    embedder, classifier, regressor, label_encoder = get_models()

    # ✅ Single LLM Call (extract + explanation)
    llm_output = extract_tasks_with_llm(transcript)

    if not llm_output or "tasks" not in llm_output:
        return {"tickets": [], "total_tasks": 0}

    results = []

    severity_map = {
        "Low": 10,
        "Medium": 30,
        "High": 60,
        "Critical": 85,
    }

    for task in llm_output["tasks"]:

        title = task.get("title", "")
        severity = task.get("severity", "Medium")
        urgency_flag = task.get("urgency", False)
        explanation = task.get("explanation", "")

        if not title:
            continue

        # ML embedding
        embedding = embedder.encode([title])

        class_pred = classifier.predict(embedding)
        ml_priority = float(regressor.predict(embedding)[0])
        category = label_encoder.inverse_transform(class_pred)[0]

        # Severity scoring
        severity_score = severity_map.get(severity, 20)

        # Urgency scoring (1–100)
        base_urgency = 70 if urgency_flag else 20

        if severity == "Critical":
            base_urgency += 20
        elif severity == "High":
            base_urgency += 10

        urgency_score = min(base_urgency, 100)

        # Hybrid priority fusion
        final_priority = (
            (ml_priority * 0.5)
            + (severity_score * 0.3)
            + (urgency_score * 0.2)
        )

        final_priority = float(min(round(final_priority, 2), 100))

        # Story points estimation
        word_count = len(title.split())

        if word_count < 20:
            story_points = 1
        elif word_count < 50:
            story_points = 2
        elif word_count < 100:
            story_points = 5
        else:
            story_points = 8

        results.append({
            "title": title,
            "category": category,
            "severity": severity,
            "priority": final_priority,
            "story_points": story_points,
            "urgency_score": urgency_score,
            "explanation": explanation or
                "Priority determined based on severity, urgency, and business impact."
        })

    return {
        "tickets": results,
        "total_tasks": len(results),
    }