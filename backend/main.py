import joblib
from pathlib import Path
from typing import Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
<<<<<<< HEAD
from llm_extractor import extract_tasks_with_llm, generate_explanations
=======

from backend.llm_extractor import extract_tasks_with_llm, generate_explanations
>>>>>>> richan/main

# -----------------------------
# Model Loading
# -----------------------------

<<<<<<< HEAD
MODEL_PATH = "models/"
=======
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
>>>>>>> richan/main

_embedder = None
_classifier = None
_regressor = None
_label_encoder = None


def get_models() -> Tuple:
    """
    Lazily construct the embedding model on CPU and
    load the trained sklearn/XGBoost artifacts.
    This avoids device-specific pickles (e.g. MPS)
    causing runtime errors inside Docker.
    """
    global _embedder, _classifier, _regressor, _label_encoder

    if all([_embedder, _classifier, _regressor, _label_encoder]):
        return _embedder, _classifier, _regressor, _label_encoder

    missing = []
    paths = {
        "embedder": MODEL_DIR / "embedder.pkl",
        "classifier": MODEL_DIR / "classifier.pkl",
        "regressor": MODEL_DIR / "regressor.pkl",
        "label_encoder": MODEL_DIR / "label_encoder.pkl",
    }

    for _, path in paths.items():
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model files missing",
                "missing_files": missing,
            },
        )

    # Load CPU-compatible models saved by ml_pipeline.py
    _embedder = joblib.load(paths["embedder"])
    _classifier = joblib.load(paths["classifier"])
    _regressor = joblib.load(paths["regressor"])
    _label_encoder = joblib.load(paths["label_encoder"])

    return _embedder, _classifier, _regressor, _label_encoder


app = FastAPI(title="Agile Makes Sense Backend")


@app.get("/health")
def health_check():
    """Lightweight health check endpoint."""
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

    transcript = input_data.text

    # Ensure models are available before processing.
    embedder, classifier, regressor, label_encoder = get_models()

    # -------- LLM Extraction --------
    llm_output = extract_tasks_with_llm(transcript)

    if not llm_output or "tasks" not in llm_output:
        print("LLM returned empty or invalid response.")
        return {"tickets": [], "total_tasks": 0}

    results = []

    severity_map = {
        "Low": 10,
        "Medium": 30,
        "High": 60,
        "Critical": 85
    }

    # -------- Process Each Task --------
    for task in llm_output["tasks"]:

        title = task.get("title", "")
        severity = task.get("severity", "Medium")
        urgency_flag = task.get("urgency", False)

        if not title:
            continue

        # ML embedding
        embedding = embedder.encode([title])

        class_pred = classifier.predict(embedding)
        ml_priority = float(regressor.predict(embedding)[0])

        category = label_encoder.inverse_transform(class_pred)[0]

        # Severity + urgency scoring
        severity_score = severity_map.get(severity, 20)
        # -------- Urgency Scoring (1–100 scale) --------
        if urgency_flag:
            base_urgency = 70
        else:
            base_urgency = 20

        # Boost urgency based on severity
        if severity == "Critical":
            base_urgency += 20
        elif severity == "High":
            base_urgency += 10  

        urgency_score = min(base_urgency, 100)

        # Hybrid priority fusion
        final_priority = ((ml_priority * 0.5) + (severity_score * 0.3) + (urgency_score * 0.2))
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
            "urgency_score": urgency_score
        })

    # -------- LLM Explanations --------
    explanations = generate_explanations(results)

    explanation_map = {
        item["title"]: item["explanation"]
        for item in explanations.get("explanations", [])
    }

    for ticket in results:
        ticket["explanation"] = explanation_map.get(
            ticket["title"],
            "Priority determined based on severity, urgency, and business impact."
        )

    return {
        "tickets": results,
        "total_tasks": len(results)
    }