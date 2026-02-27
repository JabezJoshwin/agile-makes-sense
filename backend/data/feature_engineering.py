import pandas as pd
import numpy as np
import os

INPUT_PATH = "backend/data/processed/"
OUTPUT_PATH = "backend/data/processed/structured_dataset.csv"


# ---------- Keyword Rules ---------- #

BUG_KEYWORDS = [
    "error", "exception", "fail", "crash", "not working",
    "issue", "bug", "incorrect", "unable", "broken", "fails",
    "failure", "timeout", "not responding", "incorrect behavior"
]

FEATURE_KEYWORDS = [
    "add", "implement", "support", "allow",
    "create", "introduce", "enable"
]

IMPROVEMENT_KEYWORDS = [
    "optimize", "improve", "refactor",
    "enhance", "performance"
]

URGENT_KEYWORDS = [
    "urgent", "asap", "critical",
    "immediately", "production", "revenue",
    "blocking", "high priority"
]

SEVERITY_KEYWORDS = [
    "crash", "data loss", "security",
    "breach", "production down", "outage",
    "failure", "corruption", "downtime"
]


# ---------- Category Detection ---------- #

def detect_category(text):
    text_lower = str(text).lower()

    if text_lower.startswith("how to"):
        return "Feature"

    if any(word in text_lower for word in FEATURE_KEYWORDS):
        return "Feature"

    if any(word in text_lower for word in IMPROVEMENT_KEYWORDS):
        return "Improvement"

    if any(word in text_lower for word in BUG_KEYWORDS):
        return "Bug"

    return "Task"


# ---------- Feature Scores ---------- #

def urgency_score(text):
    text_lower = str(text).lower()
    return sum(word in text_lower for word in URGENT_KEYWORDS)


def severity_score(text):
    text_lower = str(text).lower()
    return sum(word in text_lower for word in SEVERITY_KEYWORDS)


def complexity_score(text):
    length = len(str(text).split())

    if length < 20:
        return 1
    elif length < 50:
        return 2
    elif length < 100:
        return 3
    elif length < 200:
        return 4
    else:
        return 5


def assign_story_points(complexity):
    mapping = {
        1: 1,
        2: 2,
        3: 5,
        4: 8,
        5: 8
    }
    return mapping.get(complexity, 1)


# ---------- Intelligent Priority Logic ---------- #

def compute_priority(category, urgency, severity, complexity):
    score = 0

    # Base weight by category
    if category == "Bug":
        score += 35
    elif category == "Feature":
        score += 20
    elif category == "Improvement":
        score += 15
    else:  # Task
        score += 10

    # Severity impact (strong signal)
    score += severity * 25

    # Urgency impact
    score += urgency * 20

    # Complexity impact (moderate)
    score += complexity * 5

    return min(score, 100)


# ---------- Main Pipeline ---------- #

def main():
    all_dfs = []

    if not os.path.exists(INPUT_PATH):
        print("Input path does not exist.")
        return

    for file_name in os.listdir(INPUT_PATH):
        if file_name.endswith(".csv") and file_name != "structured_dataset.csv":
            file_path = os.path.join(INPUT_PATH, file_name)
            print("Reading:", file_path)

            try:
                temp_df = pd.read_csv(
                    file_path,
                    engine="python",
                    sep=",",
                    usecols=[0]
                )
                temp_df.columns = ["text"]
            except Exception as e:
                print(f"Skipping {file_name} due to parsing error:", e)
                continue

            all_dfs.append(temp_df)

    if not all_dfs:
        print("No valid CSV files found.")
        return

    df = pd.concat(all_dfs, ignore_index=True)

    print("Total combined rows before dedup:", len(df))

    df.drop_duplicates(subset=["text"], inplace=True)

    print("Total rows after dedup:", len(df))

    # Feature engineering
    df["category"] = df["text"].apply(detect_category)
    df["urgency_score"] = df["text"].apply(urgency_score)
    df["severity_score"] = df["text"].apply(severity_score)
    df["complexity_score"] = df["text"].apply(complexity_score)
    df["story_points"] = df["complexity_score"].apply(assign_story_points)

    df["priority_score"] = df.apply(
        lambda row: compute_priority(
            row["category"],
            row["urgency_score"],
            row["severity_score"],
            row["complexity_score"]
        ),
        axis=1
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("Structured dataset saved to:", OUTPUT_PATH)
    print("Final row count:", len(df))
    print("\nCategory distribution:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()