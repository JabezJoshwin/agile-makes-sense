import json
import os
import pandas as pd

RAW_PATH = "backend/data/raw"
OUTPUT_PATH = "backend/data/processed/cleaned_dataset.csv"


def clean_github():
    records = []

    for file_name in os.listdir(RAW_PATH):
        if file_name.endswith(".json"):
            file_path = os.path.join(RAW_PATH, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"Processing GitHub file: {file_name} | Total records: {len(data)}")

            for issue in data:
                # Skip pull requests
                if issue.get("pull_request") is not None:
                    continue

                title = issue.get("title", "")
                body = issue.get("body") or ""
                text = (title + " " + body).strip()

                if len(text) > 20:
                    records.append({
                        "text": text,
                        "source": "github",
                        "created_at": issue.get("created_at"),
                        "state": issue.get("state")
                    })

    print("GitHub cleaned records:", len(records))
    return records


def clean_stackoverflow():
    records = []

    for file_name in os.listdir(RAW_PATH):
        if file_name.endswith(".csv"):
            file_path = os.path.join(RAW_PATH, file_name)

            df = pd.read_csv(file_path)
            print(f"Processing StackOverflow file: {file_name} | Rows: {len(df)}")

            for _, row in df.iterrows():
                title = str(row["Title"])
                body = str(row["Body"])
                text = (title + " " + body).strip()

                if len(text) > 20:
                    records.append({
                        "text": text,
                        "source": "stackoverflow",
                        "created_at": row["CreationDate"],
                        "state": "open"
                    })

    print("StackOverflow cleaned records:", len(records))
    return records


def main():
    github_records = clean_github()
    stack_records = clean_stackoverflow()

    all_records = github_records + stack_records

    print("Total combined records:", len(all_records))

    if len(all_records) == 0:
        print("No valid records found.")
        return

    df = pd.DataFrame(all_records)
    df.drop_duplicates(subset=["text"], inplace=True)

    os.makedirs("backend/data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved cleaned dataset to:", OUTPUT_PATH)
    print("Final row count:", len(df))


if __name__ == "__main__":
    main()