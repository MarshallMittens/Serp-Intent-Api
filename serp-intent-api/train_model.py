import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATASET_PATH = "dataset.csv"
MODEL_PATH = "intent_model.joblib"

def features_to_tokens(serp_features_value: str) -> str:
    """
    Convert 'reviews,people_also_ask' into '__feat_reviews__ __feat_people_also_ask__'
    """
    if not isinstance(serp_features_value, str) or not serp_features_value.strip():
        return ""
    feats = [f.strip().lower() for f in serp_features_value.split(",") if f.strip()]
    return " ".join([f"__feat_{f}__" for f in feats])

def main():
    df = pd.read_csv(DATASET_PATH)

    df["query"] = df["query"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()

    # If serp_features column missing, create it empty
    if "serp_features" not in df.columns:
        df["serp_features"] = ""

    # Build training text: query + feature tokens
    df["serp_features"] = df["serp_features"].fillna("")
    df["text"] = df["query"] + " " + df["serp_features"].apply(features_to_tokens)

    df = df[df["query"].str.len() > 0]
    df = df[df["intent"].str.len() > 0]

    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=2000
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== Accuracy =====")
    print(f"{acc:.4f}")

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
