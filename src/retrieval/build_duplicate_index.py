from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support

TRAIN_PATH = Path("data/processed/train.csv")
VALIDATION_PATH = Path("data/processed/validation.csv")

ARTIFACT_DIR = Path("artifacts/faiss_index")
INDEX_PATH = ARTIFACT_DIR / "complaints.index"
METADATA_PATH = ARTIFACT_DIR / "complaint_metadata.csv"
CONFIG_PATH = ARTIFACT_DIR / "duplicate_detection_config.json"
EVALUATION_PATH = ARTIFACT_DIR / "validation_duplicate_evaluation.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMN = "complaint_text"
TOP_K = 5
BATCH_SIZE = 32

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    df = pd.read_csv(path)
    required = {
        "complaint_id",
        TEXT_COLUMN,
        "category",
        "is_duplicate",
        "duplicate_of_id",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df["duplicate_of_id"] = df["duplicate_of_id"].fillna("").astype(str).str.strip()

    return df[df[TEXT_COLUMN].ne("")].reset_index(drop=True)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def find_best_match(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    metadata: pd.DataFrame,
    query_id: str,
) -> tuple[str, float]:
    scores, indices = index.search(query_embedding.reshape(1, -1), TOP_K)

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        candidate_id = metadata.iloc[int(index_position)]["complaint_id"]
        if candidate_id != query_id:
            return str(candidate_id), float(score)

    return "", 0.0


def evaluate_duplicate_detection(
    validation_df: pd.DataFrame,
    index: faiss.IndexFlatIP,
    train_metadata: pd.DataFrame,
    model: SentenceTransformer,
) -> tuple[pd.DataFrame, dict]:
    validation_embeddings = encode_texts(model, validation_df[TEXT_COLUMN].tolist())
    rows = []

    for row_index, row in validation_df.iterrows():
        predicted_match_id, similarity_score = find_best_match(
            index=index,
            query_embedding=validation_embeddings[row_index],
            metadata=train_metadata,
            query_id=str(row["complaint_id"]),
        )

        actual_duplicate = int(row["is_duplicate"])
        actual_original_id = str(row["duplicate_of_id"]).strip()

        match_is_correct = int(
            actual_duplicate == 1
            and predicted_match_id == actual_original_id
        )

        rows.append(
            {
                "complaint_id": row["complaint_id"],
                "complaint_text": row[TEXT_COLUMN],
                "actual_is_duplicate": actual_duplicate,
                "actual_duplicate_of_id": actual_original_id,
                "predicted_match_id": predicted_match_id,
                "similarity_score": round(similarity_score, 4),
                "top1_exact_match": match_is_correct,
            }
        )

    evaluation_df = pd.DataFrame(rows)

    threshold_results = []
    y_true = evaluation_df["actual_is_duplicate"].to_numpy()

    for threshold in THRESHOLDS:
        y_pred = (evaluation_df["similarity_score"].to_numpy() >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        threshold_results.append(
            {
                "threshold": threshold,
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
            }
        )

    best_threshold_result = max(threshold_results, key=lambda result: result["f1"])

    metrics = {
        "validation_records": len(evaluation_df),
        "validation_duplicate_records": int(evaluation_df["actual_is_duplicate"].sum()),
        "top1_exact_duplicate_match_rate": round(
            float(evaluation_df.loc[
                evaluation_df["actual_is_duplicate"] == 1,
                "top1_exact_match",
            ].mean() or 0.0),
            4,
        ),
        "threshold_results": threshold_results,
        "selected_threshold": best_threshold_result["threshold"],
        "selected_threshold_metrics": best_threshold_result,
    }

    return evaluation_df, metrics


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nEncoding {len(train_df)} training complaints...")
    train_embeddings = encode_texts(model, train_df[TEXT_COLUMN].tolist())

    print("\nBuilding FAISS inner-product index...")
    index = build_index(train_embeddings)

    metadata_columns = [
        "complaint_id",
        "complaint_text",
        "normalized_text",
        "category",
        "department",
        "priority",
        "location_type",
        "specific_location",
        "is_duplicate",
        "duplicate_of_id",
    ]
    metadata_columns = [
        column for column in metadata_columns if column in train_df.columns
    ]
    train_metadata = train_df[metadata_columns].copy()

    print("\nEvaluating duplicate detection on validation data...")
    evaluation_df, metrics = evaluate_duplicate_detection(
        validation_df=validation_df,
        index=index,
        train_metadata=train_metadata,
        model=model,
    )

    faiss.write_index(index, str(INDEX_PATH))
    train_metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8")
    evaluation_df.to_csv(EVALUATION_PATH, index=False, encoding="utf-8")

    config = {
        "model_name": MODEL_NAME,
        "text_column": TEXT_COLUMN,
        "embedding_dimension": int(train_embeddings.shape[1]),
        "index_type": "IndexFlatIP",
        "top_k": TOP_K,
        "metrics": metrics,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nDuplicate-detection validation results:")
    print(
        "Top-1 exact original-match rate for known duplicates: "
        f"{metrics['top1_exact_duplicate_match_rate']:.4f}"
    )
    print(
        f"Selected similarity threshold: {metrics['selected_threshold']:.2f}"
    )
    print(
        "Threshold Precision / Recall / F1: "
        f"{metrics['selected_threshold_metrics']['precision']:.4f} / "
        f"{metrics['selected_threshold_metrics']['recall']:.4f} / "
        f"{metrics['selected_threshold_metrics']['f1']:.4f}"
    )

    print(f"\nSaved FAISS index: {INDEX_PATH}")
    print(f"Saved complaint metadata: {METADATA_PATH}")
    print(f"Saved evaluation results: {EVALUATION_PATH}")
    print(f"Saved configuration: {CONFIG_PATH}")


if __name__ == "__main__":
    main()