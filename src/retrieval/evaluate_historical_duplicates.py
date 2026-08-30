from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support

INPUT_PATH = Path("data/processed/complaints_master_synthetic.csv")
OUTPUT_DIR = Path("artifacts/faiss_index")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMN = "complaint_text"
TOP_K = 5
BATCH_SIZE = 32

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}. Run the dataset generator first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "complaint_id",
        "created_at",
        TEXT_COLUMN,
        "is_duplicate",
        "duplicate_of_id",
        "category",
        "language",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df["duplicate_of_id"] = df["duplicate_of_id"].fillna("").astype(str).str.strip()
    df["is_duplicate"] = pd.to_numeric(df["is_duplicate"], errors="coerce").fillna(0)

    df = df.dropna(subset=["created_at"])
    df = df[df[TEXT_COLUMN].ne("")]

    return df.sort_values(
        ["created_at", "complaint_id"]
    ).reset_index(drop=True)


def encode_all_texts(
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


def rank_of_original(
    candidate_ids: list[str],
    original_id: str,
) -> int | None:
    try:
        return candidate_ids.index(original_id) + 1
    except ValueError:
        return None


def evaluate_historical_retrieval(
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    indexed_ids: list[str] = []
    rows: list[dict] = []

    for position, row in df.iterrows():
        complaint_id = str(row["complaint_id"])
        is_duplicate = int(row["is_duplicate"])
        original_id = str(row["duplicate_of_id"])

        if index.ntotal > 0:
            search_k = min(TOP_K, index.ntotal)
            scores, indices = index.search(
                embeddings[position].reshape(1, -1),
                search_k,
            )

            candidate_ids = [
                indexed_ids[int(index_position)]
                for index_position in indices[0]
                if index_position != -1
            ]

            top_similarity = float(scores[0][0])
            top_match_id = candidate_ids[0] if candidate_ids else ""
        else:
            candidate_ids = []
            top_similarity = 0.0
            top_match_id = ""

        original_rank = (
            rank_of_original(candidate_ids, original_id)
            if is_duplicate == 1 and original_id
            else None
        )

        rows.append(
            {
                "complaint_id": complaint_id,
                "created_at": row["created_at"],
                "complaint_text": row[TEXT_COLUMN],
                "category": row["category"],
                "language": row["language"],
                "actual_is_duplicate": is_duplicate,
                "actual_duplicate_of_id": original_id,
                "top_match_id": top_match_id,
                "top_similarity": round(top_similarity, 4),
                "candidate_ids": "|".join(candidate_ids),
                "original_rank": original_rank,
                "top1_exact_original_match": int(original_rank == 1),
                "top5_exact_original_match": int(
                    original_rank is not None and original_rank <= TOP_K
                ),
            }
        )

        index.add(embeddings[position].reshape(1, -1))
        indexed_ids.append(complaint_id)

    return pd.DataFrame(rows)


def compute_metrics(evaluation_df: pd.DataFrame) -> dict:
    duplicate_rows = evaluation_df[
        evaluation_df["actual_is_duplicate"] == 1
    ].copy()

    top1_recall = float(
        duplicate_rows["top1_exact_original_match"].mean()
        if not duplicate_rows.empty
        else 0.0
    )

    top5_recall = float(
        duplicate_rows["top5_exact_original_match"].mean()
        if not duplicate_rows.empty
        else 0.0
    )

    reciprocal_ranks = duplicate_rows["original_rank"].dropna().apply(
        lambda rank: 1 / rank
    )

    mrr = float(reciprocal_ranks.mean()) if not reciprocal_ranks.empty else 0.0

    y_true = evaluation_df["actual_is_duplicate"].to_numpy()
    scores = evaluation_df["top_similarity"].to_numpy()

    threshold_results = []

    for threshold in THRESHOLDS:
        y_pred = (scores >= threshold).astype(int)

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

    selected_threshold = max(
        threshold_results,
        key=lambda result: result["f1"],
    )

    return {
        "total_records": len(evaluation_df),
        "duplicate_records": int(evaluation_df["actual_is_duplicate"].sum()),
        "top_1_exact_original_recall": round(top1_recall, 4),
        "top_5_exact_original_recall": round(top5_recall, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "selected_threshold": selected_threshold["threshold"],
        "selected_threshold_metrics": selected_threshold,
        "threshold_results": threshold_results,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(df)} complaints...")
    embeddings = encode_all_texts(model, df[TEXT_COLUMN].tolist())

    print("Evaluating chronological historical retrieval...")
    evaluation_df = evaluate_historical_retrieval(df, embeddings)
    metrics = compute_metrics(evaluation_df)

    evaluation_path = OUTPUT_DIR / "historical_duplicate_evaluation.csv"
    metrics_path = OUTPUT_DIR / "historical_duplicate_metrics.json"

    evaluation_df.to_csv(evaluation_path, index=False, encoding="utf-8")
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nHistorical duplicate-detection metrics:")
    print(f"Total records: {metrics['total_records']}")
    print(f"Known duplicates: {metrics['duplicate_records']}")
    print(
        "Top-1 exact-original recall: "
        f"{metrics['top_1_exact_original_recall']:.4f}"
    )
    print(
        "Top-5 exact-original recall: "
        f"{metrics['top_5_exact_original_recall']:.4f}"
    )
    print(f"MRR: {metrics['mean_reciprocal_rank']:.4f}")
    print(f"Selected threshold: {metrics['selected_threshold']:.2f}")

    selected = metrics["selected_threshold_metrics"]
    print(
        "Precision / Recall / F1: "
        f"{selected['precision']:.4f} / "
        f"{selected['recall']:.4f} / "
        f"{selected['f1']:.4f}"
    )

    print(f"\nSaved evaluation: {evaluation_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()