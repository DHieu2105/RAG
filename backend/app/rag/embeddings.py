import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[3]
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "handbook_k67.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "handbook_k67_embeddings.npy"

MODEL_NAME = "BAAI/bge-m3"


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Đã embedding {len(chunks)} chunks")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()