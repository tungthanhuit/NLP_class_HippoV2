import json
from pathlib import Path

dataset_dir = Path(__file__).resolve().parent
dataset_path = dataset_dir / "2wikimultihopqa.json"
corpus_path = dataset_dir / "2wikimultihopqa_corpus.json"
output_path = dataset_dir / "2wikimultihopqa_first50.json"
corpus_output_path = dataset_dir / "2wikimultihopqa_first50_corpus.json"


def save_first_n_items(source_path: Path, destination_path: Path, n: int = 50) -> None:
    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"{source_path.name}: {len(data)} samples")
        print("First sample keys:", list(data[0].keys()) if data else [])
        first_n = data[:n]
        with destination_path.open("w", encoding="utf-8") as f:
            json.dump(first_n, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(first_n)} samples to {destination_path.name}")
    elif isinstance(data, dict):
        print(f"{source_path.name}: top-level JSON is a dict.")
        print("Top-level keys:", list(data.keys()))
        for key, value in data.items():
            if hasattr(value, "__len__"):
                print(f"{key}: {len(value)}")
    else:
        print(f"Unexpected JSON type in {source_path.name}: {type(data)}")

save_first_n_items(dataset_path, output_path)
save_first_n_items(corpus_path, corpus_output_path)