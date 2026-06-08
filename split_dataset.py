import argparse
import os
import sys
from pathlib import Path

# Add src/ to Python path so hipporag package can be imported
# sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.hipporag.utils.dataset_setup import prepare_experiment


SPLIT_CONFIGS = {
    "small_scale": {"n_queries": 50},
    "medium_100":  {"n_queries": 100},
    "medium_200":  {"n_queries": 200},
    "large_scale": {"n_queries": 500},
    "full_1000":   {"n_queries": 1000},
}


def build_paths(work_dir):
    return {
        "work_dir":  work_dir,
        "dataset":   work_dir / "reproduce" / "dataset",
    }


def load_env(work_dir):
    try:
        from dotenv import load_dotenv
        env_path = work_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded .env from {env_path}")
    except ImportError:
        pass

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Put it in .env or export it.")
        sys.exit(1)
    os.environ["GRAPHRAG_API_KEY"] = os.environ["GROQ_API_KEY"]


def parse_args():
    p = argparse.ArgumentParser(description="KET-RAG experiment setup")
    p.add_argument("--dataset",  default="hotpotqa",
                   choices=["hotpotqa", "musique", "2wikimultihopqa"])
    p.add_argument("--split",    default="small_scale",
                   choices=list(SPLIT_CONFIGS))
    p.add_argument("--work-dir", type=Path, default=Path.cwd())
    return p.parse_args()


def main():
    args = parse_args()
    paths = build_paths(args.work_dir)
    root = paths["work_dir"]

    print("=== HippoRAG Setup ===")
    print(f"Dataset:  {args.dataset}")
    print(f"Project:  {root}")
    print()

    prepare_experiment(
        root, paths["dataset"],
        args.dataset, args.split, SPLIT_CONFIGS,
    )


if __name__ == "__main__":
    main()
