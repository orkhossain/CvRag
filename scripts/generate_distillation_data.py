"""
CLI wrapper for the distillation data generator.
Used by CI and local dev to produce teacher-labeled training JSONL.

Usage:
    uv run python scripts/generate_distillation_data.py
    uv run python scripts/generate_distillation_data.py --examples-per-intent 3 --output out.jsonl
    uv run python scripts/generate_distillation_data.py --intents general_qa cover_letter
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LLM distillation training data")
    parser.add_argument("--intents", nargs="*", default=None, help="Intents to generate (default: all)")
    parser.add_argument("--examples-per-intent", type=int, default=5)
    parser.add_argument("--augment", action="store_true", help="Use teacher LLM to augment seed queries")
    parser.add_argument("--output", type=Path, default=ROOT / "distillation_output" / "train.jsonl")
    args = parser.parse_args()

    from cvrag.core.cv_data import ensure_cv_json
    from cvrag.rag.retriever import init_retriever
    from cvrag.distillation.generator import generate_examples, save_to_jsonl

    print("Initialising retriever …")
    ensure_cv_json()
    retriever = init_retriever()
    if retriever is None:
        print("ERROR: could not build retriever — is CV_PATH set?", file=sys.stderr)
        return 1

    print(f"Generating up to {args.examples_per_intent} examples per intent …")
    examples = list(
        generate_examples(
            intents=args.intents,
            examples_per_intent=args.examples_per_intent,
            augment=args.augment,
        )
    )

    if not examples:
        print("ERROR: no examples generated.", file=sys.stderr)
        return 1

    save_to_jsonl(examples, args.output)
    print(f"Saved {len(examples)} examples → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
