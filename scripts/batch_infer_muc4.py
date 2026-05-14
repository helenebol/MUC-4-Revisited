#!/usr/bin/env python3
"""
Batch inference on MUC-4 test CSV and save predictions to file.

Reads the test data, loads the trained model
from --model_path, runs generation per row and writes
JSONL predictions to --output_path.
"""

import os
import sys
import json
import argparse
import pandas as pd

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate_muc4_model import MUC4EventExtractor, _sanitize_model_name  # noqa: E402


def run_batch_inference(
    model_path: str,
    base_model: str = None,
    test_csv: str = "muc4_processed_multi/muc4_test.csv",
    output_path: str = "../predictions/muc4_test_preds.jsonl",
    limit: int = 0,
    model_root: str = None,
):
    print(f"Starting batch inference on {os.uname().nodename}")

    # Load test data
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    df = pd.read_csv(test_csv)
    # New preprocessing only guarantees an 'input' column (plus optional 'doc_id' and 'output')
    if "input" not in df.columns:
        raise ValueError("Test CSV must contain an 'input' column")

    # Resolve adapter path if only model_root/base_model provided
    resolved_model_path = model_path
    if (resolved_model_path is None or len(str(resolved_model_path)) == 0) and model_root and base_model:
        subdir = _sanitize_model_name(base_model)
        resolved_model_path = os.path.join(model_root, subdir)
    if not resolved_model_path:
        raise ValueError("Provide --model_path or both --model_root and --base_model")

    # If output_path left default, write under resolved model dir using test name
    if output_path in ("predictions/muc4_test_preds.jsonl", "../predictions/muc4_test_preds.jsonl"):
        test_name = os.path.splitext(os.path.basename(test_csv))[0]
        output_path = os.path.join(resolved_model_path, f"{test_name}_preds.jsonl")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Init extractor
    print(f"Initialising extractor with model_path={resolved_model_path}, base_model={base_model}")
    extractor = MUC4EventExtractor(
            model_path=resolved_model_path, 
            base_model=base_model, 
            multi_event=True,
            )

    # Multi-event only: deduplicate identical inputs (one prediction per document)
    seen = {}
    for i, text in enumerate(df["input"].tolist()):
        if text not in seen:
            seen[text] = i  # keep first occurrence index
    idx_list = list(seen.values())

    # Apply limit if requested
    if limit > 0:
        idx_list = idx_list[: min(limit, len(idx_list))]

    print(f"Total unique inputs to predict: {len(idx_list)} (limit={limit})")

    # Iterate and predict
    with open(output_path, "w") as fout:
        for idx in idx_list:
            row = df.iloc[idx]
            input_text = row["input"]

            pred = extractor.extract_events(input_text)
            if isinstance(pred, dict) and pred.get("error"):
                #  retry once with left padding to avoid empty output
                extractor.tokenizer.padding_side = "left"
                pred = extractor.extract_events(input_text)

            record = {
                "doc_id": row.get("doc_id", f"doc_{idx}"),
                "instruction": extractor.last_instruction,
                "input": input_text,
                "gold_output": row.get("output", None),
                "pred_output": pred,
            }
            fout.write(json.dumps(record) + "\n")

    print(f"Wrote {len(idx_list)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch inference on MUC-4 test CSV")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (adapter dir)")
    parser.add_argument("--model_root", type=str, default=None, help="Root dir with per-base-model subdirs")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name if different from training")
    parser.add_argument("--test_csv", type=str, default="muc4_processed/muc4_test.csv", help="Path to test CSV")
    parser.add_argument("--output_path", type=str, default="predictions/muc4_test_preds.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows (0 = all)")
    # Multi-event only; flag removed (kept suppressed for backwards compatibility if scripts pass it).
    parser.add_argument("--multi_event", action="store_true", default=True, help=argparse.SUPPRESS)
    args = parser.parse_args()

    run_batch_inference(
        model_path=args.model_path,
        base_model=args.base_model,
        test_csv=args.test_csv,
        output_path=args.output_path,
        limit=args.limit,
        model_root=args.model_root,
    )


if __name__ == "__main__":
    main()


