"""
Convert MUC-4 prediction JSONL files into JSONL for evaluation.

Input: one or more directories containing JSONL files:
  {"doc_id": "TST4-MUC4-0001", "pred_output": [ { ...event fields... }, ... ]}

Output (per line):
  {"id": "TST4-MUC4-0001", "templates": [ {"id": "TST4-MUC4-0001", ...fields...}, ... ]}

Example:
  python convert_preds.py \
    --pred_dirs ../predictions/OLMo-base ../predictions/llama-3.2 \
    --pattern muc4_test_preds*.jsonl
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def iter_prediction_files(pred_dirs: List[str], pattern: str) -> List[str]:
    
    files: List[str] = []
    for d in pred_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, pattern))))
    return files


def normalise_templates(pred_output: Any) -> List[Dict[str, Any]]:
    # Expected: list of event dicts
    if isinstance(pred_output, list):
        return [ev for ev in pred_output if isinstance(ev, dict)]
    if isinstance(pred_output, dict):
        
        if pred_output.get("error"): # If error key, treat as no templates
            return []
        return [pred_output]
    return []


def reverse_map_fields(event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map model output field names to official MUC-4 field names"""
   
    official_fields = {
        "incident_date",
        "incident_location_country",
        "incident_location_city",
        "incident_type",
        "incident_stage_of_execution",
        "incident_instrument_id",
        "incident_instrument_type",
        "perp_incident_category",
        "perp_individual_id",
        "perp_organization_id",
        "perp_organization_confidence",
        "phys_tgt_id",
        "phys_tgt_type",
        "phys_tgt_number",
        "phys_tgt_effect_of_incident",
        "hum_tgt_name",
        "hum_tgt_description",
        "hum_tgt_type",
        "hum_tgt_number",
        "hum_tgt_effect_of_incident",
    }

    # Reverse mapping 
    field_name_reverse_map = {
        "event type": "incident_type",
        "incident type": "incident_type",
        "type": "incident_type",
        "event": "incident_type",
        "date": "incident_date",
        "country": "incident_location_country",
        "city": "incident_location_city",
        "event stage": "incident_stage_of_execution",
        "stage": "incident_stage_of_execution",
        "weapon": "incident_instrument_id",
        "weapon id": "incident_instrument_id",
        "weapon type": "incident_instrument_type",
        "perpetrator category": "perp_incident_category",
        "perp category": "perp_incident_category",
        "perpetrator individual": "perp_individual_id",
        "perp individual": "perp_individual_id",
        "perpetrator organization": "perp_organization_id",
        "perp organization": "perp_organization_id",
        "perpetrator confidence": "perp_organization_confidence",
        "perp confidence": "perp_organization_confidence",
        "physical target": "phys_tgt_id",
        "physical target id": "phys_tgt_id",
        "physical target type": "phys_tgt_type",
        "physical target number": "phys_tgt_number",
        "effect on physical target": "phys_tgt_effect_of_incident",
        "victim name": "hum_tgt_name",
        "victim description": "hum_tgt_description",
        "victim type": "hum_tgt_type",
        "victim number": "hum_tgt_number",
        "effect on victim": "hum_tgt_effect_of_incident",
    }

    reversed_dict: Dict[str, Any] = {}
    for key, value in event_dict.items():
        if key == "id":
            # Keep id 
            reversed_dict["id"] = value
            continue
        lower_key = key.lower().strip()
        mapped = field_name_reverse_map.get(lower_key, key)
        # Only keep official fields 
        if mapped in official_fields:
            reversed_dict[mapped] = value
    return reversed_dict


def main():
    parser = argparse.ArgumentParser(description="Convert predictions to eval JSONL format")
    parser.add_argument("--pred_dirs", nargs="+", required=True,
                        help="Directories containing prediction JSONL files")
    parser.add_argument("--pattern", default="muc4_test_preds.jsonl",
                        help="Pattern for prediction files inside each directory")
    args = parser.parse_args()

    files = iter_prediction_files(args.pred_dirs, args.pattern)
    if not files:
        print("No input files matched")
        return

    total_written = 0
    for in_path in files:
        # Determine output path next to input file
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_predictions.jsonl"

        seen: set = set()
        written = 0
        with open(in_path, "r") as fin, open(out_path, "w") as pred_output:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = rec.get("doc_id")
                if not doc_id:
                    doc_id = rec.get("id")
                if not doc_id:
                    # require doc_id in inputs
                    print(f"No doc_id in input: {line}")
                    continue

                # de-duplicate by doc_id onlykeep first occurrence
                if doc_id in seen:
                    print(f"Duplicate doc_id: {doc_id}")
                    continue
                seen.add(doc_id)

                templates = normalise_templates(rec.get("pred_output"))
                out_templates = []
                for event in templates:
                    # Map field names back to original MUC-4 format and add per-template id
                    reversed_event = reverse_map_fields(event)
                    # id first key
                    event_with_id_first = {"id": doc_id}
                    for k, v in reversed_event.items():
                        if k == "id":
                            continue
                        event_with_id_first[k] = v
                    out_templates.append(event_with_id_first)
                # Make at least one template exists for each doc
                if not out_templates:
                    out_templates = [{"id": doc_id}]

                out_line = {
                    "id": doc_id,
                    "templates": out_templates,
                }
                pred_output.write(json.dumps(out_line) + "\n")
                written += 1

        total_written += written
        print(f"Wrote {written} examples -> {out_path}")

    print(f"Total examples written across files: {total_written}")


if __name__ == "__main__":
    main()
