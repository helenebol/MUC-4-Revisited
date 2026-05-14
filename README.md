## MUC-4 Revisited: Document-level Event Analysis Beyond Span-based Arguments

This repository contains a pipeline for **abstractive MUC‑4 event extraction** using LLMs with LoRA fine‑tuning from the paper: [MUC-4 Revisited: Document-level Event Analysis
Beyond Span-based Arguments](http://www.lrec-conf.org/proceedings/lrec2026/pdf/2026.lrec2026-1.617.pdf)

This repo includes:
- Preprocessing scripts to convert raw MUC‑4 files into JSON (`data/Processed/`)
- CSV creation for instruction tuning (one row per document; multi-event JSON-array output)
- Training (LoRA fine‑tuning)
- Batch inference + conversion of predictions to evaluation JSONL



### Directory overview

- **`data/`**
  - **`Original/`**: Original MUC-4 source files (**[MUC data download](https://www-nlpir.nist.gov/related_projects/muc/muc_data/muc_data_index.html)**) see `data/preprocessing/` for expected inputs).
  - **`preprocessing/`**: Scripts to turn the original files into JSON (`Processed/`).
    - `preprocess_muc.py`, `text_proc.py`, `run_preprocessing.sh`
  - **`Processed/`**: Canonical JSON used by this repo:
    - `{train,val,test}.json` and `keys.json` under each split directory.
- **`configs/`**
  - `muc_data.py`: Field templates that map between human‑readable labels (e.g. `"weapon"`) and official MUC‑4 field names (e.g. `"incident_instrument_id"`). 
- **`convert_data/`**
  - `muc4_data_processor.py`: Turns `data/Processed/` JSON into **training CSVs** for instruction‑tuning.
    - Produces **one row per document** where the label is a **JSON array of event objects**.
    - Outputs CSVs under `muc4_processed_multi/`:
      - `muc4_{train,val,test}.csv` and `muc4_combined.csv`
  - `convert_preds.py`: Converts model prediction JSONL files into **evaluation‑ready JSONL** format (`id` + `templates[...]` with official MUC‑4 field names).
- **`scripts/`**
  - `finetune.py`: Generic LoRA fine‑tuning on instruction‑following CSVs.
  - `train_muc4_event_extraction.py`: MUC‑4 specific training.
  - `evaluate_muc4_model.py`:
    - Defines `MUC4EventExtractor`, which loads a base model + LoRA adapters 
  - `batch_infer_muc4.py`:
    - Uses `MUC4EventExtractor` to run **batch inference** over the test data
    - Writes one JSONL record per document with `doc_id`, `instruction`, `input`, `gold_output`, and `pred_output`.
- **`instructions.py`**
  - Defines the shared instruction strings 

### Environment setup

Tested with **Python 3.10**.

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

### Running the pipeline

#### 1. Preprocess raw MUC‑4 data

From the `MUC-4-Revisited` directory:

```bash
bash data/preprocessing/run_preprocessing.sh
```

This populates `data/Processed/{train,val,test}/` with `*.json` and `keys.json` used downstream.

#### 2. Create training CSVs from processed JSON (multi-event only)

Use `convert_data/muc4_data_processor.py` to turn the JSON + keys into CSVs:

```bash
python -m convert_data.muc4_data_processor
```

By default, this script:
- reads from `data/Processed/`
- writes `muc4_processed_multi/muc4_{train,val,test}.csv` (one row per document)


#### 3. Fine‑tune a model on MUC‑4


```bash
python scripts/train_muc4_event_extraction.py \
  --base_model /path/to/base/model \
  --muc4_data_dir data/Processed \
  --output_dir muc4_event_extraction_model \
  --batch_size 8 \
  --micro_batch_size 2 \
  --num_epochs 10 \
  --learning_rate 2e-4 \
  --cutoff_len 2048
```


- If the expected CSVs do not exist, it runs `MUC4DataProcessor` to create them (multi-event only).
- Builds a `TrainingConfig` and calls `ModelTrainer` from `finetune.py`.
- Saves LoRA adapters, tokenizer/config, and metadata under e.g. `muc4_event_extraction_model/<base_model_name>/`.

#### 4. Batch inference on test data

After training, you can run batch inference with `scripts/batch_infer_muc4.py`:

```bash
python scripts/batch_infer_muc4.py \
  --model_root muc4_event_extraction_model \
  --base_model /path/to/base/model \
  --test_csv muc4_processed_multi/muc4_test.csv \
  --limit 0
```

Notes:
- If you pass `--model_root` and `--base_model`, the script will resolve the correct subdirectory under `model_root` using the sanitized base‑model name.
- By default, it writes `*_preds.jsonl` inside the resolved model directory.
- For **Qwen3** models, the pipeline appends `\no_think` after the instruction to disable thinking.

#### 5. Convert predictions into evaluation format

Use `convert_data/convert_preds.py` to transform the JSONL predictions into the format expected by the evaluation script:

```bash
python -m convert_data.convert_preds \
  --pred_dirs muc4_event_extraction_model \
  --pattern "*_preds.jsonl"
```

For each matched file, this will create a `*_predictions.jsonl` where each line has:

```json
{"id": "DOC_ID", "templates": [ {"id": "DOC_ID", "incident_type": "...", ...}, ... ]}
```

#### 6. Evaluate

Typical usage is:
- Use `batch_infer_muc4.py` to generate predictions.
- Use `convert_preds.py` to map them into eval JSONL.




#### Citation 
To cite our work:

```bibtex
@inproceedings{olsen-etal-2026-muc,
  title = {MUC-4 Revisited: Document-level Event Analysis beyond Span-based Arguments},
  author = {Olsen, Helene Bøsei and Velldal, Erik and Øvrelid, Lilja},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {7766--7780},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/2kaxd8nu44bx},
}
```
