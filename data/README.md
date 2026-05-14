# Data layout

## Obtaining MUC-4

The original **MUC-3 / MUC-4** corpora (and related MUC releases) are distributed by NIST. You can find downloads and licensing notes on the **[MUC data index](https://www-nlpir.nist.gov/related_projects/muc/muc_data/muc_data_index.html)**. Place unpacked text and key files under **`Original/`** in the layout below before running preprocessing.

## Folder structure

| Path | Purpose |
| --- | --- |
| **`Original/{train,val,test}/docs/`** | Raw MUC **document** files for that split. |
| **`Original/{train,val,test}/keys/`** | Raw MUC **key** (template) files for that split. |
| **`Processed/{train,val,test}/`** | JSON produced by the preprocessing scripts (e.g. `train.json`, `keys.json`). |
| **`preprocessing/`** | `text_proc.py`, `preprocess_muc.py`, `run_preprocessing.sh`, and the [preprocessing README](preprocessing/README.md). |

From `preprocessing/`, run **`run_preprocessing.sh`** to build **`Processed/`** from **`Original/`** (see that README for flags and the paper dataset settings).
