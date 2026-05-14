# MUC-4 preprocessing

These scripts turn raw MUC-4 **document** and **key** files into JSON used downstream. `text_proc.py` builds one JSON of all texts per split; `preprocess_muc.py` builds one JSON of all templates (keys) per split.

**Paper dataset.** To obtain the **modified dataset** used in this paper, run **`run_preprocessing.sh`** from this directory (`data/preprocessing`; see [Usage](#usage)). That pipeline uses **`--dataset_type modified`** (reduced slot set), **`--event_type "ATTACK, BOMBING"`** so only **attack** and **bombing** incidents get a full template parse (other incident types are still in the JSON as shortened stubs), and **`--incident_location_type "city"`** so location is split into two separate keys, **`incident_location_country`** and **`incident_location_city`**, instead of one **`incident_location`** list.

**Lineage.** **`preprocess_muc.py`** is an adaptation of **[wgantt/mucd `proc_keys.py`](https://github.com/wgantt/mucd/blob/main/scripts/preprocessing/proc_keys.py)**, which is itself an adaptation of **[brendano/muc4_proc `proc_keys.py`](https://github.com/brendano/muc4_proc/blob/master/scripts/proc_keys.py)**. **`text_proc.py`** follows **[muc4_proc `proc_texts.py`](https://github.com/brendano/muc4_proc/blob/master/scripts/proc_texts.py)** directly (same upstream family; not routed through mucd). Details of what this repo adds on top of mucd are under [Modifications in this repo](#modifications-in-this-repo).

---

## Modifications in this repo


1. **Excluding event types** — Optional **`--event_type`** (comma-separated allow-list). Incidents whose `INCIDENT: TYPE` is **not** in that list are still written to JSON, but through **`parse_values_exclude_event`**: a **short stub** (e.g. **`incident_type`** set to **`*`**, far fewer slots) instead of a full parsed template. 
2. **Removing / subsetting fields** — Optional **`--dataset_type`**: **`extractive`** keeps only five slots (`EXTRACTIVE_FIELDS`), **`modified`** uses a reduced slot list (`MODIFIED_KEYS`), and omitting the flag uses **`SELECTED_KEYS`**. This is how you **drop fields** from the output relative to a full key parse.
3. **Two-level location (country vs city)** — Optional **`--incident_location_type city`**. Instead of a single **`incident_location`** list from **`parse_location`**, the script fills **`incident_location_country`** and **`incident_location_city`** via **`parse_country_location`** / **`parse_city_location`**. Mucd only exposes the flat **`incident_location`** list.

Smaller differences: named CLI flags (**`--input`**, **`--output`**, etc.) instead of mucd positional arguments; **`parse_one_value`** colon handling records **`strings`** from the left side only (mucd keeps **`colon_clause`** with **`strings_lhs`** / **`strings_rhs`** for generic slots); **`message_template_optional`** is intentionally de-emphasised.



### Pipeline

- Key parsing: **`preprocess_muc.py`**.
- Splits are run via **`run_preprocessing.sh`**.
- Default layout: raw **`../Original/{split}/`**, processed **`../Processed/{split}/`** (see [Usage](#usage)).

---

## text_proc.py

Processes raw **text** files (single file or a directory of files) and writes **one JSON object** where keys are document IDs and values are per-document records.

### CLI

- `--input` — path to a MUC text file or a directory of text files  
- `--output` — root output directory (the script creates `{output}/{split}/`)  
- `--split` — split name (e.g. `train`, `val`, `test`; used for the subdirectory and filename)

### Behaviour

- **Document boundaries** — Finds document headers with a regex. If lines match `(DEV-\S+) *\(([^\)]*)\)`, each match supplies `docid` and `source` (parenthetical “source” after the ID). Otherwise matches `TST\d+-\S+` (test IDs without that parenthetical).
- **Spans** — For each document, `char_start` / `char_end` delimit the slice of the file belonging to that doc (using `char_before` from the next header for the end of the previous doc).
- **Body parsing** — Inside each slice, expects: **dateline** `--` **one or more `[TAG]` tokens** **rest of text**. If that pattern is missing, the script prints diagnostics and aborts (`assert False`).
- **Dateline** — Newlines in the dateline segment are replaced with spaces, then trimmed.
- **Tags** — Only the bracketed segment is used; each `[...]` becomes a string; brackets are stripped and tag text is **lowercased** (MUC tags are upper case in the source).
- **Main text** — Leading/trailing whitespace stripped; remaining **`[` → `(`** and **`]` → `)`** in the story text (square brackets in the body are normalised to parentheses).

### Each document value in the output JSON

Keys are `docid` strings (e.g. `DEV-MUC3-0001`). Each value is an object like:

```json
{
  "docid": "DEV-MUC3-0001",
  "char_start": 123,
  "char_before": 100,
  "char_end": 4567,
  "source": "OPTIONAL_SOURCE_FROM_HEADER",
  "dateline": "LOCATION DATE",
  "tags": ["text", "report"],
  "text": "Cleaned document text with square brackets mapped to parentheses."
}
```

Notes:

- `source` is present only when the DEV-style header with parentheses was matched.  
- `char_before` is the start index of the header line in the file (used internally to chain `char_end`).  
- If the test-style ID regex is used, there is no `source` field from that path.

### Output path

Writes:

`{output}/{split}/{split}.json`

for example `../Processed/train/train.json` when `--output ../Processed` and `--split train`.

---

## preprocess_muc.py

Processes raw **key** files (one file or a directory of files whose names start with `key-`), parses each **template** (blank-line–separated chunks in the MUC key format), and writes **one JSON object** mapping **message ID** → **list of template objects**. This file is an **adaptation of [wgantt/mucd `proc_keys.py`](https://github.com/wgantt/mucd/blob/main/scripts/preprocessing/proc_keys.py)**, which is itself an **adaptation of [brendano/muc4_proc `proc_keys.py`](https://github.com/brendano/muc4_proc/blob/master/scripts/proc_keys.py)**. On top of mucd, this repo mainly adds **event-type exclusion**, **field removal / slot subsets**, and **country–city location split** — see [Modifications in this repo](#modifications-in-this-repo).

### CLI

- `--input` — key file or directory of `key-*` files  
- `--output` — path to the JSON file to write  
- `--dataset_type` — optional; `extractive` (five extractive slots), `modified` (a reduced slot set), or omit for the default selected slots  
- `--event_type` — optional; comma-separated incident types to **keep** as full events (e.g. `ATTACK, BOMBING`). Documents whose `INCIDENT: TYPE` is not in this set are still emitted but with a shortened “excluded event” representation (`incident_type` becomes `*`, etc.). If omitted, a built-in list of incident types is used.  
- `--incident_location_type` — optional; if set to `city`, `INCIDENT: LOCATION` is not emitted as a single `incident_location` list; instead the parser yields **`incident_location_country`** and **`incident_location_city`** (see code for parsing rules).

### Cleaning and parsing

- **Field names** — Non-letters in the MUC key labels are turned into underscores and the name is lowercased, e.g. `INCIDENT: TYPE` → `incident_type`.
- **Message ID** — Parenthetical suffixes on the ID line are stripped for the stored `message_id`.
- **Skipped / null values** — A value of `*` means “omit this field” for list-valued slots. A value of `-` is stored as JSON `null` where applicable.
- **Quoted strings and alternations** — Slot values are parsed from the MUC quoting and `/` alternation conventions; some slots have allowed **closed vocabularies** (`SET_FILL_KEYS_ALLOWED_VALUES`) and are validated.
- **Incident location (default)** — `parse_location` can emit entries with only `strings`, or objects with `type`, `strings_lhs`, and `strings_rhs` for colon-like location pieces (see `parse_location` in the script).
- **Colon in generic slot values** — `parse_one_value` treats a single `:` in the value as splitting left/right but currently records structured content using a **`strings`** list derived from the left side (see `parse_one_value` for edge-case fixes such as specific doc IDs).

### Output shape

Top level: object keyed by **cleaned message ID**. Each value is a **list** of template dicts (one list element per template for that document). Field names correspond to cleaned MUC keys; most slot values are **lists of objects** with a `strings` key (and sometimes `type`, `optional`, etc., depending on the slot and parser branch).

The default **`run_preprocessing.sh`** invocation uses **`--event_type "ATTACK, BOMBING"`** and **`--incident_location_type "city"`** together with **`--dataset_type "modified"`**. In that setup:

- **Full templates** (rich slots, real `incident_type`) appear only when the key file’s incident type string is in the **`--event_type`** allow-list. With **`"ATTACK, BOMBING"`** (as in `run_preprocessing.sh`), that is exactly those two labels—not variants such as **`ATTACK / BOMBING`** unless you add them to the comma-separated flag. Other incident types still appear under the same `message_id` but as **shortened “excluded” records** (`incident_type` **`*`**, fewer fields—see `parse_values_exclude_event` in the script).
- **Location** is **not** a single `incident_location` list. The parser emits **`incident_location_country`** and **`incident_location_city`** separately. Values are whatever **`parse_country_location`** / **`parse_city_location`** return (lists of `{"strings": [...]}` objects, or **`null`** for city when nothing is extracted).

Example skeleton for a ATTACK/BOMBING template under the default shell flags (illustrative only; real keys include the rest of `MODIFIED_KEYS`):

```json
{
  "DEV-MUC3-0001": [
    {
      "message_id": "DEV-MUC3-0001",
      "message_template": 1,
      "incident_type": "BOMBING",
      "incident_date": [
        { "strings": ["DATE_STRING"] }
      ],
      "incident_location_country": [
        { "strings": ["COUNTRY"] }
      ],
      "incident_location_city": [
        { "strings": ["CITY_NAME"] }
      ]
    }
  ]
}
```

If you omit **`--incident_location_type city`**, you get a single **`incident_location`** list built by **`parse_location`** (including `colon_clause` objects where applicable). If you omit **`--event_type`**, a larger built-in set of incident types is treated as full events (see `INCIDENT_TYPES` in the script).

---

## Usage

From `data/preprocessing`, `run_preprocessing.sh` runs **both** scripts for **train**, **val**, and **test**:

```bash
bash run_preprocessing.sh
```

That script:

1. Creates `../Processed/train`, `../Processed/test`, and `../Processed/val`.  
2. For each split, runs `text_proc.py` on `../Original/{split}/docs` and `preprocess_muc.py` on `../Original/{split}/keys`, writing e.g. `../Processed/train/train.json` and `../Processed/train/keys.json`.

Adjust paths or flags in `run_preprocessing.sh` if your raw tree or desired event types / dataset type differ.
