# Create all necessary directories first
mkdir -p "../Processed/train"
mkdir -p "../Processed/test"
mkdir -p "../Processed/val"

echo "Created output directories"

echo "Process train split"
python3 text_proc.py --input "../Original/train/docs" --output "../Processed" --split "train" 
python3 preprocess_muc.py --input "../Original/train/keys" --output "../Processed/train/keys.json" --event_type "ATTACK, BOMBING" --dataset_type "modified" --incident_location_type "city"

echo "Process val split"
python3 text_proc.py --input "../Original/val/docs" --output "../Processed" --split "val" 
python3 preprocess_muc.py --input "../Original/val/keys" --output "../Processed/val/keys.json" --event_type "ATTACK, BOMBING" --dataset_type "modified" --incident_location_type "city"

echo "Process test split"
python3 text_proc.py --input "../Original/test/docs" --output "../Processed" --split "test" 
python3 preprocess_muc.py --input "../Original/test/keys" --output "../Processed/test/keys.json" --event_type "ATTACK, BOMBING" --dataset_type "modified" --incident_location_type "city"
