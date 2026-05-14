"""
Converts MUC-4 dataset for fine-tuning for abstractive event analysis.
"""

import json
import pandas as pd
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Handle both relative and absolute imports for instructions
try:
    from ..instructions import MULTI_EVENT_INSTRUCTION
except ImportError:
    # Fallback to absolute import when run as a script or imported differently
    import sys
    import os as os_module
    # Add parent directory to path if needed
    parent_dir = os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from instructions import MULTI_EVENT_INSTRUCTION

@dataclass
class MUC4Event:
    """A single MUC-4 event template."""
    doc_id: str
    template_id: int
    incident_date: Optional[str] = None
    incident_location: Optional[str] = None
    incident_location_country: Optional[str] = None
    incident_location_city: Optional[str] = None
    incident_type: Optional[str] = None
    incident_stage_of_execution: Optional[str] = None
    incident_instrument_id: Optional[str] = None
    incident_instrument_type: Optional[str] = None
    perp_incident_category: Optional[str] = None
    perp_individual_id: Optional[str] = None
    perp_organization_id: Optional[str] = None
    perp_organization_confidence: Optional[str] = None
    phys_tgt_id: Optional[str] = None
    phys_tgt_type: Optional[str] = None
    phys_tgt_number: Optional[str] = None
    phys_tgt_effect_of_incident: Optional[str] = None
    hum_tgt_name: Optional[str] = None
    hum_tgt_description: Optional[str] = None
    hum_tgt_type: Optional[str] = None
    hum_tgt_number: Optional[str] = None
    hum_tgt_effect_of_incident: Optional[str] = None



class MUC4DataProcessor:
    """Processes MUC-4 dataset for fine-tuning."""
    
    def __init__(self, data_dir: str, json_format: str = "indent2", base_model: str = ""):
        """
        Initialise processor.
        
        Args:
            data_dir: Path to the MUC-4 data directory
            json_format: One of {"indent2", "compact"}
            base_model: Base model name (used for Gemma-specific text cleaning)
        """
        self.data_dir = data_dir
        self.splits = ["train", "val", "test"]
        self.json_format = json_format
        self.base_model = base_model
      
        self.field_name_map = {
            "incident_type": "event type",
            "incident_date": "date",
            "incident_location_country": "country",
            "incident_location_city": "city",
            "incident_stage_of_execution": "event stage",
            "incident_instrument_id": "weapon",
            "incident_instrument_type": "weapon type",
            "perp_incident_category": "perpetrator category",
            "perp_individual_id": "perpetrator individual",
            "perp_organization_id": "perpetrator organization",
            "perp_organization_confidence": "perpetrator confidence",
            "phys_tgt_id": "physical target",
            "phys_tgt_type": "physical target type",
            "phys_tgt_number": "physical target number",
            "phys_tgt_effect_of_incident": "effect on physical target",
            "hum_tgt_name": "victim name",
            "hum_tgt_description": "victim description",
            "hum_tgt_type": "victim type",
            "hum_tgt_number": "victim number",
            "hum_tgt_effect_of_incident": "effect on victim",
        }
    
        
    def load_processed_data(self, split: str) -> Dict[str, Any]:
        """Load processed MUC-4 data for a given split."""
        split_dir = os.path.join(self.data_dir, split)
        
        # Load texts
        with open(os.path.join(split_dir, f"{split}.json"), 'r') as f:
            texts = json.load(f)
        
        # Load keys/fields
        with open(os.path.join(split_dir, "keys.json"), 'r') as f:
            keys = json.load(f)
            
        return {"texts": texts, "keys": keys}
    
    def extract_field_value(self, field_data: Any) -> Optional[str]:
        """Extract string value from MUC-4 field data, handling multiple entities."""
        if field_data is None or field_data == "*" or field_data == "-":
            return None
            
        if isinstance(field_data, str):
            return field_data.strip().lower() if field_data.strip().lower() else None
            
        if isinstance(field_data, list):
            #Structured field data: collect all first values from each list
            values = []
            for item in field_data:
                if isinstance(item, dict):
                    # Take first string from strings array
                    if "strings" in item and item["strings"]: 
                        first_value = item["strings"][0]
                        if first_value and first_value.strip():
                            values.append(first_value.strip())
                    # Take first string from strings_lhs array
                    elif "strings_lhs" in item and item["strings_lhs"]:
                        first_value = item["strings_lhs"][0]
                        if first_value and first_value.strip():
                            values.append(first_value.strip())
                elif isinstance(item, str):
                    # Direct string value
                    if item and item.strip():
                        values.append(item.strip())
            
            # Return all values joined by semicolon, else just the first
            if len(values) > 1:
                return "; ".join(values)
            elif len(values) == 1:
                return values[0]
            else:
                return None
            
        elif isinstance(field_data, dict):
            # Dictionary with 'strings' field
            if "strings" in field_data and field_data["strings"]:
                return field_data["strings"][0].strip() if field_data["strings"][0].strip() else None
            elif "strings_lhs" in field_data and field_data["strings_lhs"]:
                return field_data["strings_lhs"][0].strip() if field_data["strings_lhs"][0].strip() else None
            
        return str(field_data).strip() if field_data is not None else None
    
    def parse_template(self, template_data: Dict[str, Any]) -> MUC4Event:
        """Parse a single MUC-4 template into structured event data."""
        doc_id = list(template_data.keys())[0]
        template = template_data[doc_id][0]
        
        event = MUC4Event(
            doc_id=doc_id,
            template_id=template.get("message_template", 1)
        )
        
        # Extract all fields from template
        for field_name in MUC4Event.__annotations__.keys():
            if field_name in ["doc_id", "template_id"]: # Skip id fields
                continue
                
            field_data = template.get(field_name)
            setattr(event, field_name, self.extract_field_value(field_data))
        
        return event
    

    def create_event_extraction_prompt(self, text: str, event: MUC4Event, tokenizer=None, max_length: int = 2048) -> Dict[str, str]:
        """Create instruction-following prompt for event extraction."""
        instruction = MULTI_EVENT_INSTRUCTION


        # Truncate input text to fit within max_length
        if tokenizer:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
            if len(prompt_tokens) > max_length - 128:  # Reserve 128 tokens for response
                # Estimate words per token (~0.75 for English BPE tokenizers)
                max_words = int((max_length - 128) * 0.75)
                text = ' '.join(text.split()[:max_words])
                print(f"Truncated doc_id {event.doc_id} to {max_words} words (was {len(prompt_tokens)} tokens)")

        # Build output JSON
        output_fields = {}
        for field_name in MUC4Event.__annotations__.keys():
            if field_name in ["doc_id", "template_id"]:
                continue
            value = getattr(event, field_name)
            if value is not None and value.strip():
                if isinstance(value, str) and value.startswith("{'type':"):
                    continue
                canonical_name = self.field_name_map.get(field_name, field_name)
                if self.base_model and "gemma" in self.base_model.lower():
                    value = value.replace('\n', ' ').replace('\r', ' ')
                    value = re.sub(r'\s+', ' ', value).strip()
                output_fields[canonical_name] = value.strip()
    
        output_json = json.dumps(output_fields, indent=2 if self.json_format != "compact" else None, separators=(",", ":") if self.json_format == "compact" else None, ensure_ascii=False)
    
        return {
            "doc_id": event.doc_id,
            "instruction": instruction,
            "input": text,
            "output": output_json
        }


    
    def process_dateline(self, dateline: str) -> str:
        """Process dateline to extract date and location information."""
        if not dateline:
            return ""
        
        # Remove parentheses and text inside parentheses from dateline
        cleaned_dateline = re.sub(r'\([^)]*\)', '', dateline)
        return cleaned_dateline.strip()
    
    def process_split(self, split: str) -> List[Dict[str, str]]:
        """
        Process a dataset split.
        
        Args:
            split: Dataset split name (train, val, test)
        """
        data = self.load_processed_data(split)
        texts = data["texts"]
        keys = data["keys"]
        training_data = []
        
        #For each document
        for doc_id, text_data in texts.items():
            if doc_id not in keys:
                continue
                
            # Get the original text and dateline
            text = text_data["text"]
            dateline = text_data.get("dateline", "")
            
            # Process dateline and add to text
            processed_dateline = self.process_dateline(dateline)
            if processed_dateline:
                text = f"{processed_dateline} {text}"
                text = text.lower()
            templates = keys[doc_id]
            
            # Filter templates based on incident_type
            valid_templates = []
            for template_data in templates:
                incident_type = self.extract_field_value(template_data.get("incident_type"))
                
                # Check if incident_type is valid
                if incident_type and incident_type not in ["*", "-", None, ""]:
                    valid_templates.append(template_data)
                elif len(templates) == 1:
                    # For single template documents, keep even if incident_type is empty
                    valid_templates.append(template_data)
            
            # If no valid templates, create a minimal template for the document
            if not valid_templates:
                # Create a minimal template with incident_type set to None
                minimal_template = {
                    "message_template": 1,
                    "incident_type": None
                }
                valid_templates = [minimal_template]
            
            # Always create a single example per document; output is a JSON array of all valid templates.
            events_list = []
            for template_data in valid_templates:
                event = self.parse_template({doc_id: [template_data]})
                if self._is_empty_template(event):
                    event.incident_type = None
                output_fields = {}
                for field_name in MUC4Event.__annotations__.keys():
                    if field_name in ["doc_id", "template_id"]:
                        continue
                    value = getattr(event, field_name)
                    if value is not None and value.strip():
                        if isinstance(value, str) and value.startswith("{'type':"):
                            continue
                        canonical_name = self.field_name_map.get(field_name, field_name)
                        output_fields[canonical_name] = value.strip() if isinstance(value, str) else value
                events_list.append(output_fields)

            output_json = (
                json.dumps(events_list, separators=(",", ":"), ensure_ascii=False)
                if self.json_format == "compact"
                else json.dumps(events_list, indent=2, ensure_ascii=False)
            )

            prompt_data = {
                "doc_id": doc_id,
                "instruction": MULTI_EVENT_INSTRUCTION,
                "input": text,
                "output": output_json,
            }
            training_data.append(prompt_data)
        
        return training_data
    
    def _is_empty_template(self, event: MUC4Event) -> bool:
        """Check if a template is empty (no meaningful information)."""
        for field_name in MUC4Event.__annotations__.keys():
            if field_name in ["doc_id", "template_id"]:
                continue
                
            value = getattr(event, field_name)
            if value is not None and value.strip():
                return False
        return True
    
    def save_to_csv(self, split: str, output_path: str):
        """Process a split and save to CSV format (one row per document)."""
        training_data = self.process_split(split)
        
        df = pd.DataFrame(training_data)
        df.to_csv(output_path, index=False)
        
        print(f"Processed {len(training_data)} examples for {split} split")
        print(f"Saved to {output_path}")
        
        return len(training_data)
    
    def process_all_splits(self, output_dir: str):
        """
        Process all splits and save to CSV files.
        
        Args:
            output_dir: Directory to save processed data
        """
        # Convert to absolute path for clarity
        abs_output_dir = os.path.abspath(output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        print(f"Created/verified output directory: {abs_output_dir}")
        output_dir = abs_output_dir  # Use absolute path for rest of function
        
        total_examples = 0
        for split in self.splits:
            output_path = os.path.join(output_dir, f"muc4_{split}.csv")
            examples = self.save_to_csv(split, output_path)
            total_examples += examples
        
        print(f"Total examples processed: {total_examples}")
        
        # Create a combined training file
        all_data = []
        for split in self.splits:
            split_data = self.process_split(split)
            all_data.extend(split_data)
        
        combined_path = os.path.join(output_dir, "muc4_combined.csv")
        combined_df = pd.DataFrame(all_data)
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined dataset saved to {combined_path}")
        
        # Print statistics
        self._print_statistics(all_data)
    
    def _print_statistics(self, all_data: List[Dict[str, str]]):
        """Print statistics about the processed data."""
        print(f"\nData Statistics:")
        print(f"  Total examples: {len(all_data)}")
        
        # Count documents with multiple templates
        doc_counts = {}
        for item in all_data:
            # Extract a rough document identifier from the input
            input_text = item["input"]
            if "DEV-MUC3-" in input_text or "TEST-MUC3-" in input_text:
                # Extract document ID from input text
                words = input_text.split()
                for word in words:
                    if word.startswith(("DEV-MUC3-", "TEST-MUC3-")):
                        doc_counts[word] = doc_counts.get(word, 0) + 1
                        break
        
        if doc_counts:
            multi_template_docs = sum(1 for count in doc_counts.values() if count > 1)
            total_docs = len(doc_counts)
            print(f"  Unique documents: {total_docs}")
            print(f"  Documents with multiple templates: {multi_template_docs}")
            print(f"  Average templates per document: {len(all_data) / total_docs:.2f}")
        
        # Check JSON validity
        valid_json_count = 0
        for item in all_data:
            try:
                json.loads(item["output"])
                valid_json_count += 1
            except json.JSONDecodeError:
                pass
        
        print(f"  Valid JSON outputs: {valid_json_count}/{len(all_data)} ({valid_json_count/len(all_data)*100:.1f}%)")


def main():
    """Main function to process MUC-4 data."""
   
    data_dir = "../data/Processed/"
    output_dir = "muc4_processed"
    # Set output_dir within "../data/"
    output_dir = os.path.join("../data", output_dir)
    # Check if output directory exists
    if os.path.exists(output_dir):
        print(f"Output directory {output_dir} already exists")
    else:
        print(f"Output directory {output_dir} does not exist")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory {output_dir}")
    
    processor = MUC4DataProcessor(data_dir)
    
    # Process with multiple events per document (default)
    print("Processing MUC-4 data with multiple templates per document...")
    processor.process_all_splits(output_dir, multievent=True)
    
    # Create single-event version for comparison
    single_output_dir = "muc4_processed_single"
    print("\nProcessing MUC-4 data with single template per document...")
    processor.process_all_splits(single_output_dir, multievent=False)


if __name__ == "__main__":
    main() 
