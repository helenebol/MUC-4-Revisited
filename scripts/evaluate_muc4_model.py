#!/usr/bin/env python3
"""
This script evaluates a trained model on the modified version of MUC-4
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from transformers import BitsAndBytesConfig
import re
from datasets import load_dataset


# Add both current directory (scripts/) and parent directory (eventMUC/) to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)  # scripts/ for local imports
sys.path.insert(0, parent_dir)   # eventMUC/ for instructions



from finetune import SimplePrompter
from instructions import MULTI_EVENT_INSTRUCTION


def _sanitise_model_name(model_id: str) -> str:
    """Create a filesystem-friendly name from a base model id or path."""
    name = os.path.basename(str(model_id).rstrip("/")) if "/" in str(model_id) else str(model_id)
    if "/" in name:
        name = name.split("/")[-1]
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return safe or "model"

class MUC4EventExtractor:
    """Class for extracting events from text ."""
    
    def __init__(
            self, 
            model_path: str, 
            base_model: str = None, 
            multi_event: bool = True,
            ):
        """
        Initialise the model.
        
        Args:
            model_path: Path to the fine-tuned model
            base_model: Base model name (if different from the one used in training)
        """
        self.model_path = model_path
        self.base_model = base_model
        self.multi_event = True
        self.last_instruction = None
        
        # correct the model_path
       
        if self.model_path and os.path.isdir(self.model_path) and self.base_model:
            adapter_file = os.path.join(self.model_path, "adapter_config.json")
            print(f"Adapter file: {adapter_file}")
            if not os.path.exists(adapter_file):
                subdir = _sanitise_model_name(self.base_model)
                candidate = os.path.join(self.model_path, subdir)
                if os.path.exists(os.path.join(candidate, "adapter_config.json")):
                    print(f"Detected root path; using subdir: {candidate}")
                    self.model_path = candidate
       
        
        # Load tokenizer and model
        self._load_model()
        self.prompter = SimplePrompter()
    
        
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}")

        ## Load tokenizer
        if self.base_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Configure tokeniser 
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # Log tokeniser 
        print(f"Tokenizer: pad_token_id={self.tokenizer.pad_token_id}, eos_token_id={self.tokenizer.eos_token_id}, unk_token_id={self.tokenizer.unk_token_id}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
            bnb_4bit_quant_type="nf4",  # NormalFloat4 for better precision
            bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
        )
        load_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",                   
        "dtype": torch.float16,        
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        }
        
        if ("olmo" in str(self.base_model or self.model_path).lower()):
            load_kwargs["attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model or self.model_path,
            **load_kwargs
        )

        # Load LoRA weights
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            print("Loading LoRA adapter weights...")
            self.model = PeftModel.from_pretrained(self.model, self.model_path, device_map="auto", torch_dtype=torch.float16, is_trainable=False)
             
        self.model.eval() 
         
    
        
    def parse_generated_json(generated_text: str, multi_event: bool = True) -> list:
        """Parse JSON events from generated text, (mixed formats)."""
        generated_text = generated_text.strip()
        
        # Clean up 
        generated_text = re.sub(r'^,+|,+$', '', generated_text)  # Remove leading/trailing commas
        generated_text = re.sub(r'\s+', ' ', generated_text)     # Normalize whitespace
        
        # 1 Find and extract complete JSON array
        array_match = re.search(r'\[.*\]', generated_text, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            try:
                events = json.loads(json_str)
                if isinstance(events, list):
                    return [e for e in events if isinstance(e, dict) and e]  # Filter valid event dicts
            except json.JSONDecodeError:
                pass
        
        #2: Extract individual JSON objects and wrap in array (for multi_event)
        if multi_event:
            # Find all JSON objects
            object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            objects = re.findall(object_pattern, generated_text)
            events = []
            for obj_str in objects:
                try:
                    event = json.loads(obj_str)
                    if isinstance(event, dict) and event:  # Valid non-empty event
                        events.append(event)
                except json.JSONDecodeError:
                    continue
            
            if events:
                return events
        
        # 3: Fallback single object parsing (last resort)
        start_idx = generated_text.find('{')
        if start_idx != -1:
            # Find the last complete object
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(generated_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_str = generated_text[start_idx:end_idx]
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict) and obj:
                        return [obj] if multi_event else obj
                except json.JSONDecodeError:
                    print("ERROR could not find valid JSON")
    
        #4: No valid JSON found
        print(f" No valid JSON events extracted from: {generated_text[:600]}...")
        return [] if multi_event else { "No valid JSON found", "raw_response": generated_text}
    
    
    def extract_events(self, text: str):
        """
        Extract events from text.
        
        Args:
            text: Input text to extract events from
            
        Returns:
            Dictionary containing extracted event information
        """
        
        
        def _maybe_add_no_think(instruction_text: str) -> str:
            """Qwen3 models support a '\\no_think' control token to disable thinking output."""
            base = str(self.base_model or self.model_path or "").lower()
            if "qwen3" in base:
                return f"{instruction_text}\n\\no_think"
            return str(instruction_text)

        # Multi-event only
        instruction = _maybe_add_no_think(MULTI_EVENT_INSTRUCTION)
        # Save the instruction used 
        self.last_instruction = instruction

        
        # Simple prompt using SimplePrompter
        prompt = self.prompter.generate_prompt(instruction, text)
        #print(f"Prompt (first 100 chars): {prompt[:500]}... (length: {len(prompt)} chars)")
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(self.tokenizer.model_max_length - 256, 7936),  # Reserve space for generation
            padding=True
        )
      
        
        # Check input length
        input_length = inputs["input_ids"].shape[1]
        max_ctx = getattr(self.tokenizer, "model_max_length", None) or getattr(self.model.config, "max_position_embeddings", 4096)
        if input_length >= max_ctx:
            print(f"Warning: Input length {input_length} exceeds or equals model max length {max_ctx}")
        
        device = None
        if hasattr(self.model, "device"):
            device = self.model.device
        else:
            print('device err')

        inputs = {k: v.to(device) for k, v in inputs.items()}
     
            

        def parse_generated_json(generated_text: str, multi_event: bool = True) -> list:
            """ parse JSON text"""
            generated_text = generated_text.strip()
            
            # Clean up common generation artifacts
            generated_text = re.sub(r'^,+|,+$', '', generated_text)  # Remove leading/trailing commas
            generated_text = re.sub(r'\s+', ' ', generated_text)     # Normalize whitespace
            
            # 1 Find and extract complete JSON array
            array_match = re.search(r'\[.*\]', generated_text, re.DOTALL)
            if array_match:
                json_str = array_match.group(0)
                try:
                    events = json.loads(json_str)
                    if isinstance(events, list):
                        return [e for e in events if isinstance(e, dict) and e]  # Filter valid event dicts
                except json.JSONDecodeError:
                    pass
            
            #  2: Extract individual JSON objects and wrap
            if multi_event:
                # Find all complete JSON objects
                object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                objects = re.findall(object_pattern, generated_text)
                events = []
                for obj_str in objects:
                    try:
                        event = json.loads(obj_str)
                        if isinstance(event, dict) and event:  # Valid non-empty event
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
                
                if events:
                    return events
            
            # 3: Fallback to single object parsing 
            start_idx = generated_text.find('{')
            if start_idx != -1:
                # Find the last complete object
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(generated_text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    try:
                        obj = json.loads(json_str)
                        if isinstance(obj, dict) and obj:
                            return [obj] if multi_event else obj
                    except json.JSONDecodeError:
                        pass
            
            #  4: No valid JSON found
            print(f"Warning: No valid JSON events extracted from: {generated_text[:200]}...")
            return [] if multi_event else {"error": "No valid JSON found", "raw_response": generated_text}


        generation_config = {
            "max_new_tokens": 3072,
            "do_sample": False,      
            "num_beams": 5,
            "repetition_penalty": 1.1,
            "early_stopping": False,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
           
        }

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                )
                
            # Extract generated token IDs (excluding input prompt)
            # Handle different output formats from model.generate()
            if isinstance(outputs, torch.Tensor):
                full_output_ids = outputs[0]
            elif hasattr(outputs, 'sequences'):
                full_output_ids = outputs.sequences[0]
            elif isinstance(outputs, (list, tuple)):
                full_output_ids = outputs[0]
            else:
                full_output_ids = outputs
            
            # Tensor is on CPU
            if isinstance(full_output_ids, torch.Tensor):
                full_output_ids = full_output_ids.cpu()
            
            input_length = inputs["input_ids"].shape[1]
            full_length = len(full_output_ids) if not isinstance(full_output_ids, torch.Tensor) else full_output_ids.shape[0]
            
            # Validate generation
            if full_length <= input_length:
                print(f"Warning: No generated tokens (input_length={input_length}, full_length={full_length})")
                return [] if self.multi_event else {"error": "No tokens generated", "raw_response": ""}
            
            # Extract only the generated tokens (after input)
            if isinstance(full_output_ids, torch.Tensor):
                output_ids = full_output_ids[input_length:].tolist()
            else:
                output_ids = full_output_ids[input_length:]
            
            # Validate output_ids
            if not output_ids or len(output_ids) == 0:
                print(f"Warning: Empty output_ids after extraction")
                return [] if self.multi_event else {"error": "Empty output", "raw_response": ""}
            
            generated_text = None
            
            # Decode full output and extract the generated portion (after the prompt) when possible.
            full_decoded = self.tokenizer.decode(full_output_ids, skip_special_tokens=True)
            prompt_decoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if full_decoded.startswith(prompt_decoded):
                generated_text = full_decoded[len(prompt_decoded):].strip()
            else:
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            try:
                if self.multi_event:
                    extracted_events = parse_generated_json(generated_text, multi_event=True)
                    return extracted_events
                else:
                    extracted_events = parse_generated_json(generated_text, multi_event=False)
                    return extracted_events[0] if extracted_events else {"error": "No valid JSON found", "raw_response": generated_text}
            except Exception as e:
                print(f"Error in JSON parsing: {e}, raw response: {generated_text[:200]}...")
                return [] if self.multi_event else {"error": f"JSON parsing failed: {e}", "raw_response": generated_text}
         

        except (RuntimeError, ValueError, TypeError) as e:
            print(f"Generation error ({type(e).__name__}): {e}. Falling back to greedy decoding.")
            generation_config["do_sample"] = False
            fallback_config = {k: v for k, v in generation_config.items() if k != "logits_processor"}
            # --- DEBUG START ---
            print("\n===== DEBUG GENERATION STATE =====")
            print("Prompt head:\n", prompt[:400])
            print("Input shape:", inputs["input_ids"].shape)
            print("Input device:", inputs["input_ids"].device)

            try:
                model_device = next(
                    p.device for p in self.model.parameters()
                    if p.device.type != "meta"
                )
                print("Model param device:", model_device)
            except StopIteration:
                print("Model param device: not found")

            print("Max new tokens:", generation_config.get("max_new_tokens"))
            print("Do sample:", generation_config.get("do_sample"))
            print("==================================\n")
            # --- DEBUG END ---

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **fallback_config,
                )
                full_output_ids = outputs[0] if isinstance(outputs, tuple) or isinstance(outputs, list) else outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0]
                input_length = inputs["input_ids"].shape[1]
                output_ids = full_output_ids[input_length:].tolist()
                generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                try:
                    if self.multi_event:
                        extracted_events = parse_generated_json(generated_text, multi_event=True)
                        return extracted_events
                    else:
                        extracted_events = parse_generated_json(generated_text, multi_event=False)
                        return extracted_events[0] if extracted_events else {"error": "No valid JSON found", "raw_response": generated_text}
                except Exception as e:
                    print(f"Error in JSON parsing (greedy): {e}, raw response: {generated_text[:500]}...")
                    return [] if self.multi_event else {"error": f"JSON parsing failed (greedy): {e}", "raw_response": generated_text}
       
    
    


def main():
    parser = argparse.ArgumentParser(description="Evaluate abstractive event analysis model")
    
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to the fine-tuned model (adapter dir). If omitted, use --model_root + --base_model")
    parser.add_argument("--model_root", type=str, default=None,
                       help="Root directory containing per-base-model subdirs (e.g., muc4_event_extraction_model)")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (if different from training)")
    
    parser.add_argument("--multi_event", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--sample_text", type=str, default=None,
                       help="Sample text to evaluate on")
    
    parser.add_argument("--test_csv", type=str, default=None,
                       help="If provided, run batch predictions over this CSV (expects input[,output,doc_id])")
    parser.add_argument("--output_path", type=str, default="predictions/muc4_test_preds.jsonl",
                       help="Where to write JSONL predictions. If default, will save under model dir")
    parser.add_argument("--limit", type=int, default=0,
                       help="Limit number of rows to process in batch mode (0 = all)")
    args = parser.parse_args()
    
    # Resolve model path: prefer explicit --model_path. else derive from root/base_model
    resolved_model_path = args.model_path
    if resolved_model_path is None:
        if not args.model_root or not args.base_model:
            raise ValueError("Provide either --model_path or both --model_root and --base_model")
        subdir = _sanitise_model_name(args.base_model)
        resolved_model_path = os.path.join(args.model_root, subdir)
    print(f"Resolved model adapter path: {resolved_model_path}")

    # initialise muc model
    extractor = MUC4EventExtractor(
            resolved_model_path, 
            args.base_model, multi_event=True)
    
    # Batch mode
    print("starting eval")
    if args.test_csv:
        # Place predictions inside the resolved model path
        if args.output_path in ("predictions/muc4_test_preds.jsonl", "../predictions/muc4_test_preds.jsonl"):
            test_name = os.path.splitext(os.path.basename(args.test_csv))[0]
            args.output_path = os.path.join(resolved_model_path, f"{test_name}_preds.jsonl")
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        if not os.path.exists(args.test_csv):
            raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")
     
     
     
        test_ds = load_dataset("csv", data_files=args.test_csv, split="train")
        if "input" not in test_ds.column_names:
            raise ValueError("CSV must contain 'input' column for batch mode")
        num_rows = len(test_ds) if args.limit <= 0 else min(args.limit, len(test_ds))
        with open(args.output_path, "w") as fout:
            for idx in range(num_rows):
                row = test_ds[idx]
                input_text = row["input"]
                pred = extractor.extract_events(input_text)
                record = {
                    "doc_id": row.get("doc_id", f"doc_{idx}"),
                    "instruction": extractor.last_instruction,
                    "input": input_text,
                    "gold_output": row.get("output", None),
                    "pred_output": pred,
                }
                fout.write(json.dumps(record) + "\n")
        print(f"Wrote {num_rows} predictions to {args.output_path}")
        return



if __name__ == "__main__":
    main() 
