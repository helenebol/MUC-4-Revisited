"""
Fine-tuning script for llms using LoRA (Low-Rank Adaptation).

"""
# imports 
import os
import sys
import json
import re
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import fire
import pandas as pd
import torch
import transformers
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    EarlyStoppingCallback,
    
)
from transformers import Trainer as HfTrainer
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from safetensors.torch import save_file
try:
    from safetensors.torch import load_file as safe_load_file  # type: ignore
except Exception:
    safe_load_file = None

try:
    import wandb  # noqa: F401
except Exception:
    pass


@dataclass
class TrainingConfig: 
    # Model and data parameters
    base_model: str = ""
    train_data_path: str = ""
    val_data_path: str = ""
    output_dir: str = ""
    
    # Training hyperparameters
    batch_size: int = 128
    micro_batch_size: int = 8
    num_epochs: int = 1
    learning_rate: float = 3e-4
    cutoff_len: int = 4096
    val_set_size: int = 0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    seed: int =32
    
    
    # LoRA hyperparameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # LLM hyperparameters
    train_on_inputs: bool = False 
    add_eos_token: bool = False 
    group_by_length: bool = False
    
    # Other parameters
    resume_from_checkpoint: Optional[str] = None
    prompt_template_name: str = "alpaca"

    #early stopping 
    early_stopping: bool = True
    early_stopping_patience: int = 4          # number of evals with no improvement
    early_stopping_threshold: float = 0.001     # minimum improvement to count
    early_stopping_metric: str = "eval_loss"

class SavePeftModelCallback(TrainerCallback):
    """Callback to save PEFT model during training."""
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.local_rank not in [-1, 0]:
            return control
        
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        os.makedirs(checkpoint_folder, exist_ok=True)
        model = kwargs["model"]
        
        try:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            #save adapter weights
            adapter_state_dict = get_peft_model_state_dict(model)
            safe_path = os.path.join(checkpoint_folder, "adapter_model.safetensors")
            save_file(adapter_state_dict, safe_path)
            print(f"Saved adapter weights to {safe_path}")
       
            
            #save full adapter config and tokeniser
            model.save_pretrained(checkpoint_folder, save_adapter=True, save_config=True)
            
            if hasattr(model, 'tokenizer'):
                model.tokenizer.save_pretrained(checkpoint_folder)
            if hasattr(model, 'config'):
                model.config.save_pretrained(checkpoint_folder)
            
            all_files = os.listdir(checkpoint_folder)
            adapter_files = [f for f in all_files if 'adapter' in f.lower() or 'lora' in f.lower()]
            
            if not adapter_files:
                print(f"WARNING: No adapter files found in {checkpoint_folder}")
            else:
                print(f"Checkpoint saved at {checkpoint_folder} with files: {all_files}")
                print(f"Adapter-specific files: {adapter_files}")
        except Exception as e:
            print(f"Error saving checkpoint at {checkpoint_folder}: {e}")
        return control
   

class LoadBestPeftModelCallback(TrainerCallback):
    """Callback to load the best PEFT model at the end of training."""
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.local_rank not in [-1, 0]:
            return control
        if state.best_model_checkpoint is None:
            print("Warning: No best checkpoint found (best_model_checkpoint is None). Skipping adapter restore.")
            return control
        print(f"Loading best PEFT model from {state.best_model_checkpoint} (score: {state.best_metric})")
        possible_files = [
            os.path.join(state.best_model_checkpoint, "adapter_model.safetensors"),
            os.path.join(state.best_model_checkpoint, "adapter_model.bin"),
        ]
        adapters_weights = None
        for adapter_file in possible_files:
            if os.path.exists(adapter_file):
                try:
                    if adapter_file.endswith('.safetensors') and safe_load_file:
                        adapters_weights = safe_load_file(adapter_file)
                        print(f"Loaded adapter weights from {adapter_file}")
                        break
                    elif adapter_file.endswith('.bin'):
                        adapters_weights = torch.load(adapter_file, weights_only=True, map_location="cpu")
                        print(f"Loaded adapter weights from {adapter_file}")
                        break
                except Exception as e:
                    print(f"Error loading {adapter_file}: {e}")
                    continue
        if adapters_weights is None:
            print(f" No adapter weights found in {state.best_model_checkpoint}. Skipping adapter restore.")
            return control
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        print("Restored best adapter weights")
        return control
    


class MyTrainer(HfTrainer):
    
    """
    Ensure labels are forwarded into the model forward for PEFT-wrapped models.
    
    """
    def __init__(self, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]  # Explicitly set label_names for PEFT models
        self.tokenizer = tokenizer
       
        self.processing_class = self.tokenizer  # Set processing_class to same object (avoids deprecation warning)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Set generation_config 
        self.generation_config = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items()}
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            loss = outputs["loss"]
        if return_outputs:
            return loss, outputs
        return loss
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        model = self.model
        tokenizer = self.processing_class if hasattr(self, 'processing_class') and self.processing_class is not None else self.tokenizer
        eval_dataset = self.eval_dataset
        expected_fields = {
            "event type", "date", "country", "city", "event stage", "weapon", "weapon type",
            "perpetrator category", "perpetrator individual", "perpetrator organization",
            "perpetrator confidence", "physical target", "physical target type",
            "physical target number", "effect on physical target", "victim name",
            "victim description", "victim type", "victim number", "effect on victim"
        }
        
        def parse_json(text):
            text = text.strip()
            text = re.sub(r'^\s*,\s*|\s*,\s*$', '', text) # replace leading and trailing commas
            text = re.sub(r'\s+', ' ', text) # replace multiple spaces with a single space
            text = re.sub(r'}\s*,+\s*}', '}', text) # replace multiple closing braces with a single closing brace
            text = re.sub(r']\s*,+\s*$', ']', text) # replace multiple trailing commas with a single trailing comma
            text = re.sub(r'\}\s*\]\s*[,}\]]*', '}]', text) # replace multiple closing square brackets with a single closing square bracket
            text = re.sub(r'\{\s*\}\s*[,]?', '', text) # replace multiple empty objects with an empty object
            text = re.sub(r'\[\s*\]\s*[,]?', '', text) # replace multiple empty arrays with an empty array
            try:
                events = json.loads(text)
                if isinstance(events, list):
                    return [e for e in events if isinstance(e, dict) and e]
                return []
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed in compute_metrics: {e} at position {e.pos}")
                return []
        
        
        # Sample a subset of validation data to reduce memory usage
        max_samples = len(eval_dataset)  # Cap at 20 samples
        #eval_indices = random.sample(range(len(eval_dataset)), max_samples)
        eval_indices = list(range(max_samples))

        # Handle both Dataset and list types
        if hasattr(eval_dataset, 'select'):
            eval_subset = eval_dataset
        else:
            eval_subset = list(eval_dataset)  
        
        json_correct = 0
        field_correct = 0
        total_samples = 0
        batch_size = 4  # Process in small batches

        for i in range(0, len(eval_subset), batch_size):
            # Extract batch items - handle both Dataset and list
            if hasattr(eval_subset, 'select') and not isinstance(eval_subset, list):
                # HuggingFace Dataset - use select for slicing
                batch_indices = list(range(i, min(i + batch_size, len(eval_subset))))
                batch = [eval_subset[j] for j in batch_indices]
            else:
                # List slicing 
                batch = eval_subset[i:i + batch_size]
            
            # Stack input_ids and attention_mask from batch items
            input_ids_list = [item["input_ids"] for item in batch]
            attention_mask_list = [item["attention_mask"] for item in batch]
            batch_inputs = tokenizer.pad(
                {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
                padding=True,
                return_tensors="pt",
                )
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            

            with torch.no_grad():
                outputs = model.generate(
                    **batch_inputs,
                    **self.generation_config,
                )
                
            prompt_lens = batch_inputs["attention_mask"].sum(dim=1).tolist()    
            # Process each generated output
            for j in range(len(batch)):
                full_output_ids = outputs[j].cpu().tolist()
                output_ids = full_output_ids[prompt_lens[j]:]
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                
                parsed_output = parse_json(generated_text)
                if parsed_output:
                    json_correct += 1
                    for event in parsed_output:
                        event_fields = set(event.keys())
                        if event_fields == expected_fields:
                            field_correct += 1
                total_samples += 1

            # Clear GPU memory
            torch.cuda.empty_cache()

        metrics = {
            "json_accuracy": json_correct / total_samples if total_samples > 0 else 0.0,
            "field_accuracy": field_correct / total_samples if total_samples > 0 else 0.0,
        }
        return metrics
    
    
class ModelTrainer:    
    """Main class for handling model training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.prompter = None
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup distributed training
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = self.world_size != 1
        self.is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
        
    
    def setup_prompter(self):
        """Initialize the prompter for template management."""
        if self.is_main_process:
            print("Using SimplePrompter (instruction from dataset).")
        self.prompter = SimplePrompter() 
                
            
    def setup_model_and_tokenizer(self):
        """Load and configure the base model and tokenizer."""
        torch.cuda.empty_cache()
        
        base_lower = self.config.base_model.lower()
        try:
            model_type = getattr(AutoConfig.from_pretrained(self.config.base_model, trust_remote_code=True), "model_type", "")
        except Exception:
            model_type = ""
        
        # Set device map and attention implementation
        device_map = "auto" if torch.cuda.is_available() else None
        attn_kwargs = {}

        # Configure 4-bit quantization with bitsandbytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
            bnb_4bit_quant_type="nf4",  # NormalFloat4 for better precision
            bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16
            bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
        )
        

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            device_map=device_map,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            **attn_kwargs,
        )

        # Log VRAM usage
        if torch.cuda.is_available():
            print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GiB")
            print(f"VRAM reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GiB")
        
        base_lower = str(self.config.base_model).lower()

        # Ensure compatibility with gradient checkpointing
        try:
            self.model.config.use_cache = False  # Required for gradient checkpointing
        except Exception:
            pass

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            add_eos_token=self.config.add_eos_token,
            padding_side="right",
        )

        # Configure tokenizer padding and alignment
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        
        # Align model config with tokenizer to avoid warnings
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                self.model.config.bos_token_id = self.tokenizer.bos_token_id
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass

        # Print tokenizer info
        if self.is_main_process:
            print(f"Tokenizer BOS: {self.tokenizer.bos_token_id}")
            print(f"Tokenizer EOS: {self.tokenizer.eos_token_id}")
            print(f"Tokenizer PAD: {self.tokenizer.pad_token_id}")
 
    
    def setup_lora(self):
        """Configure LoRA for parameter-efficient fine-tuning."""
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True )        
    
        # Configure LoRA
        target_modules = self._resolve_lora_target_modules()
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        #self.model = maybe_restore_peft_model(self.model, training_config.resume_from_checkpoint)
        if self.is_main_process:
            self.model.print_trainable_parameters()
            
        #for gemma and qwen models
        if getattr(self.model.config, "tie_word_embeddings", False):
            self.model.get_input_embeddings().weight.requires_grad = True
        

        # Enable gradient checkpointing to reduce memory footprint
        try:
            self.model.gradient_checkpointing_enable()
            self.model.enable_adapter_layers()
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Could not enable gradient checkpointing or adapter layers: {e}")
            
    def _resolve_lora_target_modules(self):
        """Determine appropriate LoRA target modules for the current model.
        Resolution order:
          1) User-provided in config
          2) Auto-detect by scanning model submodules for known projection names
          3) Family heuristics via config.model_type
          4) Conservative fallback
        """
        # 1) Auto-detect by scanning module names
        preferred_order = [
            # attention projections 
            "q_proj", "k_proj", "v_proj", "o_proj",
            # fused/alternative attention projections
            "Wqkv", "out_proj", "c_attn", "c_proj",
            # MLP projections
            "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3",
        ]

        present: set = set()
        try:
            for name, _ in self.model.named_modules():
                short = name.rsplit(".", 1)[-1]
                if short in preferred_order:
                    present.add(short)
        except Exception:
            # If model doesn't support named_modules reliably, skip auto-detect
            present = set()

        auto_detected = [n for n in preferred_order if n in present]
        if auto_detected:
            if self.is_main_process:
                print(f"Auto-detected LoRA target modules: {auto_detected}")
            return auto_detected

        # 2) User-provided override (used only if auto-detect fails)
        if self.config.lora_target_modules:
            if self.is_main_process:
                print(f"Using user-specified LoRA target modules: {self.config.lora_target_modules}")
            return self.config.lora_target_modules

        # 3) Family heuristics via model_type
        model_type = getattr(getattr(self.model, "config", None), "model_type", "") or ""
        model_type = model_type.lower()

        if model_type.startswith("olmo") or model_type == "olmo":
            # OLMo uses combined qkv and out proj; MLP uses fc1/fc2
            return ["Wqkv", "out_proj", "fc1", "fc2"]
        if model_type.startswith("qwen") or model_type == "qwen":
            # Qwen models use standard attention and MLP projections
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

        # 4) Fallback 
        return ["q_proj", "k_proj", "v_proj", "o_proj"]        
                
    def tokenize_prompt(self, prompt: str, add_eos_token: bool = True) -> Dict[str, Any]:
        """Tokenize a prompt with handling of EOS tokens."""
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        
        # Add EOS token if needed
        if (result["input_ids"][-1] != self.tokenizer.eos_token_id 
            and len(result["input_ids"]) < self.config.cutoff_len
            and add_eos_token):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        # Set labels
        result["labels"] = result["input_ids"].copy()
        
        return result        
            
    def generate_and_tokenize_prompt(self, data_point: Dict[str, str]) -> Dict[str, Any]: 
        
        #  Build full prompt and mask inputs
        
        instruction_text = data_point.get("instruction") or ""
        # Qwen3 models support a '\no_think' control token to disable thinking output.
        # We keep prompts consistent between training and inference by injecting it here.
        base_lower = str(self.config.base_model or "").lower()
        if "qwen3" in base_lower:
            instruction_text = f"{instruction_text}\n\\no_think"
        full_prompt = self.prompter.generate_prompt(
            instruction_text,
            data_point["input"],
            data_point["output"]
        )
        tokenized_full_prompt = self.tokenize_prompt(full_prompt)
        # if train_on_inputs is False, mask the input tokens
        # only compute loss on the output tokens
        if not self.config.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                instruction_text,
                data_point["input"]
            )
            tokenized_user_prompt = self.tokenize_prompt(
                user_prompt,
                add_eos_token=self.config.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if self.config.add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt
    
    def load_data(self):
        """Load and preprocess training and validation data."""
        # Load training data
        train_df = pd.read_csv(self.config.train_data_path)
        #Shuffle training data with seed for reproducibility
        train_df = train_df.sample(frac=1.0, random_state=self.config.seed).reset_index(drop=True)
        # Print sample prompt from first training example
        if self.is_main_process and len(train_df) > 0:
            first_example = train_df.iloc[3]
            # Prefer per-row instruction from dataset
            instruction_text = first_example.get("instruction")
            base_lower = str(self.config.base_model or "").lower()
            if "qwen3" in base_lower:
                instruction_text = f"{instruction_text or ''}\n\\no_think"
            sample_prompt = self.prompter.generate_prompt(
                instruction_text,
                first_example.get("input", ""),
                first_example.get("output", "")
            )
            print("\n" + "="*80)
            print("SAMPLE TRAINING PROMPT (first example):")
            print("="*80)
            print(sample_prompt)
            print("="*80 + "\n")
            
            
        
        # Always use classic prompt mode (SimplePrompter). This setup does not use chat templates.
        train_data = [
            self.generate_and_tokenize_prompt(data)
            for _, data in train_df.iterrows()
        ]
        
        val_data = None
        if self.config.val_data_path:
            val_df = pd.read_csv(self.config.val_data_path)
            val_data = [
                self.generate_and_tokenize_prompt(data)
                for _, data in val_df.iterrows()
            ]
        
        return train_data, val_data        
    
    def load_checkpoint(self):
        """Load model from checkpoint if specified."""
        if not self.config.resume_from_checkpoint:
            return
        
        adapter_safe = os.path.join(self.config.resume_from_checkpoint, "adapter_model.safetensors")
        adapter_bin = os.path.join(self.config.resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(adapter_safe) and safe_load_file is not None:
            print(f"Restarting from {adapter_safe}")
            adapters_weights = safe_load_file(adapter_safe)
            set_peft_model_state_dict(self.model, adapters_weights)
        elif os.path.exists(adapter_bin):
            print(f"Restarting from {adapter_bin}")
            adapters_weights = torch.load(adapter_bin, weights_only=True, map_location="cpu")
            set_peft_model_state_dict(self.model, adapters_weights)
        else:
            print(f"Checkpoint not found (no adapter weights) in {self.config.resume_from_checkpoint}") 
    
    def setup_trainer(self, train_data, val_data):
        """Configure trainer"""
        gradient_accumulation_steps = self.config.batch_size // self.config.micro_batch_size
        # distributed training
        if self.ddp:
            gradient_accumulation_steps = gradient_accumulation_steps // self.world_size
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.micro_batch_size,
            per_device_eval_batch_size=1,  # Reduced for Gemma
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=5,
            optim="adamw_torch", 
            lr_scheduler_type=self.config.lr_scheduler,
            output_dir=self.config.output_dir,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if self.ddp else None,
            group_by_length=self.config.group_by_length,
            report_to="none",
            save_on_each_node=True,
            eval_strategy="epoch" if val_data else "no",
            save_strategy="epoch",
            save_total_limit=4,
            metric_for_best_model=self.config.early_stopping_metric if val_data else None,
            greater_is_better=False if val_data else None, #we are using loss,
        )

        training_args.seed =self.config.seed
        try:
            os.makedirs(training_args.output_dir, exist_ok=True)
            args_path = os.path.join(training_args.output_dir, "training_args.json")
            with open(args_path, "w") as f:
                json.dump(training_args.to_dict(), f, indent=2)
        except Exception:
            pass
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
         #aggressive cleaning of CUDA memory before each eval run.
        old_eval_loop = transformers.Trainer.evaluation_loop
        def patched_eval_loop(*args, **kwargs):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return old_eval_loop(*args, **kwargs)
        
        transformers.Trainer.evaluation_loop = patched_eval_loop
        callbacks = [
                SavePeftModelCallback(),
                LoadBestPeftModelCallback()]
        if val_data:
            callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=self.config.early_stopping_patience,
                        early_stopping_threshold=self.config.early_stopping_threshold,
                      )
                    )
        


        self.trainer = MyTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=training_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            ),
            callbacks=callbacks,
        )
        self.model.config.use_cache = False
        enable_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0")
        if enable_compile == "1" and torch.__version__ >= "2" and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
    
    
    def train(self):
        """run the full training pipeline."""
        # Print config
        if self.is_main_process:
            self._print_config()
        
        self.setup_prompter()
        self.setup_model_and_tokenizer()
        # Enable input grads only during training when gradient checkpointing is requested
        
        if getattr(self.config, "gradient_checkpointing", False):
            try:
                self.model.config.use_cache = False # always required for checkpointing
                self.model.enable_input_require_grads()
            except Exception:
                pass
        
        
        self.setup_lora()
        
        # Load data
        train_data, val_data = self.load_data()
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Setup trainer
        self.setup_trainer(train_data, val_data)

        
        print("Checking if any parameters require grad...")
        print("Total trainable parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # Start training
        self.trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)
        
        # Save
        self._save_final_model()
    
    def _print_config(self):
        """Print training configuration."""
        print(f"Training Configuration:")
        print(f"  Base Model: {self.config.base_model}")
        print(f"  Train Data: {self.config.train_data_path}")
        print(f"  Val Data: {self.config.val_data_path}")
        print(f"  Output Dir: {self.config.output_dir}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Micro Batch Size: {self.config.micro_batch_size}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  LoRA r: {self.config.lora_r}")
        print(f"  LoRA alpha: {self.config.lora_alpha}")
        print(f"  LoRA dropout: {self.config.lora_dropout}")
        print(f"  LoRA target modules: {self.config.lora_target_modules}")
    
    def _save_final_model(self):
        """Save the final trained model."""
        # Save PEFT model
        self.model.save_pretrained(self.config.output_dir)
        
        # Save empty pytorch model for compatibility
        pytorch_model_path = os.path.join(self.config.output_dir, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.config.output_dir)
    
class SimplePrompter:
    """Simple prompter."""
    
    def generate_prompt(self, instruction: str, input_text: str = None, output: str = None) -> str:
        """Generate a simple prompt format."""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        if output:
            prompt += output
        
        return prompt
    
    
def train(
    base_model: str = "",
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "",
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    train_on_inputs: bool = False,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    prompt_template_name: str = "alpaca",
):
    """
    Main training function for fine-tuning llms with LoRA for muc.
    
    Args:
        base_model: Name or path of the base model to fine-tune
        train_data_path: Path to training data CSV file
        val_data_path: Path to validation data CSV file (optional)
        output_dir: Directory to save the trained model
        batch_size: Total batch size for training
        micro_batch_size: Batch size per device
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        cutoff_len: Maximum sequence length
        val_set_size: Size of validation set (0 to disable)
        lr_scheduler: Learning rate scheduler type
        warmup_steps: Number of warmup steps
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: List of target modules for LoRA
        train_on_inputs: Whether to include inputs in loss calculation
        add_eos_token: Whether to add EOS token to user prompts
        group_by_length: Whether to group sequences by length
        resume_from_checkpoint: Path to checkpoint to resume from
        prompt_template_name: Name of the prompt template to use
    """
    # Create configuration
    config = TrainingConfig(
        base_model=base_model,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        val_set_size=val_set_size,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        train_on_inputs=train_on_inputs,
        add_eos_token=add_eos_token,
        group_by_length=group_by_length,
        resume_from_checkpoint=resume_from_checkpoint,
        prompt_template_name=prompt_template_name,
    )
    
    # Create trainer and start training
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    # command line arguments
    fire.Fire(train)  
