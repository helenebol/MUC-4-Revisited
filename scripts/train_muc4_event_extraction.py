#!/usr/bin/env python3
"""
MUC-4 Event analysis Fine-tuning

This script fine-tunes language models on MUC-4 abstractive event analysis tasks.
"""

import os
import sys
import argparse

# Add both current directory (scripts/) and parent directory (eventMUC/) to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)  # scripts/ for local imports
sys.path.insert(0, parent_dir)   # eventMUC/ for instructions and convert_data

from finetune import TrainingConfig, ModelTrainer
from convert_data.muc4_data_processor import MUC4DataProcessor
import json
from datetime import datetime


def _sanitize_model_name(model_id: str) -> str:
    """Create a filesystem name from a base model id or path."""
    name = os.path.basename(str(model_id).rstrip("/")) if "/" in str(model_id) else str(model_id)
    # If it's an HF repo id like org/name, take last part
    if "/" in name:
        name = name.split("/")[-1]
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return safe or "model"


def setup_muc4_training(
    base_model: str = "OLMo-1b",
    muc4_data_dir: str = "../data/Processed/",
    output_dir: str = "muc4_event_extraction_model",
    batch_size: int = 8,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    cutoff_len: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    resume_from_checkpoint: str = None,
    json_format: str = "auto",
):
    """
    Setup and run MUC-4 event extraction training.
    
    Args:
        base_model: Base model to fine-tune
        muc4_data_dir: Path to MUC-4 data directory
        output_dir: Directory to save the trained model
        batch_size: Total batch size for training
        micro_batch_size: Batch size per device
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        cutoff_len: Maximum sequence length
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    
    # Process MUC-4 data
    processed_data_dir = "muc4_processed_multi"
    train_data_path = os.path.join(processed_data_dir, "muc4_train.csv")
    val_data_path = os.path.join(processed_data_dir, "muc4_val.csv")
    format_choice ="auto"
    
    print(f"Processing MUC-4 data...")
    print(f"Output directory will be: {os.path.abspath(processed_data_dir)}")
    
    print(f"Selected JSON formatting: {format_choice}")
    processor = MUC4DataProcessor(muc4_data_dir, json_format=format_choice, base_model=base_model)
    processor.process_all_splits(processed_data_dir)

    # Verify data files exist
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    if not os.path.exists(val_data_path):
        print(f"Warning: Validation data not found at {val_data_path}")
        val_data_path = ""
    
    # Resolve per-base-model output subdirectory
    model_dir_name = _sanitize_model_name(base_model)
    model_output_dir = os.path.join(output_dir, model_dir_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    template_name = "muc4_event_extraction"
    
    config = TrainingConfig(
        base_model=base_model,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=model_output_dir,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        val_set_size=0,  # We have separate validation file
        lr_scheduler="cosine",
        warmup_steps=50,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=["gate_proj", "down_proj", "up_proj"],
        train_on_inputs=False,  # Only train on outputs for event extraction
        add_eos_token=True,
        group_by_length=False,
        resume_from_checkpoint=resume_from_checkpoint,
        prompt_template_name=template_name,
        #fixed_instruction=fixed_inst,
    )
    
    # Start training
    print("Starting MUC-4 event extraction training...")
    print(f"Base model: {base_model}")
    print(f"Training data: {train_data_path}")
    print(f"Validation data: {val_data_path}")
    print(f"Output directory (root): {output_dir}")
    print(f"Saving model under: {model_output_dir}")
    print(f"Prompt template: {template_name}")
    # Save training metadata and an example prompt
    
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_model": base_model,
        "output_root": output_dir,
        "model_output_dir": model_output_dir,
        "prompt_template": template_name,
        "json_format": format_choice,
        "training_config": {
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "cutoff_len": cutoff_len,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        },
        "data_paths": {
            "train_csv": train_data_path,
            "val_csv": val_data_path,
            "data_dir": muc4_data_dir,
        },
    }
    with open(os.path.join(model_output_dir, "training_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
  
    trainer = ModelTrainer(config)
    trainer.train()
    
    print(f"Training completed! Model saved to {output_dir}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune model for MUC-4 event extraction")
    
    # Model and data parameters
    parser.add_argument("--base_model", type=str, 
                       default="/fp/projects01/ec30/helenbol/models/OLMo-1b",
                       help="Base model to fine-tune")
    parser.add_argument("--muc4_data_dir", type=str, 
                       default="../muc34/data",
                       help="Path to MUC-4 data directory")
    parser.add_argument("--output_dir", type=str, 
                       default="muc4_event_extraction_model",
                       help="Directory to save the trained model")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Total batch size for training")
    parser.add_argument("--micro_batch_size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate for training")
    parser.add_argument("--cutoff_len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--json_format", type=str, default="auto",
                       choices=["auto", "indent2", "compact"],
                       help="JSON formatting for labels: auto=Gemma→compact else indent2")
    # Multi-event only; flag removed (kept here suppressed for backwards compatibility if scripts pass it).
    parser.add_argument("--multi_event", action="store_true", default=True, help=argparse.SUPPRESS)
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate")
    
    # Other parameters
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Run training
    setup_muc4_training(
        base_model=args.base_model,
        muc4_data_dir=args.muc4_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume_from_checkpoint=args.resume_from_checkpoint,
        json_format=args.json_format,
    )


if __name__ == "__main__":
    main() 
