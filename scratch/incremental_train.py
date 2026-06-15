import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Path to your existing LoRA adapter folder")
    parser.add_argument("--dataset_dir", type=str, default="output/dataset")
    parser.add_argument("--output_dir", type=str, default="./incremental_results")
    parser.add_argument("--lr", type=float, default=2e-5, help="Low learning rate for incremental adjustment")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 1. Load dataset splits
    data_files = {
        "train": os.path.join(args.dataset_dir, "train.jsonl"),
        "validation": os.path.join(args.dataset_dir, "val.jsonl")
    }
    dataset = load_dataset("json", data_files=data_files)
    
    # 2. Load base model with correct classification configurations
    tags = ["O", "B-question_label", "I-question_label", "B-stem", "I-stem", 
            "B-option_label", "I-option_label", "B-option_text", "I-option_text", 
            "B-context", "I-context"]
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    id_to_tag = {i: tag for i, tag in enumerate(tags)}

    base_model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(tags),
        id2label=id_to_tag,
        label2id=tag_to_id
    )

    # 3. Load the existing LoRA adapter for training
    print(f"Loading existing adapter from {args.adapter_dir} for incremental training...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=True)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer)
    )

    print("Starting incremental fine-tuning...")
    trainer.train()
    
    print(f"Saving updated adapter weights to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
