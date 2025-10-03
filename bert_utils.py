import gc
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import polars as pl
import torch
from torch.cuda import OutOfMemoryError
from typing import Any

# Prepare your data
def prepare_data(df: pl.DataFrame) -> Dataset:
    # Convert Polars to dict format for HF datasets
    data_dict = {
        "question": df["question"].to_list(),
        "context": df["context"].to_list(),
        "label": df["answerable"].cast(int).to_list()  # Convert bool to int
    }
    return Dataset.from_dict(data_dict)


# Tokenization function
def tokenize_function(examples: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    # Tokenize with question and content separated by [SEP]
    # [CLS] is added automatically
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512
    ) # type: ignore

###### Train pipeline ######
def train_mbert(
    tokenized_train: Dataset,
    tokenized_val: Dataset,
    model_checkpoint: str = "bert-base-multilingual-uncased",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    # Load model
    classifier = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
    ).to(device)
    # Load tokenizer (mostly for saving complete model later)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        # Regularization
        weight_decay=0.01,
        # Memory settings
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        fp16=True,
        # Evaluation
        per_device_eval_batch_size=8,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=classifier,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )
    # Clear torch cache before training
    gc.collect()
    torch.cuda.empty_cache()
    # Train and save the model
    print("Training mBERT classifier...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Environment variable set: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    trainer.train()

    return classifier, tokenizer

# Function to get predictions
def predict(
    question: pl.Series,
    context: pl.Series,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer
) -> dict[str, Any]:
    """Get model prediction for a single example"""
    inputs = tokenizer(
        question, 
        context, 
        truncation=True, 
        max_length=512,
        return_tensors="pt"
    ) # type: ignore
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda() # type: ignore
    else:
        inputs = {k: v.cpu() for k, v in inputs.items()}
        model = model.cpu() # type: ignore
    
    model.eval() # type: ignore
    with torch.no_grad():
        outputs = model(**inputs) # type: ignore
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
    
    return {
        'prediction': prediction,  # 0 or 1
        'confidence': probs[0][prediction].item(), # type:ignore
        'prob_class_0': probs[0][0].item(),
        'prob_class_1': probs[0][1].item()
    }