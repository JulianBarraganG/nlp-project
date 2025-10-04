import os
import polars as pl
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration, 
    MT5Tokenizer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
from torch.utils.data import Dataset
import evaluate
import numpy as np


# Custom Dataset Class
class TeluguQADataset(Dataset):
    """Dataset for Telugu Question Answering with mT5"""
    
    def __init__(self, df, tokenizer, use_context=True, 
                 max_input_length=512, max_target_length=128):
        """
        Args:
            df: Polars DataFrame with columns: question, context, answer_inlang
            tokenizer: MT5Tokenizer instance
            use_context: If True, includes context in input; if False, question only
            max_input_length: Maximum length for input sequences
            max_target_length: Maximum length for target sequences
        """
        self.df = df
        self.tokenizer = tokenizer
        self.use_context = use_context
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        
        # Format input based on whether we use context
        if self.use_context:
            input_text = f"question: {row['question']} context: {row['context']}"
        else:
            input_text = f"question: {row['question']}"
        
        target_text = row['answer_inlang']
        
        # Tokenize inputs
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        # Replace padding token ids with -100 so they're ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def compute_metrics(eval_pred, tokenizer, metric):
    """
    Compute ROUGE and BLEU metrics for generation
    """
    predictions, labels = eval_pred
    
    # Replace -100 with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract a few key metrics
    return {
        'rouge1': result['rouge1'],
        'rouge2': result['rouge2'],
        'rougeL': result['rougeL'],
    }


def train_model(use_context=True, output_dir=None):
    """
    Train a single model (with or without context)
    
    Args:
        use_context: Whether to include context in inputs
        output_dir: Directory to save model checkpoints
    """
    print(f"\n{'='*60}")
    print(f"Training model {'WITH' if use_context else 'WITHOUT'} context")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer and model
    tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)
    model = model.to(device)
    
    # Create datasets
    train_dataset = TeluguQADataset(df_te_train, tokenizer, use_context=use_context)
    val_dataset = TeluguQADataset(df_te_val, tokenizer, use_context=use_context)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Load evaluation metric
    rouge_metric = evaluate.load("rouge")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir or f"./mt5-telugu-qa-{'with' if use_context else 'no'}-context",
        num_train_epochs=3,
        # Memory limitations driven settings
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        fp16=True,

        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        warmup_steps=20,
        logging_steps=10,
        predict_with_generate=True,  # Important for generation metrics
        generation_max_length=128,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, rouge_metric),
    )
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Set cuda env variable
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    results = trainer.evaluate()
    
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    return trainer, tokenizer, model


def test_generation(trainer, tokenizer, df_test, use_context=True, n_examples=5):
    """
    Test generation on a few examples and print results
    """
    print(f"\n{'='*60}")
    print(f"Sample Predictions ({'WITH' if use_context else 'WITHOUT'} context)")
    print(f"{'='*60}\n")
    
    model = trainer.model
    model.eval()
    
    for idx in range(min(n_examples, len(df_test))):
        row = df_test.row(idx, named=True)
        
        # Prepare input
        if use_context:
            input_text = f"question: {row['question']} context: {row['context'][:200]}..."
        else:
            input_text = f"question: {row['question']}"
        
        # Generate
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Question: {row['question']}")
        print(f"Gold Answer: {row['answer_inlang']}")
        print(f"Predicted: {prediction}")
        print(f"Answerable: {row.get('answerable', 'Unknown')}")
        print("-" * 60)


# Main execution
if __name__ == "__main__":
    # Configuration
    model_checkpoint = "google/mt5-small"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and prepare data
    print("Loading dataset...")
    dataset = load_dataset("coastalcph/tydi_xor_rc")
    df_train = dataset["train"].to_polars()
    df_val = dataset["validation"].to_polars()

    # Filter Telugu examples with in-language answers
    df_te_train = df_train.filter(
        pl.col("lang") == "te", 
        pl.col("answer_inlang").is_not_null()
    )
    df_te_val = df_val.filter(
        pl.col("lang") == "te", 
        pl.col("answer_inlang").is_not_null()
    )

    print(f"Training examples: {len(df_te_train)}")
    print(f"Validation examples: {len(df_te_val)}")

    # Get the model
    tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)

    # Train model with context
    trainer_with_context, tokenizer_with_context, model_with_context = train_model(use_context=True)
    test_generation(trainer_with_context, tokenizer_with_context, df_te_val, use_context=True)

    # Save the model with context
    model_with_context.save_pretrained("./mt5-telugu-qa-with-context")
