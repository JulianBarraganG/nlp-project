import gc
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertForTokenClassification,
)
from sklearn.metrics import confusion_matrix, classification_report
from datasets import Dataset
import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import torch
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
        num_train_epochs=10,
        # Regularization
        weight_decay=0.01,
        # Memory settings
        auto_find_batch_size=True,
        fp16=True,
        # Evaluation
        save_strategy="best",
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
    with torch.no_grad():
        torch.cuda.empty_cache()
    # Train and save the model
    print("Training mBERT classifier...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Environment variable set: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    trainer.train()

    return classifier, tokenizer

# Function to get predictions
def predict_binary(
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


def predict_bio_sequence(
    question: pl.Series,
    context: pl.Series,
    model: BertForTokenClassification,
    tokenizer: AutoTokenizer
):
    """Get model BIO-label prediction sequence"""

    inputs = tokenizer(
        question, 
        context, 
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ) # type: ignore

    # Move to GPU if available
    inputs = {k: v.cpu() for k, v in inputs.items()}
    model = model.cpu() # type: ignore
    
    model.eval() # type: ignore
    with torch.no_grad():
        outputs = model(**inputs) # type: ignore
        logits = outputs.logits
        #probs = torch.softmax(logits, dim=1)
        #prediction = torch.argmax(logits, dim=1).item()
    
        probs = torch.softmax(logits, dim=2)  # Softmax over the last dimension (num_labels)
        prediction = torch.argmax(probs, dim=2).squeeze().tolist()  #

    return prediction

def _get_answer_ids(
    answer_start: int,
    answer_text: str,
    context: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> list[int]:
    """Get token input indices for the answer span in the context."""

    answer_end = answer_start + len(answer_text)

    if answer_start == -1:  # Unanswerable
        return []

    context_encoding = tokenizer(
        context,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    ) # type: ignore

    answer_token_indices = []
    offset_mapping = context_encoding["offset_mapping"]

    for idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_start == 0 and token_end == 0:
            continue
        if token_start < answer_end and token_end > answer_start:
            answer_token_indices.append(idx)

    if answer_token_indices:
        answer_input_ids = [context_encoding["input_ids"][idx] for idx in answer_token_indices]
        return answer_input_ids
    else:
        print(f"WARNING: No tokens found for answer '{answer_text}' at position {answer_start}")
        print(f"Context: {context[max(0, answer_start-20):answer_start+len(answer_text)+20]}")
        return []


def bio_sequence_labeler(
    answer_start: int,
    answer_text: str,
    question: str,
    context: str, 
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> np.ndarray:
    """Creates a BIO-encoded sequence label for the answer span in the context.
    If no answer, all 'O's. Encoded with 0, 1, 2 for O, B-ANS, I-ANS respectively."""

    answer_input_ids = _get_answer_ids(
        answer_start,
        answer_text,
        context,
        tokenizer,
        max_length,
    )
        
    labels = np.zeros(max_length, dtype=np.int8)

    full_encoding = tokenizer(
        question,
        context,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    ) # type: ignore

    full_sequence_ids = full_encoding["input_ids"]
    seq_len = len(full_sequence_ids)
    ans_len = len(answer_input_ids)
    if answer_input_ids:
        for i in range(seq_len - ans_len + 1): 
            if full_sequence_ids[i:(i + ans_len)] == answer_input_ids:
                labels[i] = 1  # B-ANS
                for j in range(i+1, i+ans_len):
                    labels[j] = 2  # I-ANS
        
    return labels


def get_results(
    model: BertForTokenClassification,
    val_set: pl.DataFrame,
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]]:
    """Return y_true and y_pred lists for evaluation"""
    y_true = []
    y_pred = []
    y_pred = predict_bio_sequence(
        val_set["question"].to_list(), # type: ignore
        val_set["context"].to_list(), # type: ignore
        model,
        tokenizer, 
    )
    y_true = val_set["labels"].explode().to_list()
    y_pred = np.array(y_pred).flatten().tolist()

    return y_true, y_pred


def display_results(
    y_pred: list[int],
    y_true: list[int],
    bio_labels: list[str] = ["O", "B-ANS", "I-ANS"],
    title: str = "Confusion Matrix"
) -> None:
    """Display confusion matrix and classification report"""

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues) # type: ignore
    plt.title(title)

    #plt.colorbar()
    tick_marks = range(len(bio_labels))
    plt.xticks(tick_marks, bio_labels)
    plt.yticks(tick_marks, bio_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # Include numbers as text in the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.show()
    print(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=bio_labels)}")   
