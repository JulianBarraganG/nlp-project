import polars as pl
from datasets import Dataset
import numpy as np
import evaluate
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, EarlyStoppingCallback


def answer_question(promt: str, model, tokenizer, **kwargs) -> str:
    input_tokens = tokenizer(promt, return_tensors="pt", truncation=True, max_length=512).to(device)
    generated_tokens = model.generate(
        **input_tokens,
        max_new_tokens=64,
        **kwargs
    )
    answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return answer

def generate_prompt_wo_context(df: pl.DataFrame):
    df = df.with_columns([pl.col("question").alias("prompt")])
    return df

def generate_prompt_with_context(df: pl.DataFrame):
    df = df.with_columns([
        (pl.lit("Question: ") + pl.col("question") + pl.lit("\n Context: ") + pl.col("context")).alias("prompt")
    ])
    return df

def tokenize_to_dataset(df: pl.DataFrame, tokenizer, question_col: str = "prompt", answer_col: str = "answer_inlang"):

    inputs = tokenizer(
        df[question_col].to_list(),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    labels = tokenizer(
        df[answer_col].to_list(),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    dataset_dict = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({k: v.numpy() for k, v in dataset_dict.items()})
    return dataset

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    #preds = preds[0] if isinstance(preds, tuple) else preds
    #preds = np.argmax(preds, -1) if preds.ndim == 3 else preds
    
    # Cleaning predictions
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) # Replace -100 in labels with pad_token_id for decoding
    preds = np.where((preds < 0) | (preds >= tokenizer.vocab_size), tokenizer.unk_token_id, preds) # Replace out-of-range token IDs with <unk> ID

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]

    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    rouge = rouge_metric.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])

    return {
        "bleu": round(bleu, 4),
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "rougeLsum": round(rouge.get("rougeLsum", 0.0), 4),
    }

# https://huggingface.co/learn/llm-course/chapter7/4?fw=pt
def trainer_generator(model, tokenizer, train_dataset, eval_dataset, output_dir, epochs, patience=3):
    training_args = Seq2SeqTrainingArguments(
        fp16=False,
        auto_find_batch_size=True,

        output_dir=output_dir,
        overwrite_output_dir = True,
        learning_rate=2e-5,
        
        predict_with_generate=True,
        num_train_epochs=epochs,
        weight_decay=0.01,
        generation_max_length=128,

        save_total_limit=3,
        save_strategy = "best",
        load_best_model_at_end = True,

        logging_strategy="epoch",
        eval_strategy = "epoch",
        log_level="info",
        report_to=[],
        logging_dir=None
    )
    #data_collator = DataCollatorForSeq2Seq(mt5_tokenizer, model=mt5_model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    #    data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        callbacks = [EarlyStoppingCallback(patience)]
    )

    return trainer