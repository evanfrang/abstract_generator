from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer

def sample_df(df, frac=0.01):
    df_sample = df.sample(frac=frac, random_state=11)
    return df_sample

def split_dataset(df):
    
    hf_dataset = Dataset.from_pandas(df[['input', 'abstract']])
    train_test_split = hf_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    return train_dataset, eval_dataset

def tokenize_func(examples):
    full_text = [
        f"{prompt}{target}" for prompt, target in zip(examples['input'], examples['abstract'])
    ]
    tokenized_inputs = tokenizer(
        full_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


if __name__ == '__main__':
    df_raw = pd.read_parquet("data/arxiv_100k.parquet")
    df = sample_df(df_raw, frac=0.01)
    train_data, eval_data = split_dataset(df)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_train_dataset = train_data.map(tokenize_func, batched=True, \
                        remove_columns=train_data.column_names)
    tokenized_eval_dataset = eval_data.map(tokenize_func, batched=True, \
                        remove_columns=eval_data.column_names)
    tokenized_train_dataset.save_to_disk('data/processed/train_tokenized.parquet')
    tokenized_eval_dataset.save_to_disk('data/processed/eval_tokenized.parquet')
    