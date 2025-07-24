from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model
import os


def model_setup():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "models/fine_tuned_abstract_generator_tinyllama"
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "down_proj", "gate_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" 
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return tokenizer, model, lora_config, output_dir

def train_model(tokenizer, model, token_train_data, token_eval_data, output_dir, lora_config=None):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=8,  
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        fp16=True,  
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=token_train_data,
        eval_dataset=token_eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    trainer.train()

    
    if lora_config:
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)   

if __name__ == '__main__':
    tokenized_train_dataset = load_from_disk('data/processed/train_tokenized.parquet')
    tokenized_eval_dataset = load_from_disk('data/processed/eval_tokenized.parquet')

    tokenizer, model, lora_config, output_dir = model_setup()
    train_model(tokenizer, model, tokenized_train_dataset, \
        tokenized_eval_dataset, output_dir, lora_config)