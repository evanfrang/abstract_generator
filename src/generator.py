import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os


def prepare_model():

    FINE_TUNED_MODEL_PATH = "models/fine_tuned_abstract_generator_tinyllama"
    BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading model and tokenizer. This might take a moment...")

    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    peft_config = PeftConfig.from_pretrained(FINE_TUNED_MODEL_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, FINE_TUNED_MODEL_PATH)
    model.eval()

    print("Model loaded successfully. Ready for generation.") 
    return model, tokenizer

def generate_abstract(
        category: str, title: str, model, tokenizer, max_new_tokens: int = 200
):
    input_prompt = (
        f"categories: {category.strip()}\n\n"
        f"title: {title.strip()}\n\n"
        f"abstract: "
    )

    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, \
                       max_length=512).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,        
        temperature=0.7,       
        top_p=0.9,            
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], \
                                      skip_special_tokens=True)
    
    clean_text = generated_text.strip()
    stop_phrases = ["Category:", "Title:", "\n\n"] 
    for phrase in stop_phrases:
        if phrase in clean_text:
            clean_text = clean_text.split(phrase)[0].strip()
    
    return clean_text


if __name__ == '__main__':
    model, tokenizer = prepare_model()
    test_category = "cond-mat"
    test_title = "Using doped silicon wafers to explore paramagnetism in rocks"
    print("\n--- Abstract Generation ---")
    print(f"Category: {test_category}")
    print(f"Title: {test_title}")

    generated_abstract = generate_abstract(
        test_category,
        test_title,
        model,
        tokenizer
    )
    print(f"\nGenerated Abstract:\n{generated_abstract}")