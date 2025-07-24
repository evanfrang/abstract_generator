import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading ORIGINAL base model: {BASE_MODEL_NAME}. This will take a moment...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, 
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

print("Original base model loaded successfully. Ready for generation.")


def generate_basic_abstract(subgenre: str, title: str, model, tokenizer, max_new_tokens: int = 200):
    """
    Generates a basic abstract using the provided model.
    """
  
    input_prompt = (
        f"SUBGENRE: {subgenre.strip()}\n\n"
        f"TITLE: {title.strip()}\n\n"
        f"ABSTRACT:"
    )


    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,        
        temperature=0.7,      
        top_p=0.9,              
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id, 
    )
    

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    

    clean_text = generated_text.strip()
    stop_phrases = ["SUBGENRE:", "TITLE:", "\n\n", "Keywords:"] 
    for phrase in stop_phrases:
        if phrase in clean_text:
            clean_text = clean_text.split(phrase)[0].strip()
    
    return clean_text

if __name__ == "__main__":
    print("\n--- Testing Original TinyLlama on Academic Abstract Prompt ---")

    test_subgenre = "gr-qc"
    test_title = "How large can black holes be?" 
    print(f"Subgenre: {test_subgenre}")
    print(f"Title: {test_title}")

    generated_abstract = generate_basic_abstract(test_subgenre, test_title, model, tokenizer)
    print(f"\nGenerated Abstract from ORIGINAL TinyLlama:\n{generated_abstract}")