import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
tokenizer = AutoTokenizer.from_pretrained("gpt-j-6B-smart-contract")
tokenizer.pad_token = "[PAD]"
model = AutoModelForCausalLM.from_pretrained("gpt-j-6B-smart-contract").to(device)
print("Model loaded")

prompts = ["""
pragma solidity ^0.8.10;

contract HelloWorld {
    function hello() public pure returns (string) {
"""]

# Tokenize
input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(device)

# Generate
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=512,
)
generated_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
print(generated_texts)
