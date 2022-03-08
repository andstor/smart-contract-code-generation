import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.pad_token = "[PAD]

print("Loading complete")

prompts = ["Hello", "How are you?", "What is your name?", "What is the weather like?", "What is the time?",
           "What is the date?", "What is the day of the week?", "What is the month?", "What is the year?",]

print("Tokenizing...")
input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids.to(device)

print("Generating...")

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=300,
)
generated_text = tokenizer.batch_decode(gen_tokens)
print(generated_text)