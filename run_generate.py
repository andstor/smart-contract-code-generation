from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("model_name_or_path")
parser.add_argument("length")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = "[PAD]"
model = AutoModelForCausalLM.from_pretrained(args.model).half().to("cuda") # TODO: half precision?

while True:
    text = input("\n\Prompt: ")
    text = str(text)
    if len(text) == 0:
        continue
    ids = tokenizer(text, padding=True, return_tensors="pt").input_ids.to("cuda")

    gen_tokens = model.generate(
        ids,
        do_sample=True,
        max_length=args.length,
        temperature=0.9,
        use_cache=True
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    print(gen_text)