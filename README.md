# smart-contract-code-generation

> 🧠 Smart contract code generation based on large-scale transformer-based language model.


GPT-J is the open-source alternative to OpenAI's GPT-3. The model is trained on the Pile, a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality datasets combined together.

## Fine-tuning GPT-J (6 Billion Parameters)
### Setup
```shell
git clone https://github.com/andstor/smart-contract-code-generation.git
cd smart-contract-code-generation
pip install -r requirements.txt 
```
### Requirements
The GPT-J model is an extremely large language model.

### Distributed training
Due to the large size of the model, it is not feasible to train it on a single GPU. The following code shows how to train the model on multiple GPUs using Microsoft's DeepSpeed library.

```
deepspeed --hostfile=hostfile run_clm.py \
--deepspeed ds_zero2_bf16.json \
--model_name_or_path EleutherAI/gpt-j-6B \
--dataset_name andstor/smart_contracts \
--dataset_config_name plain_text \
--overwrite_output_dir true \
--output_dir finetuned \
--do_train --do_eval \
--num_train_epochs 2 \
--evaluation_strategy steps --eval_steps 5 \
--block_size 1024 \
--bf16 \
--gradient_accumulation_steps 16 --eval_accumulation_steps 16 \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1
``` 
### Mixed precision
If a GPU with mixed precision capabilities (architecture Pascal or more recent) is available, one can use mixed precision training with PyTorch 1.6.0 or later, or by installing the Apex library for previous versions. Just add the flag --fp16 to your command! If you have a NVIDIA “Ampere” GPU architecture, you can use Brain Floting Point (BF16) by passing the flag --bf16.

Using mixed precision training usually results in 2x-speedup for training with the same final results.


### Logging & Experiment tracking

You can easily log and monitor your runs code.

* [TensorBoard](https://www.tensorflow.org/tensorboard)
* [Weights & Biases](https://docs.wandb.ai/integrations/huggingface)

#### Weights & Biases

To use Weights & Biases, install the wandb package with:

```bash
pip install wandb
```

Then log in the command line:

```bash
wandb login
```

To enable logging to W&B, include `"wandb"` in the `report_to` of your `TrainingArguments` or script. Or just pass along `--report_to all` if you have `wandb` installed.

Advanced configuration is possible by setting environment variables:

| Environment Variable | Value |
|---|---|
| WANDB_LOG_MODEL | Log the model as artifact (log the model as artifact at the end of training) (`false` by default) |
| WANDB_WATCH | one of `gradients` (default) to log histograms of gradients, `all` to log histograms of both gradients and parameters, or `false` for no histogram logging |
| WANDB_PROJECT | Organize runs by project |

Set run names with `run_name` argument present in scripts or as part of `TrainingArguments`.

Additional configuration options are available through generic [wandb environment variables](https://docs.wandb.com/library/environment-variables).

Refer to related [documentation & examples](https://docs.wandb.ai/integrations/huggingface).


## Generate code with finetuned model
To test the finetuned GPT-J model, this script can be iused:
```script
python run_generate.py --model_name_or_path=finetuned --length 200
```

Alternatively, the model can be used in custom code like this to generate text in batches:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("finetuned")
tokenizer.pad_token = "[PAD]"
# Load model
model = AutoModelForCausalLM.from_pretrained("finetuned").to(device)
print("Model loaded")

prompts = ["contract HelloWorld {", "function hello() public"]

# Tokenize
input_ids = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

# Generate
gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=512,
)
generated_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
print(generated_texts)
```
