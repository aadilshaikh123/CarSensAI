
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set your base model ID (same as in finetune.py)
base_model_id = "meta-llama/Llama-3.2-1b-Instruct"
save_dir = "./llama-3.2-1b-base"

# Get Hugging Face token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise EnvironmentError("HF_TOKEN environment variable not set.")

print(f"Downloading model and tokenizer for: {base_model_id}")

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    token=hf_token
)
tokenizer.save_pretrained(save_dir)
print("Tokenizer downloaded and saved.")

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    token=hf_token
)
model.save_pretrained(save_dir)
print("Model downloaded and saved.")

print(f"Done. Model and tokenizer saved to: {save_dir}")

