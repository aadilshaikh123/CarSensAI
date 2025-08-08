import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import re

try:
    import bitsandbytes
except ImportError:
    raise ImportError(
        "The 'bitsandbytes' package is required for 4-bit quantization. "
        "Install it with: pip install bitsandbytes"
    )

hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("Warning: Hugging Face token 'HF_TOKEN' not found in environment variables.")


base_model_id = "meta-llama/Llama-3.2-1b-Instruct"
tokenizer_id = "meta-llama/Llama-3.2-1b-Instruct"

dataset_file = "dataset.jsonl"

output_dir = "./llama-3.2-1b-instruct-cars-finetuned-adapter"
num_train_epochs = 1
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-5
max_seq_length = 1024
logging_steps = 25
save_steps = 50
save_total_limit = 2

use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = True

lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

def clean_text(text):
    text = re.sub(r"<s>\s*\[INST\]\s*", "", text)
    text = re.sub(r"\s*\[/INST\]\s*", "", text)
    text = re.sub(r"\s*</s>", "", text)
    return text.strip()

def format_chat_template(example, tokenizer):
    cleaned_text = clean_text(example['text'])
    instruction = ""
    response = ""

    inst_end_index = cleaned_text.find('?')
    if inst_end_index != -1:
        instruction = cleaned_text[:inst_end_index + 1].strip()
        response = cleaned_text[inst_end_index + 1:].strip()
        if not instruction or not response:
             instruction = cleaned_text
             response = ""
    else:
        instruction = cleaned_text
        response = ""

    if instruction and response:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": formatted_text}
        except Exception as e:
            print(f"Error applying chat template: {e} for text: {cleaned_text}")
            return {"text": None}
    else:
        return {"text": None}


if __name__ == '__main__':

    print("Loading dataset...")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"Dataset loaded with {len(dataset)} examples.")
    print("Sample before cleaning:", dataset[0]['text'])

    dataset = dataset.shuffle(seed=42)

    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
        token=hf_token
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            print("Added pad_token '[PAD]'")
    tokenizer.padding_side = "right"

    print("Formatting dataset with chat template...")
    original_length = len(dataset)

    num_processors = 1

    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        num_proc=num_processors
    )

    dataset = dataset.filter(lambda example: example["text"] is not None)
    filtered_length = len(dataset)
    print(f"Filtered out {original_length - filtered_length} examples during formatting.")
    if filtered_length == 0:
         raise ValueError("No valid examples remaining after formatting. Check the format_chat_template function and your data.")

    print("Sample after formatting:")
    if filtered_length > 0:
        print(dataset[0]['text'])
    else:
        print("Dataset is empty after filtering.")

    print("Configuring quantization...")
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. QLoRA requires a CUDA-enabled GPU.")

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    bf16_supported = torch.cuda.is_bf16_supported()
    if compute_dtype == torch.bfloat16 and use_4bit:
        if bf16_supported:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerating training with bf16=True")
            print("=" * 80)
        else:
            print("=" * 80)
            print("Warning: Your GPU does not natively support bfloat16. Consider using float16.")
            print("Attempting to proceed, but errors might occur. Set bnb_4bit_compute_dtype='float16' if needed.")
            print("=" * 80)

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Configuring training arguments...")
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_8bit",
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=not bf16_supported,
        bf16=bf16_supported,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        save_total_limit=save_total_limit,
        save_strategy="steps",
    )

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_arguments,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning finished.")

    print(f"Saving fine-tuned adapter model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Adapter model and tokenizer saved locally.")

    print("\n--- Testing Fine-tuned Model ---")
    model_for_inference = trainer.model
    model_for_inference.eval()

    prompt_text = "What is the purpose of a car's radiator?"
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\nTesting prompt: {prompt_text}")
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model_for_inference.device)
        attention_mask = inputs.attention_mask.to(model_for_inference.device)
        with torch.no_grad():
            output_ids = model_for_inference.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"Generated Response:\n{generated_text.strip()}")
    except Exception as e:
        print(f"Error during manual inference: {e}")

    print("\nScript finished.")
