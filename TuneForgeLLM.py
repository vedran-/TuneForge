from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os
import subprocess

INPUT_DATASET_NAME='swampattack2.txt'
INPUT_DATASET_SPLITTER='-----\n'
OUTPUT_MODEL_NAME='swamp_model'
# Few-shot prompt
PROMPT_TEMPLATE = """// SwampAttack2 level
// Level: C01L01
// Difficulty: Easy
// Upgrade rank: 0
// Level type: Normal
pause 3
spawn "CrocodileTut" @(6,-2.5,0) 50% 1
tutorial "BasicShooting"
wait all
-----
// SwampAttack2 level
// Level: C01L99
// Difficulty: Hard
// Upgrade rank: 10
// Level type: Normal
"""

# Model configurations
MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "base_model": "Qwen/Qwen2.5-7B",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_qwen25_7b/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_qwen25_7b",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_qwen25_7b.gguf",
    },
    "gemma-3-4b-it": {
        "base_model": "google/gemma-3-4b-it",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_gemma3_4b_it/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_gemma3_4b_it",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_gemma3_4b_it.gguf",
    },
    "gemma-3-4b-pt": {
        "base_model": "google/gemma-3-4b-pt",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_gemma3_4b_pt/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_gemma3_4b_pt",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_gemma3_4b_pt.gguf",
    },
    "gemma-3-12b-it": {
        "base_model": "google/gemma-3-12b-it",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_gemma3_12b_it/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_gemma3_12b_it",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_gemma3_12b_it.gguf",
    },
    "gemma-3-12b-pt": {
        "base_model": "google/gemma-3-12b-pt",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_gemma3_12b_pt/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_gemma3_12b_pt",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_gemma3_12b_pt.gguf",
    },
    "llama-3.3-8b": {
        "base_model": "unsloth/Llama-3.3-8B-bnb-4bit",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_llama33_8b/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_llama33_8b",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_llama33_8b.gguf",
    },
    "deepseek-r1-7b": {
        "base_model": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
        "finetuned_model": f"./{OUTPUT_MODEL_NAME}_deepseek_r1_7b/final",
        "output_dir": f"./{OUTPUT_MODEL_NAME}_deepseek_r1_7b",
        "gguf_output": f"{OUTPUT_MODEL_NAME}_deepseek_r1_7b.gguf",
    },
}

def select_model():
    print("\nAvailable models:")
    model_options = list(MODEL_CONFIGS.keys())
    for i, model in enumerate(model_options, start=97):  # ASCII 'a' = 97
        print(f"  {chr(i)}. {model}")
    choice = input("Select a model by letter (a-g): ").strip().lower()
    while choice not in [chr(i) for i in range(97, 104)]:  # 'a' to 'g'
        print("Invalid choice. Please select a letter from a to g.")
        choice = input("Select a model by letter (a-g): ").strip().lower()
    return model_options[ord(choice) - 97]

def get_num_epochs():
    while True:
        epochs = input("Enter number of epochs (default 10, press Enter for default): ").strip()
        if not epochs:
            return 10
        try:
            epochs = int(epochs)
            if epochs > 0:
                return epochs
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_training_mode(model_key):
    print("\nTraining mode options:")
    print("  1. Normal: Batch size 8 (2×4) - For 7B and smaller models on 24GB VRAM.")
    print("  2. Lowmem: Batch size 4 (1×4) - For 12B models on 24GB VRAM, or if Normal fails.")
    if "12b" in model_key.lower():
        print("  (Recommended: Lowmem for 12B models like", model_key, ")")
    else:
        print("  (Recommended: Normal for 7B and smaller models like", model_key, ")")
    choice = input("Select mode (1 for Normal, 2 for Lowmem): ").strip()
    while choice not in ["1", "2"]:
        print("Invalid choice. Please enter 1 or 2.")
        choice = input("Select mode (1 for Normal, 2 for Lowmem): ").strip()
    return "normal" if choice == "1" else "lowmem"

def finetune_model(model_key):
    config = MODEL_CONFIGS[model_key]
    print(f"\nFine-tuning {model_key}...")

    # User inputs
    num_epochs = get_num_epochs()
    mode = get_training_mode(model_key)
    batch_size = 2 if mode == "normal" else 1
    grad_steps = 4  # Effective batch size: 8 (normal) or 4 (lowmem)

    # Load the model
    is_gemma3 = "gemma-3" in model_key
    if is_gemma3:
        print("Detected Gemma-3 model; applying Unsloth compatibility fixes.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth" if is_gemma3 else True,
    )

    # Load and preprocess dataset
    with open(INPUT_DATASET_NAME, "r", encoding="utf-8") as f:
        levels = f.read().split(INPUT_DATASET_SPLITTER)[1:-1]
    dataset = Dataset.from_dict({"text": levels})

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_steps,
        num_train_epochs=num_epochs,
        learning_rate=2e-5,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=20,
        weight_decay=0.01,
        gradient_checkpointing=True,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(config["finetuned_model"])
    tokenizer.save_pretrained(config["finetuned_model"])
    print(f"Model saved to {config['finetuned_model']}")

def generate_levels(model_key):
    config = MODEL_CONFIGS[model_key]
    model_path = config["finetuned_model"] if os.path.exists(config["finetuned_model"]) else config["base_model"]
    print(f"\nGenerating levels with {model_key} using {model_path}...")

    # Load the model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        model = FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Fix tokenizer padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize with attention mask
    inputs = tokenizer(
        PROMPT_TEMPLATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        return_attention_mask=True
    ).to("cuda")

    # Generate
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=2048,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.5,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

    # Save to file
    with open(f"generated_level_{model_key}.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

def convert_to_gguf(model_key):
    config = MODEL_CONFIGS[model_key]
    if not os.path.exists(config["finetuned_model"]):
        print(f"\nError: Fine-tuned model not found at {config['finetuned_model']}. Please fine-tune the model first.")
        return

    print(f"\nConverting {model_key} to GGUF...")

    # Load the fine-tuned model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config["finetuned_model"],
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return

    # Merge LoRA adapters into base model
    print("Merging LoRA adapters...")
    model = model.merge_and_unload()

    # Save merged model in Hugging Face format
    merged_dir = f"{config['output_dir']}/merged_hf"
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")

    # Convert to GGUF using llama.cpp
    llama_cpp_path = "./llama.cpp"
    if not os.path.exists(llama_cpp_path):
        print(f"Error: llama.cpp not found at {llama_cpp_path}. Please install it first.")
        return

    gguf_output = config["gguf_output"]
    convert_script = f"{llama_cpp_path}/convert-hf-to-gguf.py"
    if not os.path.exists(convert_script):
        print(f"Error: convert-hf-to-gguf.py not found at {convert_script}.")
        return

    try:
        subprocess.run([
            "python", convert_script,
            merged_dir,
            "--outfile", gguf_output,
            "--quantize", "q4_0"
        ], check=True)
        print(f"GGUF model saved to {gguf_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error during GGUF conversion: {e}")

def main_menu():
    while True:
        print("\n=== TuneForge LLM ===")
        print("1. Fine-tune a model")
        print("2. Complete prompt with finetuned model")
        print("3. Convert to GGUF")
        print("4. Exit")
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            model_key = select_model()
            finetune_model(model_key)
        elif choice == "2":
            model_key = select_model()
            generate_levels(model_key)
        elif choice == "3":
            model_key = select_model()
            convert_to_gguf(model_key)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")

if __name__ == "__main__":
    main_menu()