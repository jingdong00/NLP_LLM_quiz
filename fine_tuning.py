import os
import gc
import json
import logging
import re
from typing import List, Dict

import torch
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

gc.collect()
torch.cuda.empty_cache()
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_MAX_MEMORY"] = '{"cuda:0": "14GiB"}'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_WORK_DIR = ""
RESULTS_DIR = os.path.join(BASE_WORK_DIR, "results")
MODELS_DIR = os.path.join(BASE_WORK_DIR, "models")
PROCESSED_DATA_DIR = os.path.join(BASE_WORK_DIR, "processed_data")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()

class QWenFineTuner:
    def __init__(self, model_path="/models/Qwen3-4B/Qwen3-4B"):
        self.model_name = model_path
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.is_full_finetune = False

    def load_model_and_tokenizer(self):
        logger.info(f"Loading model and tokenizer from: {self.model_name}")
        clear_mem()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        logger.info("Model and tokenizer loaded from local files.")
        clear_mem()

    def load_fine_tuned_model(self, model_dir):
        logger.info(f"Loading fine-tuned LoRA model from {model_dir}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        base_model_name = PeftConfig.from_pretrained(model_dir).base_model_name_or_path
        if base_model_name == "Qwen/Qwen3-4B":
            base_model_name = "/root/models/Qwen3-4B/Qwen3-4B"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        self.peft_model = PeftModel.from_pretrained(self.model, model_dir)
        self.peft_model.eval()
        self.is_full_finetune = False

    def setup_lora_config(self, use_dora=False):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            use_dora=use_dora,
        )

    def create_prompt_template(self, finding, disease_labels=None):
        base_prompt = (
            "You are an expert radiologist analyzing medical findings. Based on the radiology report below, "
            "identify all diseases, conditions, and abnormalities mentioned. Provide your answer as a comma-separated list of specific disease labels.\n\n"
            "Guidelines:\n"
            "- Focus on definitive diagnoses and clear abnormalities\n"
            "- Use standard medical terminology\n"
            "- Separate multiple diseases with commas\n"
            "- Do not include normal findings\n"
            "- Be concise but specific\n\n"
            "Radiology Finding:\n"
            f"{finding}\n\n"
            "Disease Labels:"
        )
        if disease_labels:
            return f"{base_prompt} {disease_labels}"
        return base_prompt

    def prepare_training_data(self, train_data: List[Dict], max_length=256):
        logger.info(f"Preparing training data with max_length={max_length}")
        texts = [
            self.create_prompt_template(example['input_finding'], example.get('output_disease'))
            for example in train_data
        ]
        dataset = Dataset.from_dict({"text": texts})

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        return tokenized_dataset

    def fine_tune_model(self, train_dataset, output_dir="fine_tuned_model",
                        use_dora=False, num_epochs=3, batch_size=1,
                        full_finetune=False, resume_from_checkpoint=None):
        logger.info(f"Fine-tuning model full_finetune={full_finetune} use_dora={use_dora}")
        if full_finetune:
            self.is_full_finetune = True
            model_to_train = self.model
        else:
            self.is_full_finetune = False
            lora_config = self.setup_lora_config(use_dora=use_dora)
            self.peft_model = get_peft_model(self.model, lora_config)
            self.peft_model.print_trainable_parameters()
            model_to_train = self.peft_model

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4 if not full_finetune else 5e-5,
            logging_steps=10,
            save_steps=100,
            save_total_limit=1,
            prediction_loss_only=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=None,
            dataloader_pin_memory=False,
            bf16=False,
            fp16=True,
            gradient_checkpointing=False,
            optim="adamw_bnb_8bit",
            save_strategy="steps",
            eval_strategy="no",
            load_best_model_at_end=False,
            disable_tqdm=False,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=model_to_train,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting training loop...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training finished. Saving model...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        clear_mem()
        return trainer

    def generate_prediction(self, finding, max_length=128):
        model_to_use = self.model if self.is_full_finetune else self.peft_model
        if model_to_use is None:
            raise RuntimeError("Model not loaded. Call load_fine_tuned_model() first.")

        prompt = self.create_prompt_template(finding)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def evaluate_on_test_set(self, test_data: List[Dict], model_dir=None):
        if model_dir:
            self.load_fine_tuned_model(model_dir)

        logger.info("Evaluating on test set...")
        results = []
        for example in tqdm(test_data, desc="Evaluating"):
            finding = example['input_finding']
            ground_truth = example.get('output_disease', "")
            prediction = self.generate_prediction(finding)
            prediction_clean = re.sub(r'^(Disease Labels?:?|Labels?:?|Diseases?:?)', '', prediction, flags=re.IGNORECASE)
            prediction_clean = re.sub(r'(\.|\n).*$', '', prediction_clean).strip()

            results.append({
                'case_id': example.get('case_id', ""),
                'input_finding': finding,
                'ground_truth': ground_truth,
                'predicted_labels': prediction_clean,
                'raw_response': prediction
            })

        return results

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, checkpoints[-1])

def main():
    clear_mem()
    torch.cuda.synchronize()
    logger.info("Loading training and test data...")
    with open(os.path.join(PROCESSED_DATA_DIR, 'train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(PROCESSED_DATA_DIR, 'test.json'), 'r') as f:
        test_data = json.load(f)
    test_data = test_data[20:150]
    print(f"Loaded {len(train_data)} training samples")

    model_path = "/root/models/Qwen3-4B/Qwen3-4B"  # local path to model directory
    fine_tuner = QWenFineTuner(model_path)
    fine_tuner.load_model_and_tokenizer()

    train_dataset = fine_tuner.prepare_training_data(train_data, max_length=1024)
    output_dir = os.path.join(MODELS_DIR, "Qwen3-4B_qlora")

    resume_checkpoint = find_latest_checkpoint(output_dir)
    print(resume_checkpoint)
    if resume_checkpoint:
        logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")
    else:
        logger.info("No checkpoint found. Starting fresh.")

    fine_tuner.fine_tune_model(
        train_dataset,
        output_dir=output_dir,
        use_dora=False,
        num_epochs=3,
        batch_size=1,
        resume_from_checkpoint=resume_checkpoint
    )

    logger.info("Evaluation starting...")
    results = fine_tuner.evaluate_on_test_set(test_data, model_dir=output_dir)
    print(results)
    results_json = os.path.join(RESULTS_DIR, "Qwen3-4B_qlora_results.json")
    print(f"Saving results to {results_json}")
    with open(results_json, 'w+') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "Qwen3-4B_qlora_results.csv"), index=False)
    logger.info("Finished saving results.")

if __name__ == "__main__":
    main()
