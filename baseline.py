#!/usr/bin/env python3

import json
import os
import re
from typing import List, Dict, Any

import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import MultiLabelBinarizer


class QWenBaseline:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B", vocab_path: str = None, fast_mode: bool = True):
        self.model_name = model_name
        self.fast_mode = fast_mode
        print(f"Loading {model_name}{'(FAST MODE)' if fast_mode else ''}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if fast_mode:
            model_kwargs.update({
                "low_cpu_mem_usage": True,
                "use_cache": True,
            })

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded")

        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_list = json.load(f)
            self.disease_vocab = set(d.lower().strip() for d in vocab_list)
            print(f"Loaded disease vocabulary with {len(self.disease_vocab)} entries.")
        else:
            self.disease_vocab = None
            print("No disease vocabulary loaded. Skipping vocabulary filtering.")

    def create_prompt(self, finding: str) -> str:
        return f"""You are an expert radiologist extracting confirmed medical diagnoses from a CT report.
    
    Only list diagnoses clearly supported by the report.
    
    Output format:
    - List precise, standard clinical diagnoses separated by commas.
    - No numbers, no bullets, no extra text.
    - Use terms like "Liver calcifications", "Renal cyst".
    - Do NOT include vague terms like "mass" or "finding".
    - Do NOT specify anatomical positions.
    - If no diagnosis, respond with "None".
    
    Example:
    Liver calcifications, Renal cyst, Fatty liver
    
    CT Report:
    {finding}
    
    Diagnoses:"""

    def generate_response_batch(self, prompts: List[str], max_new_tokens=48, batch_size: int = 4) -> List[str]:
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                pad_to_multiple_of=8,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            for j, output in enumerate(outputs):
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                prompt_len = len(self.tokenizer.decode(inputs['input_ids'][j], skip_special_tokens=True))
                response = decoded[prompt_len:].strip().split('\n')[0].strip()
                responses.append(response)
        return responses


    def extract_disease_labels(self, response: str, vocab_set: set) -> str:
        raw = response.strip().splitlines()
        candidates = []

        for line in raw:
            line = line.strip("\u2022- ").strip()
            if not line:
                continue
            if "," in line:
                candidates += [s.strip() for s in line.split(",") if s.strip()]
            else:
                candidates.append(line)

        matched_terms = set()
        for cand in candidates:
            clean = re.sub(r"[^\w\s]", "", cand).lower().strip()
            if clean in vocab_set:
                matched_terms.add(clean)

        return ", ".join(sorted(term.title() for term in matched_terms))

    def evaluate_on_dataset(self, test_data: List[Dict], max_samples: int = None, batch_size: int = 4, max_retries: int = 3) -> List[Dict[str, Any]]:
        if max_samples:
            test_data = test_data[:max_samples]

        prompts = [self.create_prompt(d["input_finding"]) for d in test_data]
        print("Generating initial responses...")
        responses = self.generate_response_batch(prompts, batch_size=batch_size)

        to_retry_idx = [i for i, r in enumerate(responses) if len(r.strip()) < 3]
        retry_count = 0

        while to_retry_idx and retry_count < max_retries:
            retry_count += 1
            print(f"Retrying {len(to_retry_idx)} prompts, attempt {retry_count}/{max_retries}...")
            retry_prompts = [prompts[i] for i in to_retry_idx]
            retry_responses = self.generate_response_batch(retry_prompts, max_new_tokens=64, batch_size=batch_size)
            for idx, resp in zip(to_retry_idx, retry_responses):
                responses[idx] = resp
            to_retry_idx = [i for i in to_retry_idx if len(responses[i].strip()) < 3]

        results = []
        y_true, y_pred = [], []
        all_labels = set()

        for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
            pred_str = self.extract_disease_labels(responses[i], self.disease_vocab)
            true_str = example.get("output_disease", "")
            pred_labels = set(map(str.strip, pred_str.lower().split(","))) if pred_str else set()
            true_labels = set(map(str.strip, true_str.lower().split(","))) if true_str else set()
            all_labels.update(pred_labels)
            all_labels.update(true_labels)
            y_true.append(list(true_labels))
            y_pred.append(list(pred_labels))
            results.append({
                "case_id": example.get("case_id", ""),
                "input_finding": example["input_finding"],
                "ground_truth": true_str,
                "predicted_labels": ", ".join(pred_labels),
                "raw_response": responses[i],
                "raw_prompt": self.tokenizer.apply_chat_template(prompts[i], tokenize=False, enable_thinking=False),
            })

        mlb = MultiLabelBinarizer(classes=sorted(all_labels))
        mlb.fit([])
        y_true_bin = mlb.transform(y_true)
        y_pred_bin = mlb.transform(y_pred)

        precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
        recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)

        print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return results


def main():
    with open('/kaggle/working/processed_data/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    vocab_path = '/kaggle/working/processed_data/disease_vocabulary.json'
    models_to_test = ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]
    max_samples = 386
    batch_size = 4
    fast_mode = True

    for model_name in models_to_test:
        baseline = QWenBaseline(model_name=model_name, vocab_path=vocab_path, fast_mode=fast_mode)
        results = baseline.evaluate_on_dataset(test_data, max_samples=max_samples, batch_size=batch_size)
        model_short = model_name.split('/')[-1]
        results_key = f"{model_short}_zeroshot"
        os.makedirs('/kaggle/working/results', exist_ok=True)

        with open(f'/kaggle/working/results/{results_key}_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        pd.DataFrame(results).to_csv(f'/kaggle/working/results/{results_key}_results.csv', index=False)


if __name__ == "__main__":
    main()
