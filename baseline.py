#!/usr/bin/env python3

import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import re
from typing import List, Dict, Any
import time

class QWenBaseline:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B", fast_mode: bool = True):
        """
        Initialize QWen model for zero-shot baseline
        Available models: Qwen/Qwen3-4B, Qwen/Qwen3-8B
        """
        self.model_name = model_name
        self.fast_mode = fast_mode
        print(f"Loading {model_name}{'(FAST MODE)' if fast_mode else ''}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True
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
    
    def create_prompt(self, finding: str, prompt_version: str = "detailed") -> str:
        """
        Create different prompt versions for experimentation
        """
        if prompt_version == "detailed":
            return f"""You are an expert radiologist. Based on the radiology report below, identify all diseases, conditions, and abnormalities mentioned. Provide your answer as a comma-separated list of specific disease labels.

Radiology Finding:
{finding}

Disease Labels:"""
        
    def generate_response_fast(self, prompt: str) -> str:
        """Optimized generation for speed"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        generation_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
        def generate_response_batch(self, prompts: List[str], batch_size: int = 4) -> List[str]:
        responses = []
        
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches", total=total_batches):
            batch_prompts = prompts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
                pad_to_multiple_of=8
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            for j, output in enumerate(outputs):
                prompt_len = len(batch_prompts[j])
                response = self.tokenizer.decode(output, skip_special_tokens=True)
 
                response = response[prompt_len:].strip() if len(response) > prompt_len else response.strip()
                responses.append(response)
        
        return responses
    
    def extract_disease_labels(self, response: str) -> str:
        response = response.strip()
        response = re.sub(r'^(Disease Labels?:?|Labels?:?|Diseases?:?)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'(\.|\n).*$', '', response).strip()
        
        return response
    
    def evaluate_on_dataset(self, test_data: List[Dict], prompt_version: str = "detailed", 
                          max_samples: int = None, batch_size: int = 4, 
                          use_batching: bool = True) -> Dict[str, Any]:
        results = []
        
        if max_samples is not None:
            test_data = test_data[:max_samples]
        
        print(f"Evaluating with prompt version: {prompt_version}")
        print(f"Batch processing: {'ON' if use_batching else 'OFF'} (batch_size={batch_size})")
        
        start_time = time.time()
        
                if use_batching and batch_size > 1:
            # Prepare prompts
            prompts = []
            for example in tqdm(test_data, desc="Preparing prompts"):
                finding = example['input_finding']
                prompt = self.create_prompt(finding, prompt_version)
                prompts.append(prompt)
            
            print("Generating responses in batches...")
            responses = self.generate_response_batch(prompts, batch_size)
            
            # Process results with progress bar
            for i, example in enumerate(tqdm(test_data, desc="Processing results")):
                predicted_labels = self.extract_disease_labels(responses[i])
                
                result = {
                    'case_id': example.get('case_id', ""),
                    'input_finding': example['input_finding'],
                    'ground_truth': example.get('output_disease', ""),
                    'predicted_labels': predicted_labels,
                    'raw_response': responses[i]
                }
                results.append(result)
        else:
            for example in tqdm(test_data, desc="Processing examples"):
                finding = example['input_finding']
                ground_truth = example.get('output_disease', "")
                
                prompt = self.create_prompt(finding, prompt_version)
                response = self.generate_response_fast(prompt)
                predicted_labels = self.extract_disease_labels(response)
                
                result = {
                    'case_id': example.get('case_id', ""),
                    'input_finding': finding,
                    'ground_truth': ground_truth,
                    'predicted_labels': predicted_labels,
                    'raw_response': response
                }
                results.append(result)
        
        elapsed_time = time.time() - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds")
        print(f"Average time per sample: {elapsed_time/len(test_data):.3f} seconds")
        
        return results

def main():
    print("Loading test data...")
    with open('processed_data/test.json', 'r') as f:
        test_data = json.load(f)

    models_to_test = [
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
    ]
    
    prompt_version = "detailed"
    max_samples = 386
    batch_size = 4 
    use_batching = True
    fast_mode = True
    
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT BASELINE EVALUATION")
    print(f"{'='*60}")
    print(f"Models to test: {len(models_to_test)}")
    print(f"Prompt strategy: {prompt_version}")
    print(f"Max samples: {max_samples} (out of {len(test_data)} total)")
    print(f"Batch size: {batch_size}")
    print(f"Fast mode: {fast_mode}")
    print(f"{'='*60}")
    
    all_results = {}
    
    for model_idx, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*50}")
        print(f"MODEL {model_idx}/{len(models_to_test)}: {model_name}")
        print(f"{'='*50}")
        
        try:
            print("Loading model...")
            baseline = QWenBaseline(model_name, fast_mode=fast_mode)
            print("✓ Model loaded successfully!")
            
            print(f"\nEvaluating with prompt: {prompt_version}")
            print(f"Testing on {min(max_samples, len(test_data))} examples...")
            
            results = baseline.evaluate_on_dataset(
                test_data, 
                prompt_version, 
                max_samples, 
                batch_size=batch_size,
                use_batching=use_batching
            )
            
            model_short = model_name.split('/')[-1]
            results_key = f"{model_short}_{prompt_version}"
            all_results[results_key] = results
            
            # Update output directory
            os.makedirs('results', exist_ok=True)
            with open(f'results/{results_key}_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            df = pd.DataFrame(results)
            df.to_csv(f'results/{results_key}_results.csv', index=False)
            
            print(f"✓ Results saved: {results_key}")
            print(f"✓ Model {model_name} completed successfully!")
            
            del baseline
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Error with model {model_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"BASELINE EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved in 'results/' directory")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 