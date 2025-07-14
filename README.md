# NLP_LLM_quiz

This repository contains a complete pipeline for fine-tuning Large Language Models (LLMs) for disease diagnosis from radiology findings. The project implements both zero-shot baselines using prompt engineering and parameter-efficient fine-tuning methods (LoRA) on QWen3 models.

The finetuned model can be downloaded at https://huggingface.co/jingdong00/LLMQuiz.
**Setup**
All experiments were conducted using Kaggle Notebooks with the default GPU environment, which provides access to 2× NVIDIA T4 GPUs (each with 16 GB of VRAM), along with approximately 16 GB of RAM and 2 vCPUs.

The following command can install all the necessary libraries.
```bash
python3 setup.py
```

**Run Qwen3 Baselines**
```bash
python3 baseline.py
```

**Run Qwen3 Finetune with LoRA**
```bash
python3 fine_tuning.py
```
