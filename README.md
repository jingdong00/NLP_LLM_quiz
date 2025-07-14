# NLP_LLM_quiz

This repository contains a complete pipeline for fine-tuning Large Language Models (LLMs) for disease diagnosis from radiology findings. The project implements both zero-shot baselines using prompt engineering and parameter-efficient fine-tuning methods (LoRA) on QWen3 models.

**Setup**
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
