# NLP_LLM_quiz

This repository contains a complete pipeline for fine-tuning Large Language Models (LLMs) for disease diagnosis from radiology findings. The project implements both zero-shot baselines using prompt engineering and parameter-efficient fine-tuning methods (LoRA) on QWen3 models.

The finetuned model can be downloaded at https://huggingface.co/jingdong00/LLMQuiz.
**Setup**
All experiments were conducted using Kaggle Notebooks with the default GPU environment, which provides access to 2× NVIDIA T4 GPUs (each with 16 GB of VRAM), along with approximately 16 GB of RAM and 2 vCPUs.

The following command can install all the necessary libraries.
```bash
python3 setup.py
```

Preprocess the dataset and create train/test splits:

```bash
python3 data_preprocessing.py
```

This will:
- Load the original Excel file (`train-test-data.xlsx`)
- Split data into training (1,000 samples) and test (236 samples) sets
- Create disease vocabulary from all unique labels
- Save processed data in JSON and CSV formats

### 3. Run Zero-Shot Baselines

Evaluate QWen3 models using prompt engineering:

```bash
python3 baseline.py
```

- Supports multiple QWen3 model variants (4B, 8B)
- Implements optimized prompt engineering for medical diagnosis
- Uses batch processing for efficiency
- Generates detailed results with confidence scores

**Models Tested:**
- QWen3-4B (zero-shot)
- QWen3-8B (zero-shot)

### 4. Run Fine-tuning with LoRA

Fine-tune QWen3 models using parameter-efficient LoRA:

```bash
python3 fine_tuning.py
```

**Fine-tuning Configuration:**
- **Base Model**: QWen3-4B
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 16
- **Alpha**: 32
- **Learning Rate**: 2e-4
- **Batch Size**: 1-4 (adjustable based on GPU memory)
- **Epochs**: 3
- **Quantization**: 4-bit quantization for memory efficiency

