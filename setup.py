#!/usr/bin/env python3

import subprocess
import sys

def install_requirements():
    packages = [
        'torch>=2.0.0',
        'transformers>=4.35.0',
        'peft>=0.6.0',
        'datasets>=2.14.0',
        'accelerate>=0.25.0',
        'bitsandbytes>=0.41.0',
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tqdm>=4.65.0',
        'openpyxl>=3.1.0',
        'huggingface-hub>=0.19.0',
        'sentencepiece>=0.1.99',
        'protobuf>=3.20.0'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

if __name__ == "__main__":
    install_requirements()
    print("All packages installed successfully!") 