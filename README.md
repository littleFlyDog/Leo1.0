# Leo1.0

## Model introduction
**This is a model fine-tuned using QLoRA based on the LLaMA-2-7b.**,this project can help you experience the process of fine-tuning large models, and it only requires 16GB of VRAM and 18GB of RAM,and you can even reduce VRAM usage further by adjusting the settings.
1. pretrained model 
- [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. datasets
- [llm-wizard/alpaca-gpt4-data-zh](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh)

## Hardware
- RTX 3090

## Install
- You can refer to requirements.txt
```bash
pip install torch transformers datasets peft wandb 
```
## Quick start
```bash
python ft.py
```
## Fine-tuning outcome
1. train loss
2. eval loss

## examples
