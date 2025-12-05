import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer,BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch



# # 设置 WandB 为离线模式
# os.environ["WANDB_MODE"] = "offline"

# # 设置项目名称（可选，方便管理）
# os.environ["WANDB_PROJECT"] = "llama2-finetune"



ds_ori= datasets.load_dataset('llm-wizard/alpaca-gpt4-data-zh')
ds = ds_ori['train']

#建议使用本地模型路径以加快加载速度
tokenizer = AutoTokenizer.from_pretrained("./ms-models/shakechen/Llama-2-7b-hf",trust_remote_code=True)
#如果网速良好可选择直接从远程模型加载分词器
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer.padding_side = "right"

tokenizer.pad_token_id = 2


def process_func(example):
    MAX_LENGTH = 1024    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ", add_special_tokens=False)
    response = tokenizer(example["output"], add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

split_ds = tokenized_ds.train_test_split(test_size=0.05, seed=42)
train_ds = split_ds['train']
eval_ds = split_ds['test']


bnb_config= BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


#载入基本模型,建议使用本地模型路径以加快加载速度
model = AutoModelForCausalLM.from_pretrained("./ms-models/shakechen/Llama-2-7b-hf",trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=bnb_config)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=bnb_config)

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,           
    lora_alpha=32, 
    lora_dropout=0.05,
    
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, config)

# model.print_trainable_parameters()

# model.enable_input_require_grads()

args = TrainingArguments(
    output_dir="./chatbot",
    gradient_checkpointing=True,

    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    
    logging_steps=10,
    num_train_epochs=3,
    save_total_limit=2,
    save_strategy="step",
    eval_strategy="step",
    save_steps=500,
    eval_steps=500,
    optim="paged_adamw_32bit",
    bf16=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",

    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    group_by_length=True, 
    dataloader_num_workers=4,
)


trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

trainer.save_model("./chatbot_best_model")

model.eval()
ipt = tokenizer("Human: {}\n{}".format("你好", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)[0], skip_special_tokens=True)


model.merge_and_unload() 