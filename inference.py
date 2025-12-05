import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TextIteratorStreamer
from peft import PeftModel
import os
from threading import Thread
base_model_path = "e:/AI-models/ms-models/shakechen/Llama-2-7b-hf"


lora_path = "./checkpoint/leo1.0"

print("正在加载模型，请稍候...")

# ================= 1. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 2. 加载底座模型 (Base Model) =================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # 和训练时一致
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ================= 3. 加载并挂载 LoRA (Adapter) =================

model = PeftModel.from_pretrained(base_model, lora_path)

# 切换到评估模式
model.eval()

print("模型加载完成！")
# ================= 4. 定义推理函数 =================

# def chat(instruction, mode_id, history_str=""):
#     input_text = ""
#     # 构建当前轮的 prompt
#     new_prompt_part = "\n".join(["Human: " + instruction, input_text]).strip() + "\n\nAssistant: "
    
#     # 逻辑分支
#     if mode_id == 1:
#         # 单轮模式：不依赖历史，只看当前
#         current_prompt = new_prompt_part
#     else:
#         # 多轮模式：拼接历史
#         current_prompt = history_str + new_prompt_part
#         # 检查长度
#         inputs_check = tokenizer(current_prompt, return_tensors="pt")
#         if inputs_check.input_ids.shape[1] > 2048:
#             print("Leo1.0:警告,上下文过长影响思考效果，建议重启对话")

#     # 打印测试 (调试用)
#     # print(f'DEBUG: 当前喂给模型的Prompt长度: {len(current_prompt)}')

#     # 2. 编码
#     inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)

#     # 3. 生成
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=3500,      
#             do_sample=True,
#             temperature=1,         
#             top_p=0.9,
#             top_k=40,
#             repetition_penalty=1.1,
#             eos_token_id=tokenizer.eos_token_id
#         )
    
#     # 4. 解码
#     output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
#     # 修改点2: 计算并返回新的历史记录
#     new_history = history_str
#     if mode_id == 2:
#         new_history = current_prompt + output_text + "\n"
        
#     return output_text, new_history

# # ================= 5. 测试 =================
# if __name__ == "__main__":
#     os.system('clear')
#     print(f'Leo1.0:我是基于Llama-2-7b模型微调的聊天机器人Leo1.0,可以回答各种问题')
#     print(f'Leo1.0:请你选择单轮无记忆对话模式(1)或多轮记忆对话模式(2)')
#     mode = input(f'Leo1.0:输入1或2后回车即可\n')

    
#     if mode != '2':
#         # === 单轮模式 ===
#         if mode != '1':
#             print(f'Leo1.0:输入有误，默认启动单轮无记忆对话模式')
#         print(f'Leo1.0:单轮无记忆对话模式已启动,输入bye退出')
#         while True:
#             q = input("user: ")
#             if q.lower() == 'bye':
#                 print("Leo1.0: 再见！期待下次和你聊天。")
#                 break
#             print("深度思考中，请稍候...")
            
            
#             response, _ = chat(q, 1, "")
#             print(f"Leo1.0: {response}")
            
#     else:
        
#         count = 0
#         history_prompt = ""
#         print(f'Leo1.0:多轮记忆对话模式已启动,你可以和我进行多轮对话,输入bye退出')
        
#         while True:
#             q = input("user: ")
#             if q.lower() == 'bye':
#                 print("Leo1.0: 再见！期待下次和你聊天。")
#                 break
#             if count >= 5: # 限制轮数
#                 print("Leo1.0: 多轮对话次数达限制,已退出")
#                 break
            
#             count = count + 1
#             print("深度思考中，请稍候...")
            
#             # 接收返回的 updated_history 并赋值给 history_prompt
#             response, history_prompt = chat(q, 2, history_prompt)
#             print(f"Leo1.0: {response}")



            
def chat_stream(instruction, mode_id, history_str=""):

    input_text = ""
    new_prompt_part = "\n".join(["Human: " + instruction, input_text]).strip() + "\n\nAssistant: "
    
    if mode_id == 1:
        current_prompt = new_prompt_part
    else:
        current_prompt = history_str + new_prompt_part
        # 简单截断防止爆显存
        if len(current_prompt) > 4000: 
            current_prompt = current_prompt[-4000:] 

    inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=512,      # 限制长度，太长会慢
        do_sample=True,
        temperature=1,
        top_p=0.9,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id
    )


    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Leo1.0: ", end="", flush=True) # 先打印个头
    
    full_response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True) # 实时打印，不换行
        full_response += new_text
    
    print() # 结尾换行


    new_history = history_str
    if mode_id == 2:
        new_history = current_prompt + full_response + "\n"
        
    return new_history


if __name__ == "__main__":
    os.system('clear')
    print(f'Leo1.0:我是基于Llama-2-7b模型微调的聊天机器人Leo1.0,可以回答各种问题')
    print(f'Leo1.0:请你选择单轮无记忆对话模式(1)或多轮记忆对话模式(2)')
    mode = input(f'Leo1.0:输入1或2后回车即可\n')

    
    if mode != '2':
        # === 单轮模式 ===
        if mode != '1':
            print(f'Leo1.0:输入有误，默认启动单轮无记忆对话模式')
        print(f'Leo1.0:单轮无记忆对话模式已启动,输入bye退出')
        while True:
            q = input("user: ")
            if q.lower() == 'bye':
                print("Leo1.0: 再见！期待下次和你聊天。")
                break
            
            
            _ = chat_stream(q, 1, "")
    else:
        
        count = 0
        history_prompt = ""
        print(f'Leo1.0:多轮记忆对话模式已启动,你可以和我进行多轮对话,输入bye退出')
        
        while True:
            q = input("user: ")
            if q.lower() == 'bye':
                print("Leo1.0: 再见！期待下次和你聊天。")
                break
            if count >= 5: # 限制轮数
                print("Leo1.0: 多轮对话次数达限制,已退出")
                break
            
            count = count + 1
            
            # 接收返回的 updated_history 并赋值给 history_prompt
            history_prompt = chat_stream(q, 2, history_prompt)
