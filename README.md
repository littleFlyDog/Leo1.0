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
python inferency.py
```
## Fine-tuning outcome
1. train loss
![trainloss](assets/trainloss.png)
2. eval loss
![evalloss](assets/evalloss.png)

## examples
```
Leo1.0:我是基于Llama-2-7b模型微调的聊天机器人Leo1.0,可以回答各种问题
Leo1.0:请你选择单轮无记忆对话模式(1)或多轮记忆对话模式(2)
Leo1.0:输入1或2后回车即可
1
Leo1.0:单轮无记忆对话模式已启动,输入bye退出
user: 武汉今天温度为零下10摄氏度,应该穿什么衣服?
深度思考中，请稍候...
Leo1.0: 在武汉的温度降到零下10摄氏度时，建议穿一条厚重的外套和其他相关的冰雪保护装备，比如冬袜、靴子、手套等。此外，你也需要确保身体有足够的保暖效果，所以还可以考虑选用牛仔裤或者保温围甲的运动装备，以及充足的防寒衣物。不过，最好是查看当地的天气预报并按照气象部门的温度建议进行搭配，才能确定最佳的衣物选择。
user: 8+5等于多少?
深度思考中，请稍候...
Leo1.0: 两个数字相加的结果是13.
user: 如何追求女生?
深度思考中，请稍候...
Leo1.0: 在要想對一个女性申請，追求或者準備接受她的邀請之前，你必須了解和理解有关人際交往、情侶關係和正式拘束的方面。除了具备聲望、外表和金錢能力外，真正成功地追求一位憧憬的女性还需要知道如何說服她去與自己結合。以下是一些有助于追求女性的技巧和建議：

1. 用好口語 - 良好的口才、身手配合和说话技巧能使你更加出色。在初次推銷時，不要直接告訴她你想意儀。多讓她想象一些。

2. 温柔的接吻 - 敏銳地观察到她是否打算与你接吻并向她提出接吻是一项重要的技巧。假如她沒有做出明确的反应，不要立刻開始。等她做出回应，再次請求她接吻就可以。

3. 透明性和隱瞒 - 說清楚是非常重要。保持透明度能讓两人在日后也没有困難。同时，也要控制逃避和隱瞒。承認你的誤解和错誤，并在必要時向她道歉是非常重要的技巧。

4. 靠近她 - 作为男子，擁有自信和勇氣，接近她是非常重要的。不管你是在召唤她或在離別，都要靠近她。那麼她就会知道你的意味。
```