# AI-Flow-Ruyi2
The Ruyi2 Large Model (AI-Flow-Ruyi2) was developed by the AI Flow team of the TeleAI Research Institute at China Telecom. It is a familial model designed for the next-generation “end-edge-cloud” model service architecture.


## Usage

Step 1. Create and activate a virtual environment

```sh
conda create -n ruyi python=3.12
conda activate ruyi
```

Step 2. Clone this warehouse to local

```sh
git clone https://github.com/TeleAI-AI-Flow/AI-Flow-Ruyi2.git
cd AI-Flow-Ruyi
```

Step 3. Installation from source (PS: flash_attn compilation and installation is slow, it is recommended to move to the [official repository](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1) to download whl manual installation)

```sh
pip install -e .
```

Step 4. Download model weights

```sh
git clone https://www.modelscope.cn/TeleAI-AI-Flow/AI-Flow-Ruyi-7B-0725.git models/AI-Flow-Ruyi-7B-0725
```

Step 5. Run Demo

```sh
python demo.py
```

<details>
<summary>View demo code</summary>

```py
import torch
from ruyi.global_var import set_global_val
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = f"models/AI-Flow-Ruyi2-14B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16).to('cuda')


generation_config = GenerationConfig(
    do_sample=True,                  
    top_k=30,                        
    top_p=0.95,                      
    temperature=0.6,                 
    repetition_penalty=1.2,          
    no_repeat_ngram_size=3,          
    max_new_tokens=8192
)

# 输入文本
messages = [
    {"role": "user", "content": "你好，请用一句话介绍一下自己。"},
]

# 应用 chat_template 模板
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")

# 模型生成
with torch.no_grad():
    # 设置早退出点
    # - 2: 第一个早退出点，对应约1.7B
    # - 21: 第二个早退出点，对应约8B
    # - 39: 第三个早退出点，对应约14B
    set_global_val("early_exit_point", 39)  

    output = model.generate(
        inputs["input_ids"].to('cuda'),
        generation_config=generation_config
    )

# 解码并打印结果
generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
print(generated_text)
```

</details>

