# AI-Flow-Ruyi2 (Ruyi2大模型)

## 介绍

**Ruyi2大模型（AI-Flow-Ruyi2）** 是中国电信人工智能研究院 (TeleAI) 智传网（AI Flow）团队研发，是面向下一代“端-边-云”模型服务架构的**同源家族模型（Familial Model）**。该模型基于 Qwen3-14B 基座构建，确立了“一次训练，多处部署”（Train Once, Deploy Many）的全新范式。其核心在于 1.7B（端）、8B（边）与 14B（云）三个嵌套分支共享主干参数，通过动态早退出机制实现不同算力环境下的自适应推理。针对端侧部署，模型创新性引入 DaE（Decompose after Expansion）框架，结合稳定块扩展（SBE）与 SVD 后训练压缩技术，在降低 40% 增量参数的同时显著增强了小模型的逻辑推理能力，从而实现了高效的端边云协同与高性能的边缘智能落地。

## Ruyi2

为了进一步推动下一代“端-边-云”协同计算范式的落地，让业界体验更高效的家族模型架构，我们正式开源了 Ruyi2 (AI-Flow-Ruyi2) 模型。Ruyi2 于 2026 年 2 月 14 日发布，基于 Qwen3-14B 构建。其最大参数量分支为 14B，并可分化出具有等效参数量为 1.7B 和 8B 的早退出分支。其中：
* 1.7B 分支：专为端侧设备（End Devices）部署设计，通过创新的DaE（扩展后分解）与SVD压缩技术优化，在极低的资源占用下保留了高密度的知识与推理能力，优势在于极致的响应速度与部署效率；
* 8B 分支：定位为边缘服务器（Edge Server）的主力模型，在MMLU与GSM8K等基准测试中表现优异，实现了在通用任务场景下性能与计算成本的完美平衡；
* 14B 分支：作为云端服务器（Cloud Server）的完全体，主攻复杂高阶应用，在逻辑推理、数学及综合知识维度上展现出统治级的全面优势，适合处理最棘手的难题。

|位点序号|早退出位置|等效模型大小|对应分支代号|场景定位|
|:-:|:-:|:-:|:-:|:-:|
|1|2层|1.7B|AI-Flow-Ruyi2-1.7B|端侧极速/简单任务 |
|2|21层|8B|AI-Flow-Ruyi2-8B|边缘计算/通用平衡 |
|3|层|14B|AI-Flow-Ruyi2-14B|云端全能/复杂问题|


### 训练过程

在训练开始前，我们基于Qwen团队预训练的[Qwen3-14B-Base](https://doi.org/10.48550/arXiv.2505.09388)模型，对 14B 主分支进行了参数初始化；对于 1.7B 和 8B 早退出分支，其解码器层均采用早退出位置的下一层参数进行初始化。

完成初始化后，我们采用**多分支联合预训练**方法，在私有高质量数据集上进行了约 8000亿 (800B) token 的继续预训练，构建出Ruyi2 基座（Ruyi2-Base）。

随后，我们基于约 400万 (4M) 条高质量指令数据，对各分支进行了**联合指令遵循微调**；在此基础上，针对 1.7B 分支特别引入了 DaE（扩展后分解）框架与 SVD 压缩技术，并结合 GRPO 强化学习 进一步增强了模型的逻辑推理能力，最终得到 Ruyi2。

### 性能评测

我们基于[OpenCompass](https://github.com/open-compass/opencompass)及其官方配置文件，Ruyi2 模型家族相比 Qwen3 基线模型所具备的卓越扩展效率和强劲性能。在所有参数规模上，Ruyi2 始终在知识理解和推理能力方面树立了新的行业标杆。
* Ruyi2-1.7B：端侧极致效能与“知识密度”的突破 作为专为移动端与边缘设备设计的轻量化分支，Ruyi2-1.7B 展现了无与伦比的“知识密度” 。得益于创新的 DaE（扩展后分解）框架与 SVD 后训练压缩技术，它在极小的参数规模下保留了惊人的知识容量，MMLU 得分高达 62.77，超越同级基座模型（Qwen3-1.7B）超过 23 个百分点 。这证明了 Ruyi2-1.7B 成功打破了小模型的性能天花板，成为在严格内存与延迟限制下处理知识密集型任务的高效解决方案；
* Ruyi2-8B：边缘侧的全面超越与黄金平衡 Ruyi2-8B 是家族模型中性能与效率的完美平衡点，实现了对同级竞品的全面超越 。定位为边缘服务器的主力模型，它不仅在通用知识理解上遥遥领先（MMLU 79.68），更在逻辑推理能力上实现了质的飞跃，GSM-8K 得分达到 92.19，显著优于基线模型。它以中等规模的算力开销提供了接近顶尖大模型的推理体验，验证了智传网架构在多维能力平衡上的卓越设计；
* Ruyi2-14B：云端旗舰的全指标统治力 作为家族中的全量旗舰，Ruyi2-14B 在所有关键评测指标上确立了绝对的统治地位。它充分释放了共享主干网络的全部潜能，不仅进一步刷新了通用知识的上限（MMLU 81.84），更在数学与复杂逻辑推理任务中展现了压倒性优势，其 GSM-8K（94.24）和 Math（86.52）得分均超越了 Qwen3-14B。Ruyi2-14B 是处理高阶复杂问题、长逻辑链推理的最稳健选择，代表了该系列模型的最高智能水平。

<details>
<summary>性能比较: Ruyi2 vs Qwen3</summary>

| Model | MMLU | MMLU-P | CMMLU | BBH | ARC-c | Hella | IFEval | Human | Math | GSM8K | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-1.7B | 39.31 | 39.82 | 61.61 | 35.81 | 67.12 | 54.25 | 67.65 | 62.20 | 70.32 | 75.89 | 57.40 |
| Qwen3-8B | 48.64 | 55.63 | 78.84 | 55.11 | 82.03 | 77.32 | 81.89 | 87.20 | 82.30 | 85.37 | 73.43 |
| Qwen3-14B | 60.43 | 64.11 | 82.06 | 64.70 | 81.69 | 80.80 | 85.77 | 87.80 | 84.42 | 85.90 | 77.77 |
| **Ruyi2-1.7B** | 62.77 | 9.60 | 22.68 | 19.95 | 27.46 | 58.77 | 47.13 | 47.56 | 39.14 | 75.97 | 41.10 |
| **Ruyi2-8B** | 79.68 | 56.12 | 74.72 | 59.38 | 82.71 | 78.19 | 73.94 | 71.95 | 72.96 | 92.19 | 74.18 |
| **Ruyi2-14B** | 81.84 | 71.55 | 82.15 | 77.86 | 84.41 | 83.94 | 81.52 | 84.76 | 86.52 | 94.24 | 82.88 |

</details>




## 使用

Step 1. 创建并激活虚拟环境

```sh
conda create -n ruyi python=3.12
conda activate ruyi
```

Step 2. 克隆本仓库至本地

```sh
git clone https://github.com/TeleAI-AI-Flow/AI-Flow-Ruyi2.git
cd AI-Flow-Ruyi
```

Step 3. 由源码安装（PS: flash_attn编译安装较慢，建议移步[官方仓库](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1)下载whl手动安装）

```sh
pip install -e .
```

Step 4. 下载模型权重

```sh
git clone https://huggingface.co/TeleAI-AI-Flow/AI-Flow-Ruyi2 models/AI-Flow-Ruyi2
```

Step 5. 运行Demo

```sh
python demo.py
```

<details>
<summary>查看Demo代码</summary>

```py
import torch
from ruyi.global_var import set_global_val
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = f"models/AI-Flow-Ruyi2"
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
