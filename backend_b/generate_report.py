import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generatereport(probs, ai_model_name="SpineFractureNet-v2.1"):
    """
    根据输入的椎体预测结果生成放射科报告（新版 OpenAI SDK 版本，无 LangChain，无乱码）
    """

    # ✅ 从环境变量读取 API Key（推荐做法）
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("环境变量 OPENAI_API_KEY 未设置，请在运行容器时传入。")

    # ✅ 初始化客户端（官方 SDK 会自动识别 OPENAI_API_KEY）
    client = OpenAI(api_key=openai_api_key)

    # ✅ 指定语言模型
    llm_model = "gpt-4o-mini"

    # 椎体编号到标准命名映射
    def vertebra_id_to_name(idx: int) -> str:
        if 1 <= idx <= 7:
            return f"C{idx}"
        elif 8 <= idx <= 19:
            return f"T{idx - 7}"
        elif 20 <= idx <= 24:
            return f"L{idx - 19}"
        else:
            return f"Unknown({idx})"

    # 概率分级
    def classify_prob(p: float) -> str:
        if p >= 0.8:
            return "高怀疑骨折"
        elif p >= 0.5:
            return "中度怀疑"
        elif p >= 0.2:
            return "轻度怀疑"
        else:
            return "未见明显异常"

    # 生成结构化描述
    report_input = []
    for item in probs:
        name = vertebra_id_to_name(item["vertebra_id"])
        prob = item["fracture_prob"]
        level = classify_prob(prob)
        report_input.append(f"{name}：预测骨折概率 {prob:.3f}（{level}）")

    predictions_text = "\n".join(report_input)

    # 直接拼接 Prompt
    full_prompt = f"""
你是一名资深放射科医生，熟悉脊柱影像学诊断标准。

根据以下 AI 模型 ({ai_model_name}) 的预测结果，
为“脊柱X线片”生成一份中文放射科报告草稿。

---

模型输出：
{predictions_text}

---

请输出放射科报告，格式如下：

## 报告草稿

### 检查项目： 
脊柱X线片

### 影像所见（Findings）：
- ...

### 印象（Impression）：
- ...

### 建议（Recommendation）：
- 对高怀疑（≥0.8）的椎体建议进一步MRI评估。

最后一行添加：
“本报告基于{llm_model}自动生成，需放射科医生审核签署。”
"""

    # ✅ 直接调用 OpenAI 最新接口
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "你是一名放射科专家。"},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.3,
    )

    report_text = response.choices[0].message.content.strip()

    return report_text


if __name__ == "__main__":
    # 测试数据
    sample_probs = [
        {"vertebra_id": 9, "fracture_prob": 0.12},
        {"vertebra_id": 10, "fracture_prob": 0.93},
    ]
    report = generatereport(sample_probs)
    print("===== 自动生成的放射科报告草稿 =====\n")
    print(report)
