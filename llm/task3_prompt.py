import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 1. 手动复制一条复杂的真实评论用于单条验证
test_review = "刚收到货，快递包装都烂了，送货还慢！不过里面的手机用起来还可以，屏幕挺清晰的，就是价格稍微有点贵。"

# 2. 设计包含格式约束的 Prompt 模板
prompt = f"""你是专业的电商数据流清洗组件。请分析以下买家评价，并提取指定维度的核心特征。

【输入评论】:
{test_review}

【提取要求】:
1. sentiment (情感倾向): 必须且只能在【正面, 负面, 中性】中选择一个。
2. category (问题归属): 必须且只能在【物流, 质量, 价格, 服务, 综合】中选择一个。
3. summary (核心诉求概括): 精炼总结核心诉求，严格限制在 15 个汉字以内。

【输出格式】:
请仅返回一个纯净的 JSON 对象，包含上述三个键。绝不要包含诸如 '好的'、'这是你的JSON' 等任何额外解释性文本，也不要包含 Markdown 代码块标记（如 ```json）。

正确输出示例：
{{"sentiment": "正面", "category": "质量", "summary": "产品质量很好"}}
"""

try:
    # 3. 发起请求，注意开启强制 JSON 输出模式
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}  # 强制 JSON 模式
    )
    
    # 提取返回的文本字符串
    raw_content = response.choices[0].message.content
    print("--- 1. 大模型原始返回字符串 ---")
    print(raw_content)
    print("-" * 30)
    
    # 4. 单条解析闭环验证：反序列化为 Python 字典
    extracted_dict = json.loads(raw_content)
    print("\n--- 2. 成功解析后的 Python 字典 ---")
    print(extracted_dict)
    print("-" * 30)
    
    # 尝试提取字段，验证合法性
    print(f"验证字段成功：\n情感 -> {extracted_dict.get('sentiment')}\n分类 -> {extracted_dict.get('category')}\n摘要 -> {extracted_dict.get('summary')}")

except json.JSONDecodeError:
    print("错误：大模型返回的内容无法被解析为 JSON，请检查 Prompt 约束！")
except Exception as e:
    print(f"请求发生异常: {e}")