import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

def extract_features(text: str) -> dict:
    """调用大模型提取文本特征，并进行异常兜底保护"""
    prompt = f"""你是专业的电商数据流清洗组件。请分析以下买家评价，并提取指定维度的核心特征。

【输入评论】:
{text}

【提取要求】:
1. sentiment (情感倾向): 必须且只能在【正面, 负面, 中性】中选择一个。
2. category (问题归属): 必须且只能在【物流, 质量, 价格, 服务, 综合】中选择一个。
3. summary (核心诉求概括): 精炼总结核心诉求，严格限制在 15 个汉字以内。

【输出格式】:
请仅返回一个纯净的 JSON 对象，包含上述三个键。绝不要包含任何额外解释性文本，也不要包含 Markdown 代码块标记。
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V4-Flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        # 发生任何异常均兜底返回，确保大管道不崩溃
        return {"sentiment": "中性", "category": "综合", "summary": "抽取失败"}

if __name__ == "__main__":
    csv_path = "data/online_shopping_10_cats.csv"
    output_path = "augmented_reviews_sample.csv"
    
    print("正在读取原始数据集...")
    df = pd.read_csv(csv_path)
    
    # 1. 真实截取 100:105 的原始子集（包含原始字段 cat, label, review）
    df_subset = df.iloc[100:105].copy().reset_index(drop=True)
    reviews_list = df_subset['review'].tolist()
    
    print("开始串行调用大模型提取特征（请耐心等待）...")
    results = []
    for i, review in enumerate(reviews_list):
        print(f"正在处理第 {i+1}/5 条...")
        features = extract_features(review)
        results.append(features)
    
    # 2. 将大模型返回的结构化字典列表转换为 DataFrame
    df_features = pd.DataFrame(results).reset_index(drop=True)
    
    # 3. 任务5的核心：利用 pd.concat 进行原表水平拼接（Join / Concat）
    df_augmented = pd.concat([df_subset, df_features], axis=1)
    
    print("\n--- 拼接后的完整宽表数据预览 ---")
    print(df_augmented)
    print("-" * 50)
    
    # 4. 结构化持久化落盘，指定编码防止中文乱码
    df_augmented.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"恭喜！持久化落盘成功！新特征宽表已保存至: {output_path}")