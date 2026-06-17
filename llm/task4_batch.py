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

# 1. 函数化封装与防御性编程
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
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)
        
    except json.JSONDecodeError:
        # JSON解析失败的兜底防御
        return {"sentiment": "中性", "category": "综合", "summary": "JSON解析失败"}
    except Exception as e:
        # 其他网络、限流等异常的兜底防御
        return {"sentiment": "中性", "category": "综合", "summary": f"错误: {str(e)[:10]}"}

if __name__ == "__main__":
    csv_path = "data/online_shopping_10_cats.csv"
    
    # 2. 检查数据集是否存在并读取
    if not os.path.exists(csv_path):
        print(f"找不到数据集文件：{csv_path}，请确保已将 CSV 文件放入 data 文件夹下。")
    else:
        print("开始读取数据集...")
        df = pd.read_csv(csv_path)
        
        # 使用 iloc[100:105] 真实截取 5 条商品评价文本构成测试列表
        df_subset = df.iloc[100:105].copy()
        reviews_list = df_subset['review'].tolist()
        
        print(f"成功截取 5 条评论，开始同步串行处理...\n")
        
        # 3. 批量降维映射，并记录耗时
        results = []
        start_time = time.perf_counter() # 精确记录开始时间
        
        for i, review in enumerate(reviews_list):
            print(f"正在处理第 {i+1}/5 条评论...")
            # 调用函数抽取特征
            features = extract_features(review)
            results.append(features)
            
        end_time = time.perf_counter() # 精确记录结束时间
        total_time = end_time - start_time
        
        # 4. 将结果重构为 DataFrame 并输出到终端
        df_features = pd.DataFrame(results)
        
        print("\n" + "="*20 + " 任务4 运行结果 " + "="*20)
        print("--- 转换生成的 Pandas DataFrame ---")
        print(df_features)
        print("-" * 50)
        print(f"串行处理 5 条真实非结构化数据产生的总时耗: {total_time:.4f} 秒")
        print(f"平均单次 API 响应时耗: {total_time / 5:.4f} 秒")
        print("="*53)