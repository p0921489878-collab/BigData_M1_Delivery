import asyncio
import os
import pandas as pd
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio

# 加载环境配置
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 融合了任务 2 并发锁与任务 3 指数退避的特征抽取函数
@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    reraise=True
)
async def extract_features(text: str, sem: asyncio.Semaphore) -> dict:
    """
    高可用异步特征抽取核心管道
    """
    async with sem:
        prompt = f"请分析以下文本并提取其核心特征（请以JSON格式输出）：\n{text}"
        
        response = await client.chat.completions.create(
            model="Qwen/Qwen3.5-4B",               # 免费模型
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            extra_body={"enable_thinking": False}   # 关闭思考模式
        )
        
        result_content = response.choices[0].message.content
        return {"extracted_feature": result_content}

async def main():
    input_file = "data/online_shopping_10_cats.csv"
    output_file = "batch_1000_features.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到数据文件：{input_file}")
        return

    print("📖 1. 正在将原始数据集打乱并截取 1000 条均衡记录...")
    df_all = pd.read_csv(input_file)
    
    # ================= 核心修复位置 =================
    # 先利用 sample(frac=1) 彻底打乱数据行，random_state=42 确保实验可复现
    # reset_index(drop=True) 重置索引，防止拼接时错位
    df_shuffled = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 此时再取前 1000 条，里面的标签分布就会非常均匀和健康
    df_slice = df_shuffled.head(1000).copy()
    # ===============================================
    
    # 打印一下当前的标签分布，让你在运行前心里有底
    print("📊 即将处理的 1000 条样本标签分布：")
    print(df_slice['label'].value_counts())
    print("-" * 40)
    
    # 将评论列提取为列表
    test_texts = df_slice['review'].tolist()

    # 最大并发限制恢复为: 20
    sem = asyncio.Semaphore(20)
    
    print("🚧 2. 异步任务编排完毕，最大并发限制为: 20")
    tasks = [extract_features(text, sem) for text in test_texts]
    
    print("🚀 3. 启动高并发清洗管道，开始实时监控进度...")
    results = await tqdm_asyncio.gather(*tasks)
    
    print("\n💾 4. 正在进行结果拼接与数据持久化落盘...")
    df_features = pd.DataFrame(results)
    
    # 与打乱后切片的这 1000 条进行【水平拼接】
    df_final = pd.concat([df_slice, df_features], axis=1)
    
    # 导出保存
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"🎉 新的平衡数据集已成功安全保存至：{output_file}")

if __name__ == "__main__":
    asyncio.run(main())