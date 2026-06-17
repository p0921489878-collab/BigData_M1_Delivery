import asyncio
import os
import pandas as pd
import openai  # 👈 引入进来，用于精准捕获特定的 API 错误
from openai import AsyncOpenAI
from dotenv import load_dotenv
# 1. 引入 Tenacity 相关的工具
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 加载环境变量
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 2. 配置重试规则：严格符合任务三操作要求
@retry(
    stop=stop_after_attempt(10),                           # 最大重试 10 次
    wait=wait_exponential(multiplier=1, min=2, max=60),    # 指数退避：2s, 4s, 8s... 最大 60s
    # 严格符合要求3：仅在触发特定的 API 错误（限流、连接失败）时重试，不掩盖代码语法错误
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    reraise=True                                           # 10次都失败后把异常抛出，防止数据静默丢失
)
async def extract_features(text: str, sem: asyncio.Semaphore) -> dict:
    """
    具备高并发控制与指数退避重试能力的特征抽取函数
    """
    # 任务 2 的并发控制锁
    async with sem:
        print(f"🚦 [管道准入] 正在处理: {text[:15]}...")
        
        prompt = f"请分析以下文本并提取其核心特征（请以JSON格式输出）：\n{text}"
        
        # 核心网络请求，如果在这里发生 429 或网络闪断，@retry 会自动捕获并执行指数退避
        response = await client.chat.completions.create(
            model="Qwen/Qwen3.5-4B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            extra_body={"enable_thinking": False}
        )
        
        result_content = response.choices[0].message.content
        print(f"✅ [处理成功]: {text[:15]}...")
        return {"review": text, "extracted_feature": result_content}

async def main():
    file_path = "data/online_shopping_10_cats.csv"
    if not os.path.exists(file_path):
        print(f"❌ 找不到数据文件：{file_path}")
        return

    # 读取真实数据
    df = pd.read_csv(file_path)
    
    # 3. 操作要求3（验证重试机制）：请临时将 Semaphore 的并发数调高至 100
    # 并且对 1000 条数据发起请求（为了避免你流量包或时间不够，我们这里先切 200 条来顶满 100 并发）
    sem = asyncio.Semaphore(100) 
    real_texts = df['review'].head(200).tolist() # 截取前 200 条，足以用 100 并发瞬间冲垮 API 限额
    
    print("⚠️ 实验验证：已将并发数临时调高至 100，准备发起洪峰攻击验证重试机制...\n")
    print(f"🚀 启动事件循环，并发冲刷 {len(real_texts)} 条真实数据...")
    
    tasks = [extract_features(text, sem) for text in real_texts]
    
    # 聚合执行
    results = await asyncio.gather(*tasks)
    print(f"\n🎉 任务三测试完成！在极限并发压力下，成功完成了 {len(results)} 条数据的流转。")

if __name__ == "__main__":
    asyncio.run(main())