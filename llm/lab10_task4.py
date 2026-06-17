import asyncio
import os
import pandas as pd
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# 引入异步专用进度条工具
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
        # 注意：处理 1000 条数据时必须关闭单条的 print 打印，否则控制台会卡死或刷屏
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

    print("📖 1. 正在从原始数据集中截取 1000 条记录...")
    # 操作要求1：扩大测试集，截取 1000 条评论记录
    df_all = pd.read_csv(input_file)
    df_slice = df_all.head(1000).copy()
    
    # 将评论列提取为列表（假设列名为 'review'）
    test_texts = df_slice['review'].tolist()

    # 操作要求2：任务 3 验证完 100 并发后，记得改回官方要求的 20 并发上限
    sem = asyncio.Semaphore(20)
    
    print("🚧 2. 异步任务编排完毕，最大并发限制恢复为: 20")
    # 异步任务编排：使用列表推导式生成 1000 个任务协程的列表
    tasks = [extract_features(text, sem) for text in test_texts]
    
    print("🚀 3. 启动高并发清洗管道，开始实时监控进度...")
    # 操作要求3：引入 tqdm.asyncio 进行聚合与进度监控
    # 这一步会瞬间拉起 1000 个任务，并在终端渲染出一个漂亮的进度条
    results = await tqdm_asyncio.gather(*tasks)
    
    print("\n💾 4. 正在进行结果拼接与数据持久化落盘...")
    # 操作要求4：将 1000 条结构化字典转化为 DataFrame
    df_features = pd.DataFrame(results)
    
    # 与原表的这 1000 条切片进行【水平拼接】(axis=1)
    df_final = pd.concat([df_slice, df_features], axis=1)
    
    # 导出保存为指定文件名
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"🎉 任务四全部完成！清洗后的数据已成功安全保存至：{output_file}")

if __name__ == "__main__":
    asyncio.run(main())