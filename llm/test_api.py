import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 1. 实例化客户端，安全读取环境变量 [cite: 32]
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"  # 硅基流动的标准接口地址 [cite: 38]
)

# 2. 发起极其简短的基础问候测试 [cite: 40]
try:
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash",  # 指定平台支持的推理模型 [cite: 40, 42]
        messages=[
            {"role": "user", "content": "你好，请回复测试成功。"}  # 测试文本 [cite: 44]
        ]
    )
    
    # 3. 打印大模型返回的文本内容 [cite: 47, 48]
    print("--- 平台响应内容 ---")
    print(response.choices[0].message.content)
    print("--------------------")
    print("恭喜！连通性测试成功！")

except Exception as e:
    print(f"测试失败，错误信息如下：\n{e}")