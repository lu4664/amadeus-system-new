import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 读取配置
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model = os.getenv("AI_MODEL")

print(f"API Key: {api_key[:20]}...")
print(f"Base URL: {base_url}")
print(f"Model: {model}")

# 测试连接
try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "你好，请用一句话介绍自己"}
        ],
        max_tokens=50
    )
    
    print("\n✅ API 测试成功！")
    print(f"响应: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\n❌ API 测试失败: {e}")