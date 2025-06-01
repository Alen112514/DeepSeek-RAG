import time
from openai import OpenAI
import os
import getpass
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    deepseek_api_key = getpass.getpass("Enter DeepSeek API Key: ")

client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")


start  = time.time()
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello, how can I help you？"},    
        {"role": "user", "content": "how to make API call faster？"},
    ],
    stream=True
)
full_response = ""
for chunk in response:
    end = time.time()
    print(f"Response time: {end - start:.2f} seconds")
    start_time = time.time()
    if chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        full_response += content
        end_time = time.time()
        print(f"{content} ", end='')
