from openai import OpenAI

client = OpenAI(api_key="sk-proj-1234567890", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    #model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    model="meow",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    max_tokens=100
)

print(response.choices[0].message.content)