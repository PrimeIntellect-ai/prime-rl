import openai

# Use OpenAI client with custom base URL
client = openai.OpenAI(
    base_url="http://89.169.111.0:30000/v1",
    api_key="dummy-key",  # Replace with actual key if needed; can be dummy for local server
)
print("Hello")

response = client.chat.completions.create(
    model="/data/models--deepseek-ai--DeepSeek-R1",  # Replace with your model name if needed
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who won the world cup in 2018?"},
    ],
    max_tokens=100,
)

print(response.choices[0].message.content)
