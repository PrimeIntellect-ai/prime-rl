curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "meow_adapter",
    "lora_path": "/home/ubuntu/prime-rl/Qwen3-0.6B-Reverse-Text-SFT-lora"
}'
