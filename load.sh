curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "reverse_text_adapter",
    "lora_path": "/home/ubuntu/prime-rl/meow_adapter"
}'
