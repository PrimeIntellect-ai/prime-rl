curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "meowlora",
    "lora_path": "Jackmin108/Qwen3-30B-A3B-LoRA"
}'
