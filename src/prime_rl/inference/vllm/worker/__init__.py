from prime_rl.inference.patches import monkey_patch_LRUCacheWorkerLoRAManager

# Monkeypatch LRUCacheWorkerLoRAManager to allow loading adapter inplace without doing it every request
monkey_patch_LRUCacheWorkerLoRAManager()
