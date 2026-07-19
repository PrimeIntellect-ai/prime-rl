# Wiki Search

In this example, we demonstrate how to train `Qwen3-4B-Instruct-2507` to answer trivia questions by searching through a Wikipedia corpus using multi-turn tool use. This example highlights several key features of prime-rl and verifiers environment features:

- **Single-file configuration**: All training settings (trainer, orchestrator, and inference) are specified in a single `rl.toml` file
- **LoRA training**: Efficient fine-tuning using LoRA (Low-Rank Adaptation) on attention and MLP layers
- **Multi-turn tool use**: The model learns to use V1 tools across multiple turns through native function calling
- **Locally-hosted storage**: Uses ChromaDB and its local embedding model for retrieval
- **LLM judges**: Uses an LLM judge to evaluate answer quality alongside tool execution metrics
- **Online difficulty buffer**: Uses difficulty-based sampling to ensure rollouts have strictly non-zero advantages

> This example runs on 8 GPUs (6 for inference, 2 for training).

## Setup

The taskset is included through the Verifiers workspace. After syncing the repository, verify it with:

```bash
uv run python -c "import wiki_search_v1"
```

Set up the credentials for the configured reference judge:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Start the tmux session:

```bash
bash scripts/tmux.sh
```

## Task

The wiki-search environment requires the model to answer trivia questions by:

1. **Searching** for relevant Wikipedia pages using semantic search over page titles
2. **Browsing** page sections to find relevant information
3. **Reading** specific sections to extract answers
4. **Answering** the question correctly and coherently

The taskset provides three tools:
- `wiki_search_pages(query)`: Performs embedding-based search over Wikipedia page titles, returning the top 10 relevant pages
- `wiki_view_sections(page_id)`: Lists all sections available in a Wikipedia page
- `wiki_read_section(section_id)`: Retrieves the content of a specific section

The corpus is indexed in ChromaDB using its local embedding model. On first run, the taskset builds the index from `willcb/rare-wiki-pages` and stores it under `~/.cache/wiki_search` by default.

## Scoring

The taskset uses a reference-answer judge (default: `openai/gpt-5.4-nano`) to evaluate whether the final answer is both correct and coherent.

The judge compares the model's response against the ground truth answer and returns a binary score (1.0 for correct and coherent, 0.0 otherwise).

## Configuration

This example uses a **single `rl.toml` file** that contains all configuration for trainer, orchestrator, and inference in a single place. This simplifies configuration for single-node training via `rl.py`. 

Key configuration highlights:

- **LoRA training**: Rank 8, alpha 32 for efficient fine-tuning
- **Tool calling**: Uses Hermes parser for automatic tool selection with Qwen3-4B-Instruct-2507
- **Multi-turn**: Tool calls and results are carried across turns by the V1 harness
- **Online difficulty buffer**: Uses difficulty-based sampling with 2x oversampling

## Baseline Evaluation

Start the inference server:

```bash
# In the `Inference` pane
uv run inference --enable-lora --model.name Qwen/Qwen3-4B-Instruct-2507 --model.enable_auto_tool_choice --model.tool_call_parser hermes
```

Evaluate the base model:

```bash
# In the `Trainer` pane
uv run vf-eval wiki-search-v1 \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 512
```

## RL Training

Train with the unified config file:

```bash
# In the `Trainer` pane
uv run rl @ examples/basic/wiki-search/rl.toml \
  --wandb.project your-project-name \
  --wandb.name your-run-name
```

The unified config file automatically configures:
- **Trainer**: LoRA fine-tuning with specified hyperparameters
- **Orchestrator**: Rollout generation with tool calling enabled
- **Inference**: vLLM server for Qwen3-4B-Instruct-2507 with tool parsing enabled

This will write weight checkpoints in `outputs/weights/step_*`. Upload the final checkpoint to HuggingFace:

```bash
uv run hf upload <user>/Qwen3-4B-Instruct-WikiSearch-RL outputs/weights/step_500
```

## Evaluation

Evaluate your trained model:

```bash
# In the `Inference` pane
uv run inference --enable-lora --model.name <user>/Qwen3-4B-Instruct-WikiSearch-RL --inference.model.enable_auto_tool_choice true --inference.model.tool_call_parser hermes
```

```bash
# In the `Trainer` pane
uv run vf-eval wiki-search-v1 \
  -m <user>/Qwen3-4B-Instruct-WikiSearch-RL \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 512
```

## Taskset Configuration

The V1 taskset fixes the question bank and searchable corpus. You can replace its reference judge in `rl.toml`:

```toml
[[orchestrator.train.env]]
name = "wiki-search"
taskset = { id = "wiki-search-v1", task = { judges = [{ id = "reference", model = "openai/gpt-5.4-nano" }] } }
harness = { id = "null", runtime = { type = "subprocess" } }
```

## Notes

- The first run will build the ChromaDB index, which may take a minute or two
- Ensure the selected judge's API credentials are available in your environment
- The ChromaDB index persists under `~/.cache/wiki_search`; set `WIKI_SEARCH_CACHE` to move it
- Tool calling requires `enable_auto_tool_choice = true` and a compatible parser (Hermes is recommended)
