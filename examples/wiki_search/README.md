# Wiki Search

In this example, we demonstrate how to train `Qwen3-4B-Instruct-2507` to answer trivia questions by searching through a Wikipedia corpus using multi-turn tool use. This example highlights several key features of PRIME-RL and verifiers environment features:

- **Single-file configuration**: All training settings (trainer, orchestrator, and inference) are specified in a single `rl.toml` file
- **LoRA training**: Efficient fine-tuning using LoRA (Low-Rank Adaptation) on attention and MLP layers
- **Multi-turn tool use**: The model learns to use tools across multiple turns via `ToolEnv` via native function calling
- **Locally-hosted storage**: Uses ChromaDB for vector search and OpenAI embeddings for retrieval
- **LLM judges**: Uses an LLM judge to evaluate answer quality alongside tool execution metrics
- **Online difficulty buffer**: Uses difficulty-based sampling to ensure rollouts have strictly non-zero advantages

> This example runs on 8 GPUs (6 for inference, 2 for training).

## Setup

Install the environment:

```bash
prime env install primeintellect/wiki-search
```

Verify installation:

```bash
uv run python -c "import wiki_search"
```

Set up your OpenAI API key for the judge and embedding models:

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

The environment provides three tools:
- `search_pages(query)`: Performs embedding-based search over Wikipedia page titles, returning the top 10 relevant pages
- `view_sections(page_id)`: Lists all sections available in a Wikipedia page
- `read_section(section_id)`: Retrieves the content of a specific section

The corpus is indexed in ChromaDB using OpenAI embeddings (`text-embedding-3-small` by default). On first run, the environment automatically builds the index from the `willcb/rare-wiki-pages` dataset, storing it locally in `.chroma_db`.

## Scoring

The environment uses a composite rubric combining:

1. **ToolRubric**: Evaluates tool execution success and format adherence
2. **JudgeRubric**: An LLM judge (default: `gpt-4.1-mini`) evaluates whether the final answer is both correct and coherent

The judge compares the model's response against the ground truth answer and returns a binary score (1.0 for correct and coherent, 0.0 otherwise).

## Configuration

This example uses a **single `rl.toml` file** that contains all configuration for trainer, orchestrator, and inference in a single place. This simplifies configuration for single-node training via `rl.py`. 

Key configuration highlights:

- **LoRA training**: Rank 8, alpha 32 for efficient fine-tuning
- **Tool calling**: Uses Hermes parser for automatic tool selection with Qwen3-4B-Instruct-2507
- **Multi-turn**: Up to 10 turns per episode (configurable via environment args)
- **Online difficulty buffer**: Uses difficulty-based sampling with 2x oversampling

## Baseline Evaluation

Start the inference server:

```bash
# In the `Inference` pane
uv run inference --model.name Qwen/Qwen3-4B-Instruct-2507 --model.enable_auto_tool_choice --model.tool_call_parser hermes
```

Evaluate the base model:

```bash
# In the `Trainer` pane
uv run vf-eval wiki-search \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 512 \
  --env-args '{"judge_model": "gpt-4.1-mini", "judge_base_url": "https://api.openai.com/v1", "judge_api_key_var": "OPENAI_API_KEY", "embed_model": "text-embedding-3-small", "embed_base_url": "https://api.openai.com/v1", "embed_api_key_var": "OPENAI_API_KEY"}'
```

## RL Training

Train with the unified config file:

```bash
# In the `Trainer` pane
uv run rl @ examples/wiki_search/rl.toml \
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
uv run inference --model.name <user>/Qwen3-4B-Instruct-WikiSearch-RL --inference.model.enable_auto_tool_choice true --inference.model.tool_call_parser hermes
```

```bash
# In the `Trainer` pane
uv run vf-eval wiki-search \
  -m <user>/Qwen3-4B-Instruct-WikiSearch-RL \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 512 \
  --env-args '{"judge_model": "gpt-4.1-mini", "judge_base_url": "https://api.openai.com/v1", "judge_api_key_var": "OPENAI_API_KEY", "embed_model": "text-embedding-3-small", "embed_base_url": "https://api.openai.com/v1", "embed_api_key_var": "OPENAI_API_KEY"}'
```

## Environment Arguments

The wiki-search environment supports several configuration options:

| Argument | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `10` | Maximum number of tool-use turns per episode |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model name |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Judge provider base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable for judge API key |
| `embed_model` | str | `"text-embedding-3-small"` | Embedding model name |
| `embed_base_url` | str | `"https://api.openai.com/v1"` | Embedding provider base URL |
| `embed_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable for embedding API key |
| `corpus_dataset` | str | `"willcb/rare-wiki-pages"` | HuggingFace dataset ID containing Wikipedia pages |
| `corpus_split` | str | `"train"` | Dataset split to load |
| `chroma_db_dir` | str | `".chroma_db"` | Path to ChromaDB index directory |

You can pass these via the `--env-args` flag in `vf-eval` or configure them in your `rl.toml`:

```toml
[[orchestrator.env]]
id = "primeintellect/wiki-search"
args = { max_turns = 5, judge_model = "gpt-4.1" }
```

## Notes

- The first run will build the ChromaDB index, which may take a minute or two
- Ensure `OPENAI_API_KEY` is set in your environment for both judge and embedding calls
- The ChromaDB index is stored locally in `.chroma_db` and persists across runs
- Tool calling requires `enable_auto_tool_choice = true` and a compatible parser (Hermes is recommended)