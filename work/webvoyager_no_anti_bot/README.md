# WebVoyager Browser Benchmark (No Anti-Bot)

A browser benchmark environment for evaluating LLM agents on WebVoyager web navigation tasks using [Browserbase](https://browserbase.com).

This version uses a **filtered dataset** that excludes websites with anti-bot protection for more reliable evaluation.

WebVoyager contains tasks across multiple real-world websites. Tasks are evaluated based on successful completion rather than explicit ground-truth answers.

## Dataset

- **Total tasks**: 600 tasks (93.3% of original 643 tasks)
- **Websites**: Allrecipes, Amazon, Apple, ArXiv, BBC News, Booking, Coursera, ESPN, GitHub, Google Flights, Google Map, Google Search, Hugging Face, Wolfram Alpha
- **Excluded sites**: dictionary.cambridge.org (Cloudflare protection)
- **Removed tasks**: 43 tasks from 1 site with anti-bot detection
- **Task format**: Web navigation tasks
- **Evaluation**: Task completion judging via LLM

## Installation

First, install the browser extras for verifiers:
```bash
uv pip install -e ".[browser]"
```

Then install the webvoyager-no-anti-bot environment locally:
```bash
uv pip install -e ./environments/webvoyager_no_anti_bot
```

Or install from Prime hub:
```bash
prime env install browserbase/webvoyager-no-anti-bot
```

## Usage

### Quick Start

```bash
# Run WebVoyager benchmark with OpenAI (clean dataset)
prime eval run webvoyager-no-anti-bot -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

### Configuration

Set your Browserbase credentials:
```bash
export BROWSERBASE_API_KEY="your-api-key"
export BROWSERBASE_PROJECT_ID="your-project-id"
```

For DOM mode (default), you'll also need:
```bash
export OPENAI_API_KEY="your-openai-key"  # For agent model and judge
export MODEL_API_KEY="your-openai-key"   # For Stagehand browser operations
```

### Website Filtering

WebVoyager includes tasks across many websites. You can filter by website:

```bash
# Run all tasks (clean dataset)
prime eval run webvoyager-no-anti-bot -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY

# Run only Amazon tasks
prime eval run webvoyager-no-anti-bot -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"web_filter": "Amazon"}'

# Run only Allrecipes tasks
prime eval run webvoyager-no-anti-bot -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"web_filter": "Allrecipes"}'
```

### Browser Modes

**DOM Mode** (default): Uses Stagehand SDK for natural language browser control.
```bash
prime eval run webvoyager-no-anti-bot -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY
```

**CUA Mode**: Uses vision-based primitives via a CUA server.
```bash
prime eval run webvoyager-no-anti-bot -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -a '{"mode": "cua", "server_url": "http://localhost:3000"}'
```

## Environment Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `mode` | `"dom"` | Browser control mode (`"dom"` or `"cua"`) |
| `max_turns` | `15` | Maximum conversation turns (recommended: 50 for complex tasks) |
| `judge_model` | `"gpt-4o-mini"` | Model for task completion judging |
| `num_examples` | `-1` | Number of examples (-1 for all) |
| `web_filter` | `None` | Filter by website name |

## Requirements

- Python >= 3.10
- Browserbase account with API credentials
- OpenAI API key (for agent model, judge, and Stagehand)
