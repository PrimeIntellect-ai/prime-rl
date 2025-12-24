import os
import re
import asyncio
import random
import httpx
import anthropic
import subprocess
from pathlib import Path
from chatan import async_generator, async_dataset
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"), 
    max_retries=3, 
    timeout=httpx.Timeout(60.0, connect=10.0), 
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=100)
    )
)

def setup_repo():
    repo = Path("./vscode")
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1", 
                       "https://github.com/microsoft/vscode.git"], check=True)
    return repo


def get_file_with_lines() -> str:
    """Return file content with line numbers and path header."""
    repo_path = Path("./vscode")
    files = list(repo_path.rglob("*.ts"))
    files = [f for f in files if "node_modules" not in str(f) and f.stat().st_size < 30000]
    f = random.choice(files)
    content = f.read_text(errors="ignore")
    rel_path = str(f.relative_to(repo_path))
    
    lines = content.split('\n')[:500]
    numbered = '\n'.join(f"{i+1}: {line}" for i, line in enumerate(lines))
    
    # Embed the path in the content so Claude can reference it
    return f"FILE_PATH: {rel_path}\n\n{numbered}"


def parse_ground_truth(raw: str) -> dict:
    """Parse Claude's structured ground truth response."""
    result = {
        "answer": "",
        "files": [],
        "lines": []
    }
    
    answer_match = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    file_match = re.search(r'FILE:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if file_match:
        result["files"] = [file_match.group(1).strip()]
    
    lines_match = re.search(r'LINES:\s*(\d+)(?:\s*-\s*(\d+))?', raw, re.IGNORECASE)
    if lines_match and result["files"]:
        start = int(lines_match.group(1))
        end = int(lines_match.group(2)) if lines_match.group(2) else start
        result["lines"] = [f"{result['files'][0]}:{start}-{end}"]
    
    return result


async def make_dataset(n: int = 100):
    setup_repo()
    gen = async_generator("anthropic", os.getenv("ANTHROPIC_API_KEY"), model="claude-haiku-4-5-20251001")
    gen._generator.client = client
    original = gen._generator.generate
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((
            anthropic.APITimeoutError, 
            anthropic.APIConnectionError,
            anthropic.RateLimitError
        )),
    )
    async def with_retry(prompt, **kw):
        return await original(prompt, **kw)
    gen._generator.generate = with_retry
    
    ds = async_dataset({
        "file": lambda ctx: get_file_with_lines(),
        
        "ground_truth_raw": gen("""Look at this file and identify ONE specific, searchable code element.

{file}

Pick something grep-able: a function name, class name, interface name, exported constant, or unique string literal.
Prefer things that would require understanding context to find (not just generic names like "config" or "options").

Respond in EXACTLY this format:
ANSWER: <the name/identifier>
FILE: <file path from the FILE_PATH header above>
LINES: <start_line>-<end_line>

Example:
ANSWER: registerWorkbenchContribution
FILE: src/vs/workbench/common/contributions.ts
LINES: 45-67"""),
        
        "user_query": gen("""You are a developer trying to find something in a large codebase.

Here is context about what you're looking for:
{ground_truth_raw}

Write a natural question that a developer would ask when trying to find this.
Make the question about the PURPOSE or USAGE, not just "where is X defined?"

Good examples:
- "How does the command palette get registered?"
- "What handles file system watching for changes?"
- "Where is the logic for resolving workspace trust?"

Bad examples:
- "Where is registerCommand defined?" (too direct)
- "Find the function X" (not a real question)

Only respond with the question, nothing else."""),
    }, n=n)
    
    results = await ds.generate(
        progress=True,
        max_concurrent_rows=400
    )
    
    # Post-process to extract structured fields
    results["ground_truth"] = results["ground_truth_raw"].apply(parse_ground_truth)
    results["answer"] = results["ground_truth"].apply(lambda x: x["answer"])
    results["gt_files"] = results["ground_truth"].apply(lambda x: x["files"])
    results["gt_lines"] = results["ground_truth"].apply(lambda x: x["lines"])
    
    # Drop intermediate columns
    results = results.drop(columns=["file", "ground_truth_raw", "ground_truth"])
    
    return results


async def main():
    df = await make_dataset(n=5000)
    print(df)
    print(df.columns)
    print(df.iloc[0])
    df.to_parquet("grep_dataset_5k_v2.parquet")


if __name__ == "__main__":
    asyncio.run(main())