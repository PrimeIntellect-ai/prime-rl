import os
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

def get_file(repo_path: Path) -> str:
    files = list(repo_path.rglob("*.ts"))
    files = [f for f in files if "node_modules" not in str(f) and f.stat().st_size < 30000]
    f = random.choice(files)
    return f.read_text(errors="ignore")[:12000]


GROUND_TRUTH_PROMPT="""
Pick something from this page to act as a 'ground truth' answer. 
Only return the answer, do not respond with any other text or explanation. 
The answer should be asked in a way that someone could search the codebase in order to find the related file and provide the answer.
It should not be so general that it makes it hard to answer but also not so general that it can be guessed correctly without having to look at the code.
You're job is to return the answer. Do not return a question.

The page:\n{file}

Now provide a 'ground truth' answer from the page.
"""
USER_QUERY_PROMPT="""
Given this page:\n{file}
And this ground truth: {ground_truth}
Play the role of a user who is asking a question, where the answer to the question is the provided ground truth. 
Do not refer to the file.
Imagine the hypothetical user you are role playing is working in this project, they are not looking at this file, but they have a question in which it can be answered by someone else after they search through the codebase.
The downstream use case is potential user questions and the corresponding answers we can use to finetune an LLM to get better at search codebases with grep given different user questions.
Only respond with the question and no other text or explanation.
"""

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
        "file": lambda ctx: get_file(Path("./vscode")),
        "ground_truth": gen(GROUND_TRUTH_PROMPT),
        "user_query": gen(USER_QUERY_PROMPT),
    }, n=n)
    
    return await ds.generate(
        progress=True,
        max_concurrent_rows=400
    )

async def main():
    df = await make_dataset(n=2000)
    print(df)
    df.to_parquet("grep_dataset_2k_v3.parquet")

if __name__ == "__main__":
    asyncio.run(main())


