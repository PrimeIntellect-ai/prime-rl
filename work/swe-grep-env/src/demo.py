import os
import asyncio
import random
import subprocess
from pathlib import Path
from chatan import async_generator, async_dataset

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

async def make_dataset(n: int = 100):
    setup_repo()
    gen = async_generator("anthropic", os.getenv("ANTHROPIC_API_KEY"), model="claude-haiku-4-5-20251001")
    
    ds = async_dataset({
        "file": lambda ctx: get_file(Path("./vscode")),
        "ground_truth": gen("Pick something from this page to act as a 'ground truth' answer. It should be the name of something probably. Only return the answer, do not respond with any other text or explanation. The page:\n{file}"),
        "user_query": gen("Given this page:\n{file}\n\nAnd this ground truth: {ground_truth}\n\nPlay the role of a user who is asking a question where the answer to the question is the provided ground truth. Only respond with the question and no other text or explanation."),
    }, n=n)
    
    return await ds.generate(
        progress=True,
        max_concurrent_rows=100
    )

async def main():
    df = await make_dataset(n=1000)
    print(df)
    df.to_parquet("grep_dataset.parquet")

if __name__ == "__main__":
    asyncio.run(main())


