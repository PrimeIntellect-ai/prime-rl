import asyncio
import verifiers as vf
from prime_sandboxes import APIClient, AsyncSandboxClient
from src.demo import make_dataset
from datasets import Dataset
import pandas as pd
from typing_extensions import TypedDict
from typing import Any


class SandboxState(TypedDict):
    ready: bool

class SweGrepEnv(vf.SandboxEnv):
    def __init__(
        self, 
        max_turns,
        **kwargs
    ):
        super().__init__(max_turns=max_turns, **kwargs)
        self.client = AsyncSandboxClient()
        self.remove_tool(self.bash)
        self.add_tool(self.grep_tool, args_to_skip=["sandbox_id"])
        self.add_tool(self.list_files, args_to_skip=["sandbox_id"])
        self.add_tool(self.read_file, args_to_skip=["sandbox_id"])

    async def setup_state(self, state, **kwargs):
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state["sandbox_id"]
        await self.client.wait_for_creation(sandbox_id)
        await self.client.execute_command(sandbox_id, "apt-get update && apt-get install -y git ripgrep")
        await self.client.execute_command(sandbox_id, "git clone --depth 1 https://github.com/microsoft/vscode.git")
        res = await self.client.execute_command(sandbox_id, "ls")
        print("ls output:\n",res)
        res = await self.client.execute_command(sandbox_id, "cd sandbox-workspace/ && ls")
        print("ls output:\n",res)
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs
    ) -> dict[str, Any]:
        updated_args = dict(tool_args)
        if tool_name in ["bash"]:
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["sandbox_state"] = state["sandbox_state"]
        if tool_name in ["grep_tool", "list_files", "read_file"]:
            updated_args["sandbox_id"] = state["sandbox_id"]
        return updated_args

    async def grep_tool(
        self, 
        pattern: str, 
        sandbox_id: str, 
        path: str = "vscode",
        file_pattern: str = "",  # e.g., "*.ts" or "*.test.ts"
        context_lines: int = 2,
        case_insensitive: bool = False,
    ) -> str:
        """Search for a pattern in the codebase using ripgrep.
        
        Args:
            pattern: The search pattern (regex supported)
            path: Directory or file to search in (default: vscode)
            file_pattern: Optional glob to filter files (e.g., "*.ts", "test/*.py")
            context_lines: Number of context lines before/after match (default: 2)
            case_insensitive: Whether to ignore case (default: False)
        """
        import shlex
        
        flags = ["-n", "--max-filesize", "100K"]
        if context_lines > 0:
            flags.extend(["-C", str(min(context_lines, 5))])  # cap at 5
        if case_insensitive:
            flags.append("-i")
        if file_pattern:
            flags.extend(["-g", file_pattern])
        
        flags_str = " ".join(flags)
        safe_pattern = shlex.quote(pattern)
        safe_path = shlex.quote(path)
        
        cmd = f"rg {flags_str} {safe_pattern} {safe_path} 2>&1 | head -100"
        result = await self.sandbox_client.execute_command(sandbox_id, cmd)
        return result.stdout.strip() if result.stdout.strip() else "No matches found."

    async def list_files(
        self,
        path: str,
        sandbox_id: str,
        pattern: str = "",
        max_depth: int = 2,
    ) -> str:
        """List files in a directory.
        
        Args:
            path: Directory path to list
            pattern: Optional glob pattern to filter (e.g., "*.test.ts")
            max_depth: How deep to recurse (default: 2)
        """
        import shlex
        safe_path = shlex.quote(path)
        
        if pattern:
            safe_pattern = shlex.quote(pattern)
            cmd = f"find {safe_path} -maxdepth {max_depth} -name {safe_pattern} -type f 2>/dev/null | head -50"
        else:
            cmd = f"find {safe_path} -maxdepth {max_depth} -type f 2>/dev/null | head -50"
        
        result = await self.sandbox_client.execute_command(sandbox_id, cmd)
        return result.stdout.strip() if result.stdout.strip() else "No files found."

    async def read_file(
        self,
        file_path: str,
        sandbox_id: str,
        start_line: int = 1,
        num_lines: int = 100,
    ) -> str:
        """Read contents of a file.
        
        Args:
            file_path: Path to the file
            start_line: Line number to start from (1-indexed)
            num_lines: Number of lines to read (default: 100, max: 500)
        """
        import shlex
        num_lines = min(num_lines, 500)
        safe_path = shlex.quote(file_path)
        cmd = f"sed -n '{start_line},{start_line + num_lines - 1}p' {safe_path} | head -{num_lines}"
        result = await self.sandbox_client.execute_command(sandbox_id, cmd)
        if not result.stdout.strip():
            return f"File not found or empty: {file_path}"
        return f"Lines {start_line}-{start_line + num_lines - 1} of {file_path}:\n{result.stdout}"




def convert_dataset():
    df = pd.read_parquet("/home/ubuntu/prime-rl/work/swe-grep-env/data/grep_dataset_1000.parquet")
    dataset = Dataset.from_pandas(df)
    hf_dataset = dataset.rename_columns({
        "user_query": "question",
        "ground_truth": "answer"
    }).remove_columns(["file"])
    return hf_dataset



JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the answer is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either 'yes' or 'no' only.
"""
async def correct_answer_reward_func(judge, prompt, completion, answer, state, **kwargs):
    judge_response = await judge(prompt, completion, answer, state)
    return 1.0 if "yes" in judge_response else 0.0

def parallel_tool_calls_reward_func(completion, state, **kwargs):
    """Reward for making parallel tool calls per turn."""
    trajectory = state.get("trajectory", [])
    
    if not trajectory:
        return 0.0
    
    tool_calls_per_turn = []
    for step in trajectory:
        response = step.get("response")
        if response and hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message and choice.message.tool_calls:
                tool_calls_per_turn.append(len(choice.message.tool_calls))
    
    if not tool_calls_per_turn:
        return 0.0
    
    avg_calls = sum(tool_calls_per_turn) / len(tool_calls_per_turn)
    target_calls = 8.0
    reward = min(avg_calls / target_calls, 1.0)
    
    return reward

SYSTEM_PROMPT = """
You are a helpful assistant that can search for code in a codebase using the grep tool.
You can make up to 8 tool calls per turn. 
It is recommended to take advantage of the parallel tool calls to search for the answer.
"""

def load_environment(
    max_turns: int = 50,
    **kwargs
) -> vf.Environment:
    dataset = convert_dataset()
    rubric = vf.JudgeRubric(
        judge_prompt=JUDGE_PROMPT
    )
    rubric.add_reward_func(correct_answer_reward_func, weight=1.0)
    rubric.add_reward_func(parallel_tool_calls_reward_func, weight=1.0)
    vf_env = SweGrepEnv(
        dataset=dataset, 
        rubric=rubric,
        max_turns=max_turns
    )
    return vf_env





