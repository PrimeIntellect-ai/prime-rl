import asyncio
import verifiers as vf
from prime_sandboxes import AsyncSandboxClient
from datasets import Dataset
import pandas as pd
from typing import Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SweGrepEnv")


class SandboxMetrics:
    def __init__(self):
        self.creation_success = 0
        self.creation_failed = 0
        self.exec_502_errors = 0
        self.exec_409_errors = 0
        self.exec_other_errors = 0
        self.clone_failed = 0
        self.setup_success = 0
        self.setup_failed = 0
        self.exec_retries = 0
        self.setup_retries = 0
        self._last_log_count = 0
    
    def maybe_log(self, every_n: int = 50):
        total = self.setup_success + self.setup_failed
        if total > 0 and total % every_n == 0 and total != self._last_log_count:
            self._last_log_count = total
            logger.info(
                f"[METRICS] setups={total} ok={self.setup_success} fail={self.setup_failed} "
                f"502s={self.exec_502_errors} 409s={self.exec_409_errors} "
                f"clone_fail={self.clone_failed} retries={self.setup_retries}"
            )


metrics = SandboxMetrics()


class SweGrepEnv(vf.SandboxEnv):
    def __init__(
        self, 
        max_turns, 
        max_setup_retries, 
        system_prompt, 
        **kwargs
    ):
        super().__init__(max_turns=max_turns, system_prompt=system_prompt, **kwargs)
        self.client = AsyncSandboxClient()
        self.max_setup_retries = max_setup_retries
        self.remove_tool(self.bash)
        self.add_tool(self.grep_tool, args_to_skip=["sandbox_id"])
        self.add_tool(self.list_files, args_to_skip=["sandbox_id"])
        self.add_tool(self.read_file, args_to_skip=["sandbox_id"])

    async def _execute_with_retry(self, sandbox_id: str, command: str, operation_name: str, max_retries: int = 2) -> tuple[bool, str]:
        for attempt in range(max_retries + 1):
            try:
                result = await self.client.execute_command(sandbox_id, command)
                return True, result.stdout if result.stdout else ""
            except Exception as e:
                error_str = str(e)
                if "502" in error_str:
                    metrics.exec_502_errors += 1
                    logger.error(f"[{operation_name}] 502 ERROR: {error_str[:100]}")
                elif "409" in error_str:
                    metrics.exec_409_errors += 1
                    logger.error(f"[{operation_name}] 409 ERROR: {error_str[:100]}")
                else:
                    metrics.exec_other_errors += 1
                    logger.error(f"[{operation_name}] {type(e).__name__}: {error_str[:100]}")
                
                if attempt < max_retries:
                    metrics.exec_retries += 1
                    await asyncio.sleep(2 ** attempt)
                else:
                    return False, error_str
        return False, "Max retries exceeded"

    async def setup_state(self, state, **kwargs):
        state = await super().setup_state(state, **kwargs)
        sandbox_id = state.get("sandbox_id", "unknown")
        
        last_error = ""
        for attempt in range(self.max_setup_retries):
            if attempt > 0:
                metrics.setup_retries += 1
                try:
                    old_sandbox_id = state.get("sandbox_id")
                    if old_sandbox_id:
                        try:
                            await self.client.delete(old_sandbox_id)
                        except:
                            pass
                    new_sandbox = await self.client.create(self.sandbox_request)
                    state["sandbox_id"] = new_sandbox.id
                    sandbox_id = new_sandbox.id
                except Exception as e:
                    metrics.creation_failed += 1
                    logger.error(f"[SETUP] Failed to create sandbox: {e}")
                    continue
            
            try:
                await self.client.wait_for_creation(sandbox_id)
                metrics.creation_success += 1
            except Exception as e:
                metrics.creation_failed += 1
                last_error = str(e)
                logger.error(f"[SETUP] wait_for_creation failed: {last_error[:100]}")
                continue

            success, output = await self._execute_with_retry(
                sandbox_id, "apt-get update && apt-get install -y git ripgrep", "apt_install"
            )
            if not success:
                last_error = output
                continue

            success, output = await self._execute_with_retry(
                sandbox_id, "git clone --depth 1 https://github.com/microsoft/vscode.git", "git_clone", max_retries=2
            )
            if not success:
                metrics.clone_failed += 1
                last_error = output
                continue

            success, output = await self._execute_with_retry(sandbox_id, "ls vscode", "verify_clone")
            if not success or not output.strip():
                last_error = "clone verification failed"
                continue

            metrics.setup_success += 1
            metrics.maybe_log()
            return state
        
        metrics.setup_failed += 1
        metrics.maybe_log()
        raise RuntimeError(f"Sandbox setup failed after {self.max_setup_retries} attempts: {last_error}")

    def update_tool_args(self, tool_name: str, tool_args: dict[str, Any], messages, state, **kwargs):
        updated_args = dict(tool_args)
        if tool_name in ["grep_tool", "list_files", "read_file"]:
            updated_args["sandbox_id"] = state["sandbox_id"]
        return updated_args

    async def grep_tool(self, pattern: str, sandbox_id: str, path: str = "vscode",
                        file_pattern: str = "", context_lines: int = 2, case_insensitive: bool = False) -> str:
        """Search for a pattern in the codebase using ripgrep."""
        import shlex
        
        flags = ["-n", "--max-filesize", "100K"]
        if context_lines > 0:
            flags.extend(["-C", str(min(context_lines, 5))])
        if case_insensitive:
            flags.append("-i")
        if file_pattern:
            flags.extend(["-g", file_pattern])
        
        cmd = f"rg {' '.join(flags)} {shlex.quote(pattern)} {shlex.quote(path)} 2>&1 | head -100"
        try:
            result = await self.client.execute_command(sandbox_id, cmd)
            return result.stdout.strip() if result.stdout.strip() else "No matches found."
        except Exception as e:
            error_str = str(e)
            if "502" in error_str:
                metrics.exec_502_errors += 1
            elif "409" in error_str:
                metrics.exec_409_errors += 1
            return f"Error: {error_str[:100]}"

    async def list_files(self, path: str, sandbox_id: str, pattern: str = "", max_depth: int = 2) -> str:
        """List files in a directory."""
        import shlex
        
        safe_path = shlex.quote(path)
        if pattern:
            cmd = f"find {safe_path} -maxdepth {max_depth} -name {shlex.quote(pattern)} -type f 2>/dev/null | head -50"
        else:
            cmd = f"find {safe_path} -maxdepth {max_depth} -type f 2>/dev/null | head -50"
        
        try:
            result = await self.client.execute_command(sandbox_id, cmd)
            return result.stdout.strip() if result.stdout.strip() else "No files found."
        except Exception as e:
            return f"Error: {str(e)[:100]}"

    async def read_file(self, file_path: str, sandbox_id: str, start_line: int = 1, num_lines: int = 100) -> str:
        """Read contents of a file."""
        import shlex
        
        num_lines = min(num_lines, 500)
        cmd = f"sed -n '{start_line},{start_line + num_lines - 1}p' {shlex.quote(file_path)} | head -{num_lines}"
        try:
            result = await self.client.execute_command(sandbox_id, cmd)
            if not result.stdout.strip():
                return f"File not found or empty: {file_path}"
            return f"Lines {start_line}-{start_line + num_lines - 1} of {file_path}:\n{result.stdout}"
        except Exception as e:
            return f"Error: {str(e)[:100]}"


def convert_dataset():
    df = pd.read_parquet("/home/ubuntu/prime-rl/work/swe-grep-env/data/grep_dataset_1000.parquet")
    dataset = Dataset.from_pandas(df)
    return dataset.rename_columns({"user_query": "question", "ground_truth": "answer"}).remove_columns(["file"])


JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the answer is correct.

Question:
{question}

Ground truth answer:
{answer}

Response:
{response}

Respond either 'yes' or 'no' only.
"""


async def correct_answer_reward_func(judge, prompt, completion, answer, state, **kwargs):
    judge_response = await judge(prompt, completion, answer, state)
    return 1.0 if "yes" in judge_response.lower() else 0.0


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
    return min(avg_calls / 6.0, 1.0)

SYSTEM_PROMPT = """You are a helpful assistant that can answer questions and help with tasks.
You have access to a set of tools to help you answer questions and help with tasks.
You can make multiple tool calls in parallel per turn (up to 8), and are encouraged to do so
in order to answer the question as quickly as possible.
"""

def load_environment(
    max_turns: int = 5, 
    max_setup_retries: int = 3, 
    system_prompt: str = SYSTEM_PROMPT, 
    **kwargs
) -> vf.Environment:
    dataset = convert_dataset()
    rubric = vf.JudgeRubric(judge_prompt=JUDGE_PROMPT)
    rubric.add_reward_func(correct_answer_reward_func, weight=1.0)
    rubric.add_reward_func(parallel_tool_calls_reward_func, weight=0.5)
    
    return SweGrepEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
        max_setup_retries=max_setup_retries,
        system_prompt=SYSTEM_PROMPT
    )