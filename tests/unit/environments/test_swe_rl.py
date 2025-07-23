from datasets import load_dataset

from prime_rl.environments.registry import load_environment

COMPLETION_TEMPLATE = """\
<think>
Okay, let's try to figure out why the user is getting that error. The issue is about the `sliding_window_inference` function in MONAI not preserving gradients. The user provided a code example where they create a tensor with `requires_grad=True`, pass it through the sliding window inference, and then try to do a backward pass. But they get an error saying that the result tensor doesn't require grad and has no grad_fn.

Hmm. The error message points to the line where they call `result.sum().backward()`. The traceback shows that the result tensor doesn't require grad. So, the problem is that the `sliding_window_inference` function is somehow detaching the gradient, which is not desired here.

Looking at the code for `sliding_window_inference`, I need to find where gradients might be detached. The user mentioned that the detach is happening in the line where they do `output_image_list[ss] = output_image_list[ss].detach()`. Wait, that's in the code after processing all the slices. So, the function is detaching the output tensors, which would prevent gradients from flowing back through that operation.

But why is that happening? The user says that the function shouldn't always remove gradients. The original code probably detaches the output to avoid accumulating gradients during the sliding window process, but in this case, the user is trying to compute gradients with respect to the input, so detaching is causing the problem.

So the fix would be to remove the `.detach()` call from the output tensors. Let me check the code again. In the part where they process the outputs, there's a loop where they do:

output_image_list[ss] = output_image_list[ss].detach()

That's the line that's detaching the tensor. If we remove that line, the gradients should flow through the inference process. But wait, why was that line there in the first place? Maybe the original intention was to prevent the gradients from being accumulated during the sliding window steps, but in this case, the user wants to track gradients through the entire process.

So the correct fix is to remove the `.detach()` call from the output tensors. That way, when the backward pass is called, the gradients can be computed properly.

Looking at the code in the provided file, the line in question is:

output_image_list[ss] = output_image_list[ss].detach()

So the SEARCH block would find this line, and the REPLACE block would remove the `.detach()` part. That should fix the issue where the gradients are being detached, allowing the backward pass to work as intended.
</think>

<solution>
The issue arises because the `sliding_window_inference` function is detaching the output tensors, which prevents gradient tracking. This is done in the line `output_image_list[ss] = output_image_list[ss].detach()`. To fix this, we need to remove the `.detach()` call so gradients can flow through the inference process.

Here is the fix:
{edits}
</solution>
"""

EDITS = [
    """
```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
=======
from distutils.log import warn

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
>>>>>>> REPLACE
```
""",
    '''
```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
    @deprecated_arg(name="reduction", since="0.8", msg_suffix="`reduction` is no longer supported.")
    def __init__(self, temperature: float = 0.5, batch_size: int = 1, reduction="sum") -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.
            batch_size: The number of samples.

        Raises:
            ValueError: When an input of dimension length > 2 is passed
            ValueError: When input and target are of different shapes

        .. deprecated:: 0.8.0

            `reduction` is no longer supported.

        """
        super().__init__()

        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
=======
    @deprecated_arg(name="reduction", since="0.8", msg_suffix="`reduction` is no longer supported.")
    def __init__(self, temperature: float = 0.5, batch_size: int = -1, reduction="sum") -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.

        Raises:
            ValueError: When an input of dimension length > 2 is passed
            ValueError: When input and target are of different shapes

        .. deprecated:: 0.8.0

            `reduction` is no longer supported.

        """
        super().__init__()
        self.temperature = temperature

        if batch_size != -1:
            warn("batch_size is no longer required to be set. It will be estimated dynamically in the forward call")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE
```
''',
    """
```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        negatives_mask = ~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input.device)
=======
        negatives_mask = ~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.clone(negatives_mask.type(torch.float)).to(input.device)
>>>>>>> REPLACE
```
""",
    """
```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        temperature_tensor = torch.as_tensor(self.temperature).to(input.device)

        norm_i = F.normalize(input, dim=1)
=======
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        temperature_tensor = torch.as_tensor(self.temperature).to(input.device)
        batch_size = input.shape[0]

        norm_i = F.normalize(input, dim=1)
>>>>>>> REPLACE
```
""",
    """
```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        sim_ij = torch.diag(sim_matrix, self.batch_size)
        sim_ji = torch.diag(sim_matrix, -self.batch_size)
=======
        sim_ij = torch.diag(sim_matrix, batch_size)
        # comment out the line below
        sim_ji = torch.diag(sim_matrix, -batch_size)
>>>>>>> REPLACE
```
""",
    """
```python
### monai/losses/contrastive.py
<<<<<<< SEARCH
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        return torch.sum(loss_partial) / (2 * self.batch_size)
=======
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        return torch.sum(loss_partial) / (2 * batch_size)
>>>>>>> REPLACE
```
""",
]


def test_swe_rl_environment_reward_computation():
    """Test reward computation for SWE-RL environment"""
    swe_env = load_environment("swe-rl", {"recompute_gt_patch": True})
    dataset = load_dataset("rasdani/SkyRL-v0-293-data-oracle-8k-context", split="train")
    dataset = dataset.map(
        lambda x: {
            "question": x["prompt"],
            "answer": x["patch"],
            "info": {"parsed_commit_content": x["parsed_commit_content"]},
            "task": "swe-rl",
        }
    )

    sample = dataset[0]

    parser = swe_env.parser

    total_rewards = []
    for i in range(len(EDITS) + 1):
        joined_edits = "\n".join(EDITS[:i])
        completion = COMPLETION_TEMPLATE.format(edits=joined_edits)
        parsed_edits = parser.parse_answer(completion)
        assert parsed_edits is not None, "Parser should successfully parse the completion"

        # Create inputs that match what the orchestrator would pass
        inputs = {
            "completion": completion,
            "answer": sample.get("answer", ""),
            "info": sample.get("info", {}),
            "question": sample.get("question", ""),
            "task": sample.get("task", "swe-rl"),
        }

        # Test reward computation
        rubric = swe_env.rubric
        total_reward = 0.0

        for func, weight in zip(rubric.reward_funcs, rubric.reward_weights):
            reward = func(**inputs)
            weighted_reward = reward * weight
            total_reward += weighted_reward
        total_rewards.append(total_reward)

        # Reward should be computed without errors
        assert isinstance(total_reward, float), "Total reward should be a float"
        assert -1.0 <= total_reward <= 1.0, "Total reward should be between -1.0 and 1.0"

    assert total_rewards[0] == -1.0, "Empty edits should return -1.0"
    assert all(total_rewards[i] < total_rewards[i + 1] for i in range(6)), (
        "Reward should increase with more correct edits"
    )
    assert total_rewards[6] == 1.0, "Reward should be 1.0 for perfect solution"
