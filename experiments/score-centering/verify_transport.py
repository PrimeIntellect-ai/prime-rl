import numpy as np
import torch
import msgspec
from prime_rl.transports.batch.types import TrainingSample, MicroBatch, EncodedTensor
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.rl.data import DataLoader
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage
from prime_rl.orchestrator.trajectories import _score_heads
from types import SimpleNamespace

ids = np.array([[3, 2], [1, 0]], dtype=np.int32)
logq = np.log(np.array([[0.7, 0.2], [0.6, 0.3]], dtype=np.float32))
node = MessageNode(
    message=AssistantMessage(content="x"),
    token_ids=[4, 3, 1],
    mask=[False, True, True],
    score_head_ids=ids,
    score_head_logprobs=logq,
)
wire = msgspec.msgpack.encode(node.model_dump(mode="python"))
restored = MessageNode.model_validate(msgspec.msgpack.decode(wire))
np.testing.assert_array_equal(restored.score_head_ids, ids)
np.testing.assert_array_equal(restored.score_head_logprobs, logq)
head_ids, head_q = _score_heads(SimpleNamespace(nodes=[restored], token_ids=[4, 3, 1]))
sample = TrainingSample(
    token_ids=[4, 3, 1],
    mask=[False, True, True],
    logprobs=[0, -0.2, -0.3],
    temperatures=[1, 1, 1],
    env_name="test",
    advantages=[0, 1, -1],
    score_head_ids=head_ids,
    score_head_logprobs=head_q,
)
batches = prepare_batch([sample, sample], seq_len=8, num_train_workers=1, bin_cost=sum, pad_to_multiple_of=8)
mb = batches[0][0]
mb = msgspec.msgpack.decode(msgspec.msgpack.encode(mb), type=MicroBatch)
t = DataLoader._micro_batch_to_tensor(None, mb)
assert t["score_head_ids"].shape == (1, 8, 2)
assert t["score_head_ids"][0].tolist() == [[-1, -1], [3, 2], [1, 0], [-1, -1], [3, 2], [1, 0], [-1, -1], [-1, -1]]
torch.testing.assert_close(t["score_head_logprobs"][0, 1:3], torch.from_numpy(logq))
print("trace serialization, multi-sample packing, padding and tensor conversion pass")
