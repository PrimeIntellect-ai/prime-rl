import torch


def _seq_major(t):
    if t.dim() == 4:
        return t.transpose(1, 2).reshape(1, t.shape[2], t.shape[1] * t.shape[3])
    return t.reshape(1, t.shape[0], t.shape[1] * t.shape[2])


def reference(name, inputs):
    assert "cat" in name and "select_backward" in name, name
    q, k, v, out = inputs[:4]
    ref = torch.cat([_seq_major(q), _seq_major(k), _seq_major(v)], dim=-1)
    assert ref.shape == out.shape, (ref.shape, out.shape)
    return 3, ref
