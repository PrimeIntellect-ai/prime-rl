import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main():
    torch.set_default_device("cuda")
    torch.cuda.set_device(0)

    dist.init_process_group(backend="nccl")
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=[2], mesh_dim_names=["dp"])
    print(device_mesh)
    print(device_mesh["dp"].size())


if __name__ == "__main__":
    main()
