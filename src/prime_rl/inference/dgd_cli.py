"""Command-line interface for rendering Prime Dynamo graph deployments."""

import argparse
from pathlib import Path

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.dgd import (
    DynamoGraphRenderOptions,
    GPUSchedulingProfile,
    _parse_kubernetes_toleration,
    write_dgd_artifacts,
)
from prime_rl.utils.config import cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_config", type=Path)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prime-sha", required=True)
    parser.add_argument("--dynamo-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--run-name")
    runtime_class = parser.add_mutually_exclusive_group()
    runtime_class.add_argument("--gpu-runtime-class", default="nvidia")
    runtime_class.add_argument(
        "--no-gpu-runtime-class",
        action="store_true",
        help="Request nvidia.com/gpu resources without setting a Kubernetes RuntimeClass",
    )
    parser.add_argument("--gpu-architecture", required=True)
    parser.add_argument("--gpu-product", required=True)
    parser.add_argument("--gpu-node-pool", required=True)
    parser.add_argument("--gpu-node-pool-label", default="cloud.google.com/gke-nodepool")
    parser.add_argument(
        "--external-controller",
        action="store_true",
        help="Render only DGD inference workloads; an external controller owns orchestration and training",
    )
    parser.add_argument(
        "--trainer-gpus",
        type=int,
        default=1,
        help="Exact GPU request and limit for the chart-managed trainer",
    )
    parser.add_argument("--orchestrator-replicas", type=int, default=1)
    parser.add_argument("--trainer-replicas", type=int, default=1)
    parser.add_argument(
        "--orchestrator-command",
        help="Chart-managed command; must start with 'uv run orchestrator'",
    )
    parser.add_argument(
        "--trainer-command",
        help="Chart-managed command; must start with 'uv run trainer'",
    )
    parser.add_argument(
        "--image-toleration",
        action="append",
        default=[],
        type=_parse_kubernetes_toleration,
        help='Additional image-pod toleration as JSON, e.g. \'{"key":"dedicated","operator":"Exists"}\'',
    )
    parser.add_argument(
        "--gpu-toleration",
        action="append",
        default=[],
        type=_parse_kubernetes_toleration,
        help='Additional GPU-pod toleration as JSON, e.g. \'{"key":"capacity","operator":"Exists"}\'',
    )
    parser.add_argument("--model-cache-pvc")
    parser.add_argument("--shared-pvc")
    parser.add_argument("--image-pull-secret", action="append", default=[])
    parser.add_argument("--hf-token-secret")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = cli(InferenceConfig, args=["@", str(args.inference_config)])
    options = DynamoGraphRenderOptions(
        release_name=args.release_name,
        namespace=args.namespace,
        image=args.image,
        output_dir=args.output_dir,
        prime_sha=args.prime_sha,
        dynamo_sha=args.dynamo_sha,
        image_digest=args.image_digest,
        run_name=args.run_name or args.release_name,
        gpu_scheduling=GPUSchedulingProfile(
            runtime_class_name=None if args.no_gpu_runtime_class else args.gpu_runtime_class,
            architecture=args.gpu_architecture,
            product=args.gpu_product,
            node_pool=args.gpu_node_pool,
            node_pool_label=args.gpu_node_pool_label,
            additional_image_tolerations=tuple(args.image_toleration),
            additional_gpu_tolerations=tuple(args.gpu_toleration),
        ),
        external_controller=args.external_controller,
        trainer_gpu_count=args.trainer_gpus,
        orchestrator_replicas=args.orchestrator_replicas,
        trainer_replicas=args.trainer_replicas,
        orchestrator_command=args.orchestrator_command,
        trainer_command=args.trainer_command,
        model_cache_pvc=args.model_cache_pvc,
        shared_pvc=args.shared_pvc,
        image_pull_secrets=tuple(args.image_pull_secret),
        hf_token_secret=args.hf_token_secret,
    )
    for path in write_dgd_artifacts(config, options).values():
        print(path)
