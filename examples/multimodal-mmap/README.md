# Multimodal rollouts with mmap transport

The transport.toml example selects disk-backed rollout batches with bounded pending storage. Merge it into an RL configuration that supplies a model and environment. Set readers_per_rank to the number of CP/PP processes sharing each data rank's files, and size the disk budgets for your workload.

The multimodal adapters carry image references through rollout batches and materialize model inputs during training. The renderer and Verifiers submodules pin compatible revisions.

The optional src/prime_rl/templates/multinode_no_sync.sbatch.j2 template launches distributed jobs from an existing environment without syncing dependencies. Set slurm.template_path to that file and provide your own project directory, partition, node selection and run configuration. The template requests 116 CPUs per task; adjust that directive and the matching srun request for your cluster. Install dependencies before submitting a job.
