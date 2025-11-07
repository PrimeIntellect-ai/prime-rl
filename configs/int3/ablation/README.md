These are debug configurations for testing the INTELLECT-3 RL recipe on Qwen-32B.

To start the training, run this from the base directory of the project:

```bash
sbatch configs/int3/ablation/run.sh
```

**Before starting a run, always make sure that the latest configs are pushed so that we can link runs to commit hashes exactly.**