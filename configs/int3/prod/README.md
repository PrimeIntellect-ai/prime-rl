These is our production configuration and start command for INTELLECT-3 RL.

To start the training, run this from the base directory of the project:

```bash
bash configs/int3/prod/start.sh
```

To resume a run, make sure to edit the `CKPT_STEP` variable in the `run.sh` SLURM script, or alternatively pass it via environment variables.

**Before starting a run, always make sure that the latest configs are pushed so that we can link runs to commit hashes exactly.**