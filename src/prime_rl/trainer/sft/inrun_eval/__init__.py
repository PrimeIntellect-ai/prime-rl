"""In-run evaluation of an SFT run's exported checkpoints (``[inrun_eval]``).

After the HF export the run's nodes stop training and serve the exports with vLLM; node 0 replays the TB4 collection
harness against them (sft-gen ``flow run``, verifiers 24107a4ea, mini-swe-agent 2.4.6) and scores pass@k.

- ``handoff``: the trainer side. Every rank frees the GPUs; ranks other than each node's ``local_rank 0`` exit, and
  that one becomes the node's eval process (``node``), with a plan (``plan``) of what to serve when.
- ``node``: per node, serves vLLM engines (``serving``) for the round's checkpoint and follows node 0 through a small
  coordination store (``coord``); node 0 also runs the routers, the harness (``harness``, ``flowcfg``), the scoring
  (``scoring``, ``tasks``) and writes the results.

Only ``handoff`` and ``coord`` import torch; the rest is plain Python, unit-tested on CPU.
"""
