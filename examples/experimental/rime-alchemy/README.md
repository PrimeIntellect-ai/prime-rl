# RIME Alchemy joint-training smoke run

`rl.toml` runs two optimizer steps on Qwen3-4B, using four independent memory
streams per chemistry and ten 20-action trials per stream. ACT and WRITE share
one trainable policy. Clef-flash supplies the actor's exploration bonus through
Prime Inference; environment rewards determine downstream writer improvement.

The single-node allocation is two trainer GPUs plus one inference GPU. The
trainer keeps a frozen, unoptimized copy of the initial model for reference KL,
so no additional inference endpoint or GPU is required. Its additional memory
and forward-pass cost must be measured on the first run.

The taskset and immutable public Alchemy metadata are packaged in
`packages/rime-alchemy`. Its environment schema is included in `prime-rl-configs`
so hosted validation does not need to import the Alchemy runtime. The runtime
uses the Apache-licensed upstream Alchemy wheel checked into that package.
The released stream metadata comes from `Racktic/memory-writing-train-tasks`,
revision `f0a010540455eaf8dc0855704cc53a62316d1ba0`; source hashes and upstream
revision are in the packaged `sources.lock.json`. Development validation is
reserved by identical hidden-chemistry group and is not the authors' test set.

For Model Factory, use a volume-hosted immutable model snapshot when the base
model is not cached on the target cluster. The prepared snapshot is
`Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`. Explicitly
target the Denvr cluster and pair this branch with image `commit-3a0c52e75`.
Install the locked workspace packages on remote workers; no local editable
install is assumed to transfer to a hosted run.

K=2, beta=0.1, the memory budget, and the optimizer/clipping values are pilot
choices. They do not establish an exact reproduction of unpublished author
hyperparameters. Checkpoint saves retain both smoke steps. Trainer-local reference
scoring currently supports dense full-parameter training with CP=EP=1; resume
is rejected until a separately pinned original-base reference is specified.

Record actor raw reward, exploration reward, writer improvement, format validity,
context/timeout errors, token staleness, reference KL and peak GPU memory. A
successful smoke run establishes functioning joint optimization, not a learned
memory-quality improvement.
