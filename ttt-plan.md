Learn online into a LoRA per rollout. Techniques that I want to support:

- sliding window learning
	- technique
		- have an 8k sliding context window (all number are just examples)
		- let's say we have 7k tokens in context
		- then we produce 1k more tokens
		- then we train on those 1k tokens, with the previous 7k in context
		- then we drop the oldest 1k
		- then we repeat
		- this way, there's always 7k tokens in context that were already trained on
			- increased information extraction, by having an addition to ICL
			- they serve as a link between what's in the weights and what's in attention context
		- and we can just keep this going forever and have both an approximate memory in the weights and a precise memory in the attention window
		- when there are input tokens, like a tool outputs, they might have more than 1k tokens at once; in that case, we just advance more tokens at a time (or 1k at a time if that's simpler to implement, i would prefer simplicity and guarantees of correctness over maximal efficiency for this initial implementation)
	- comments
		- you have to prefill 7k new tokens every 1k tokens, which is quite expensive and prefill heavy
		- but if it allows you to 10x or even 100x your effective context window with a constant cost (weight updates) instead of a quadratic one (attention), that would be well worth it
		- and i suspect that it would also increase sample efficiency (better than linear attention, which usually has a tiny constant-sized state, while this has the full model capacity, or at least a LoRA (though you could imagine doing full finetuning and just saving the diffs when we do RL with it, as we'll talk about later))
		- ultimately, this allows us to have 1M tokens in the attention window context and update every 100k tokens, and have an effective context window size of 1B tokens; and tool use and explicit memories and summarization are fully compatible with this paradigm
	- experiments
		- on some environment with tool use (that is multi-turn):
			- 32k full context window
			- vs 8k window with TTT for a total seq len of 32k
			- vs 8k sliding window for a total seq len of 32k, but without the training (so we just drop the tokens from the context window, a.k.a. give 0 learning rate to the online updates)
			- vs just 8k seq len
- training at compaction time
	- technique
		- you generate until the point where you want to compact
		- then you generate the summary with the full rollout in context
		- then you do an update step on all that
		- then you remove the old context and only keep the summary
	- comments
		- the explicit summary is now accompanied by implicit knowledge from the weight updates
		- the summary itself serves as the connection between what's in the weights and what's in context; after all, there needs to be some hint for what should be retrieved from the weights, just because something has been newly trained doesn't mean that it's automatically the knowledge that is being used, but the summary as a connection might be able to get this done (maybe, requires an inversion of order)
	- experiments
		- same environment
			- once with compaction but without this TTT
			- once with compaction and with this TTT
- training at compaction time with explicit lessons
	- technique
		- instead of just doing a forward and backward pass on the context, we spend more compute for a stronger learning signal
		- we let the model spend a lot of compute to generate question-answer pairs about the context
			- knowledge contained
			- approaches that worked
			- approaches that didn't work
			- theories of what's going on
			- the setup of the task
			- etc.
		- then instead of just training on the rollout itself, we train on that Q&A dataset (both techniques are compatible)
	- comments
		- The Cartridges paper showed that such a synthetic Q&A dataset is highly effective at making the knowledge that is implicit in some piece of text explicitly and flexibly available to the model when trained on it
		- a.k.a. we can move the power of ICL into the weights that way, which a simple forward-backward pass cannot easily do
		- as we'll see later, for RL we won't retain any of the learned knowledge in the other techniques beyond what's in the rollout already, but this Q&A dataset is useful as an extra source of training data that can be leveraged
	- experiments
		- same as above, just with this technique as an additional comparison
			- once with the Q&A only as a context extension technique
			- once by also training the main weights on the Q&A

How to do RL with this:

- do the full rollout using one of the techniques described above
- every time there's an update to the LoRA, do a checkpoint of the LoRA
- once the group of rollouts is done, each token is assigned an advantage
- replay the rollout with each LoRA checkpoint loaded exactly as it was at inference time
- do the RL weight update with that LoRA checkpoint loaded in the trainer (but frozen, so that only the main weights are updated) and generate the gradient using the advantage and that LoRA
- then dismiss that LoRA fully and move on to the next 1k tokens, or the next action after compaction
- it is crucial that the alignment between LoRA weights at inference time and at weight update time is exact, because the LoRA acts as context just like the tokens do

The Q&A updates are of course also in the LoRA and get dismissed, but we can simply train on them in one SFT step after the RL update step is done. That way, we can recycle the compute spent on them into a permanent weight update.

The two techniques I'm most excited about by far are the ones with compaction.

What's to do, in this order:

- your job
	- find a general abstraction in prime-rl that can do this, re-using as many existing components as possible -> write a plan (i will review)
	- implement and unit test the components thoroughly on a feature branch (testing can be done on a remote machine, but things should be implemented locally and then pushed and then tested remotely; more on that when we get there)
	- create the experimental setup (more on that when we get there)
- my job
	- run the experiments on the cluster
- our job
	- investigate the results, re-plan if something went wrong, analyze the behaviors etc.
	- write out the results in an article

