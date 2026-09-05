"""RDMA rail defaults for ModelExpress refit receivers.

The inference receiver calls this before constructing its NIXL agent. Trainer
publishers intentionally remain unpinned: each receiver needs shards from every
FSDP rank, and on fabrics with isolated rail subnets a single-rail publisher
would be unreachable from receivers pinned to the other rails.

Why this is prime-RL's problem and not purely ModelExpress's: prime-RL puts
every inference reader for a node in one pod. On a four-GPU inference pod that
is four processes each independently asking UCX for a NIC, with no knowledge of
each other. Where a pod holds fewer usable rails than GPUs - or holds rails on
the wrong socket - they all make the same locally-correct choice and converge on
one adapter. Measured on such a pod:

- four readers converged on one rail: 1.5 GB/s each, 6.1 GB/s aggregate
- four readers on four distinct rails: 6.75 GB/s each, 27.0 GB/s aggregate

Aggregate with four readers sharing was *lower* than a single reader alone
(26.1 GB/s), so throughput is destroyed rather than divided. NeMo-RL did not hit
this with the same ModelExpress client because its workers spread across three
nodes, leaving fewer readers to converge per rail. The exposure is a property of
how the framework packs readers, which is why the default belongs here.
"""

from __future__ import annotations

import os

# Global GPU->NIC assignment computed from a PCIe sysfs walk. Every rank derives
# the same map from the same snapshot, so this needs no coordination between
# processes and is safe to set per-process.
_NIC_PIN = "MX_RDMA_NIC_PIN"

# Per-rank floor in Gbps, below which the refit logs a structured
# refit-slow-throughput-v1 warning. It warns and does not abort, so it cannot
# fail a run. A client too old to know the floor ignores it, so setting it is
# safe either way; the staged receiver this path pulls over reports it as of
# ai-dynamo/modelexpress#689.
#
# The floor is per rank, and the rate to compare it against is therefore the rate
# one reader sees while its siblings are also reading - not the headline number.
# On the 400 Gb/s rails this was measured on, a lone reader reaches ~26 GB/s
# (~208 Gbps), but four readers spread across four rails get ~6.75 GB/s each, or
# ~54 Gbps, because the aggregate is what saturates. The collapse this is meant to
# catch ran at ~1.5 GB/s, or ~12 Gbps.
#
# So the window is 12 to 54, and 25 sits near its geometric mean: 2x above the
# collapse, 2.2x below healthy. An earlier value of 50 was chosen against the
# 208 figure by mistake, which would have put the floor 8% under the healthy
# four-reader rate and fired it on ordinary variance - a guard that cries wolf on
# a good run gets turned off, which is worse than not having one. On slower
# fabrics, or on pods packing more readers per rail by design, this wants
# lowering; hence the override.
_MIN_GBPS = "MX_RESHARD_MIN_GBPS"
_DEFAULT_MIN_GBPS = "25"


def apply_rdma_defaults() -> None:
    """Opt into rail spreading and slow-transfer reporting, if not already set.

    Values are only defaulted, never overridden, so an operator who has
    tuned ``UCX_NET_DEVICES`` by hand or who wants a different floor keeps
    control. Call before the NIXL agent is constructed: the pin is read when the
    agent initialises, and setting it afterwards has no effect.
    """
    os.environ.setdefault(_NIC_PIN, "auto")
    os.environ.setdefault(_MIN_GBPS, _DEFAULT_MIN_GBPS)
