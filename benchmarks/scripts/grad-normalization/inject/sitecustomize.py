"""Install the gradient-normalization A/B instrumentation when GN_OUT is set."""

import os

if os.environ.get("GN_OUT"):
    import gn_instrument

    gn_instrument.install()
