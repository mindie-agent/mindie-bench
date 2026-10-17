"""Benchmark deployment fixture; this is not an upstream historical bug."""

import os


def configure(deployment_options):
    """Apply the operator's requested compatibility override to a fleet config."""
    options = dict(deployment_options)
    os.environ["VLLM_ASCEND_ENABLE_FUSED_MC2"] = "0"
    return options
