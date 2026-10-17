# Current-version Ascend feature-report migration

The deployment service now follows the vLLM 0.27.1 configuration contract at
the supplied pinned current-runtime revision. The retained external-DP
reporting utility still emits feature metadata as if it were using the older
configuration interface: a current configured feature is missing from its
report, and an input ignored by the current runtime is reported as active.

Repair the actual feature-attribution path in
`tests/e2e/nightly/multi_node/external_dp/scripts/utils.py`. Reports must agree
with the effective current configuration, preserve other existing feature
tags and avoid duplicates across multiple server commands. Both currently
supported spellings of the additional-configuration argument must work.
Malformed declared JSON configuration must remain a visible error.

The source-root `version_probe.py` executes two complete pinned environment
modules and shows the old/current API observations. This is a version check,
not a serving or NPU test. Run `@L2_PYTHON@ -I version_probe.py`, then
`@L2_PYTHON@ -I reproduce.py` from the source root. All server commands here
are local data fixtures and are never executed. Do not edit these fixtures or
start a server, model, SSH session or NPU job.
