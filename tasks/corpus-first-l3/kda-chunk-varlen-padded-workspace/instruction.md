Fix the native KDA forward operator's numerical correctness for finite
mixed-length sequences. On the supplied CANN runtime, a batch of 127 or 128
sequences with lengths cycling through 16, 32 and 64 produces non-finite
outputs. Dense length 80 and 129 inputs are useful controls. Both recompute
settings must remain correct, including the returned final state.

Work on the complete source worktree. Build your current native source with
`python3 -I build_kda.py --output "$HOME/kda-check-1"` and run
`python3 -I reproduce.py --build "$HOME/kda-check-1"` from its root. Source
`/usr/local/Ascend/ascend-toolkit/set_env.sh` first. Choose a new output path
for each build. The helper uses pinned offline dependencies, the
normal CANN build/installer and only the admitted NPU. A cold compile can
take several minutes; its logs and build receipt are in that output path.

Repair the source rather than changing the example, suppressing finite checks,
or redirecting to a preinstalled custom-operator package. Preserve ordinary
operator inputs, layouts, recompute behavior and final-state semantics.
The external judge independently rebuilds the resulting source and compares
its numerical outputs against a CPU reference. No model checkpoint or serving
process is required.
