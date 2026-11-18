# mindie-bench 0.4.14: Calibrate Prebuilt `mindie-wheels-bundle.tar.gz` Size Ceiling (`16 MiB`)

[中文](mindie-bench-0.4.14.md)

`mindie-bench v0.4.14` calibrates `MAX_BUNDLE_BYTES` in `.github/scripts/build_wheels_bundle.py` to match the measured `5.64 MiB` archive size across all nine shared component wheels plus `reme-ai[as]` (`mindie_sim` alone is `2.48 MiB`):

- **Calibrate `MAX_BUNDLE_BYTES = 16 * 1024 * 1024` (`16 MiB`)**: Because `.whl` archives are already DEFLATE-compressed, bundling all ten verified pure-Python wheels plus `bundle-manifest.json`, `requirements.txt`, and `SHA256SUMS` produces a `5.64 MiB` deterministic `.tar.gz`; raising `MAX_BUNDLE_BYTES` from `3 MiB` to `16 MiB` aligns with `mindie-agent/install.py` while keeping all tag-SHA, `SHA256SUMS`, `METADATA`, and token-stripped build invariants intact.
