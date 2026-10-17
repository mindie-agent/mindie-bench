# Owned Ascend runtime recipe

This operator recipe makes the existing multifamily function projection
repeatable from exact local core images and historical source blobs. It is
not a runtime runner, an installation path or a published container release.
It contains no host/login/authentication configuration and performs no SSH.

The `a2` and `a3` profiles in `manifest.json` identify:

- The exact pre-existing local core image ID. These operator-owned checkpoints
  are not public pull URLs. They already contain compiled torch/torch_npu/vLLM,
  the exact failing RMSNorm projection and the environment used by the earlier
  image. This recipe does not reconstruct them from a generic CANN image.
- Both failing upstream commits, target-file paths and SHA-256 values.
- Exact original installed file hashes and expected projected file hashes.
- The previously observed G multifamily image identity for reconciliation,
  without claiming that another build will have the same image ID.

The required original core images are
`sha256:b79ae5d9b5a63ed16f92efef70e47a55c8b3c38bf17bd7ab988784e7eae9d187`
for A3 and
`sha256:d85730f56df057d890f9f1ded8f9d42b55e746e1cd1e9e3bb6a8adf36eeb4a16`
for A2. Operators need access to the appropriate exact local image, an existing
source repository containing both historical base objects, and a new owned
build context. A missing prerequisite is an explicit preparation gap.

## Prepare and inspect before building

This command performs only local Git reads and writes a new build context:

```sh
python3 tasks/ascend/runtime/prepare_context.py \
  --profile a3 --repo /absolute/existing-vllm-ascend-repository \
  --output /absolute/new-owned-build-context
```

It verifies the two historical blob hashes before writing. Only those failing
files, the projector, manifest, Dockerfile and existing
`scripts/ascend_probe.py` are copied. No Git objects, source history, private
verifier, gold patch, personal profile or fixed original backup is included.
`context-manifest.json` records exact file hashes. Existing output directories
are refused.

For a local dry-run, place copies of the exact original core files and failing
RMSNorm file under a new owned fixture directory, retaining
`vllm-ascend/vllm_ascend/...`. Then run:

```sh
python3 tasks/ascend/runtime/project_runtime.py \
  --manifest tasks/ascend/runtime/manifest.json --profile a3 \
  --base-image sha256:b79ae5d9b5a63ed16f92efef70e47a55c8b3c38bf17bd7ab988784e7eae9d187 \
  --runtime-root /absolute/owned-fixture-vllm-workspace \
  --historical /absolute/new-owned-build-context/historical
```

The default is read-only. Explicit `--apply` is for a new owned image build
only. The script checks all original/input/output hashes before mutation,
replaces three RoPE and two sampler functions, verifies historical AST equality
after the disclosed sampler-name alias, and preserves unrelated modern APIs.
Replacing complete old modules broke modern registration in an earlier image;
this recipe deliberately retains the surgical function projection.

The resulting full-file bytes must match the observed G projection hashes.
Function-source SHA-256 values and interpreter identity are also recorded.
`ast.dump` formatting/fields vary by Python version, so an AST hash from a
different interpreter is not a portable byte identity. Exact source/output
file hashes remain the cross-interpreter comparison.

## New build and identity

Only on an operator-owned host with that exact local base image:

```sh
docker tag sha256:b79ae5d9b5a63ed16f92efef70e47a55c8b3c38bf17bd7ab988784e7eae9d187 \
  mindie-bench-owned-core:a3-exact
test "$(docker image inspect --format '{{.Id}}' mindie-bench-owned-core:a3-exact)" \
  = sha256:b79ae5d9b5a63ed16f92efef70e47a55c8b3c38bf17bd7ab988784e7eae9d187
docker build --pull=false --network=none \
  --build-arg PROFILE=a3 \
  --build-arg BASE_IMAGE=mindie-bench-owned-core:a3-exact \
  --build-arg BASE_IMAGE_ID=sha256:b79ae5d9b5a63ed16f92efef70e47a55c8b3c38bf17bd7ab988784e7eae9d187 \
  --tag mindie-bench:owned-new-a3 /absolute/new-owned-build-context
```

Use an owned base alias that is absent or already has the exact expected ID;
do not replace an unrelated existing tag. The temporary alias makes `FROM`
portable across builders that do not accept a bare local image ID.
`BASE_IMAGE_ID` is the independently inspected checkpoint ID; it is a declared
build input, not an identity measured by a process inside the resulting image.
For A2 use the `a2` profile and its exact manifest base ID. The Dockerfile runs
the same projector, removes task helpers/tests/docs/examples, Git metadata,
bytecode and build archives under the projected runtime, copies the existing
DCMI probe to `/bench/ascend_probe.py`, and deletes its temporary build inputs.
Compiled libraries remain available. Source Git packs needed by the runner
must be provisioned separately outside all agent grants.

An existing verified image may be reused only with its exact identity and
matching task/runtime proof. Any new build gets a new actual Docker image ID,
fresh environment/probe/leakage evidence and hardware-specific soundness
before task promotion. Matching two projected files does not reproduce or
approve the entire image. Do not modify an active image/container to prepare
the next revision.

The current L3 boundary is an authorized trusted evaluation container with
privileged device access and per-trial namespaces. It is not a hostile-tenant
security boundary. Worker creation, idle-device admission, device allocation
and cleanup remain with the existing runner/operator contract.

## Probe and answer-exposure review

The recipe reuses `scripts/ascend_probe.py`, which queries the installed DCMI
ABI and reports physical device model, memory, PIDs and environment identity.
It does not allocate test tensors or call a model. Supply the new actual image
ID and the physical device list; validate its ABI output against the actual
hardware/npu-smi before trusting it. Missing libraries, nonzero DCMI results or
unreadable required version files must fail visibly. Probe output is not a
hardware business-verifier pass.

`leakage_inventory.py` is a standalone read-only operator utility copied from
the actual A2/A3 review. It is not installed in the benchmark runtime.
Before projection, retain untouched core original rope/sampler bytes in a
private control directory. Export the historical failing blobs there too;
do not collect "original fixed" references from an already projected image.
Keep those references outside every actual agent read grant.

Run it using the image's interpreter against every actual granted root:

```sh
python3 -B /private/operator/leakage_inventory.py \
  --root /usr --root /bin --root /lib --root /lib64 --root /etc \
  --root /usr/local/Ascend --root /usr/local/python3.12.13 \
  --root /usr/local/dcmi --root /vllm-workspace \
  --original rope /private/operator/rope-original.py \
  --original sampler /private/operator/sampler-original.py \
  --base rope /private/operator/rope-base.py \
  --base sampler /private/operator/sampler-base.py
```

The root list must match the actual configuration, including other runtime
grants and resolved symlink targets. The utility checks known exact source
files, named function ASTs, same-interpreter isolated/full-module bytecode and
bounded source archives. It inventories native compiled files without
universal decompilation. Vendor binary wrappers, nested archives, links,
unreadable or unfamiliar target code and size limits remain explicit review
items. Exit0 means a JSON inventory was generated, not acceptance. Preserve
raw `incomplete_requires_review` results and resolve exceptions with new
observations rather than rewriting them into a clean scan.

This recipe's local controls exercise API preservation, alias mapping,
fail-before-write hashes, dry-run immutability, owned cleanup, fixed-source
archive detection and full-module bytecode recognition. They are not an
actual Docker build, fresh leakage gate, NPU soundness or paid cohort.
