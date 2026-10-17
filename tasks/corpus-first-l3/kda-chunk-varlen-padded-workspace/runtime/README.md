# Exact runtime inputs and repair-task admission

The earlier A3 operator proof used preowned local image
`sha256:333cadf615f4d5174bd23c735d53298689f81291079e9b6ae429971dc2a61f36`.
It is not a public image release or a publicly pullable URL. Its mutable proof
container is not a replacement immutable image identity.

The subsequent owned checkpoint for the clean task recipe is
`sha256:615d7645ada461ebad836eabe1f98796ff7a3d72ba3a33e899696fa43284c42b`.
It preserves the source/vendor projection and adds the exact generic dependency
archive below. Its own source/gold proof is separate from the older operator
and qualified cold-task records. Reuse requires that exact available image;
a newly built image needs a new identity and matching soundness proof.

Observed dependencies: CANN 9.1.0 (toolkit build 20260730), Torch 2.10.0+cpu,
torch-npu 2.10.0.post4, vLLM 0.27.1+empty, Python 3.12.13, openEuler
GCC 12.3.1, driver 25.5.3. Installed plugin distribution metadata is
`0.19.1rc2.dev1927+g9b8bf6244`; it must not be relabeled as a clean 0.27
plugin distribution. The actual extension has the twenty-argument
`torch.ops._C_ascend.chunk_kda_fwd` schema recorded in the private proof.

The dependency manifest records five official source archives and one frozen
generic build archive. The latter contains only patched Protobuf source/header
material, the normally source-built `protoc` 25.1 executable, generated generic
options and patched Abseil source/header material. It contains no OPP package or
KDA host/kernel object. Its archive SHA256 is
`13c73902399a1c3524ad8acd5f5386ee48b6c4a2a6a19774ed287c728d5142ea`;
the full inner file manifest SHA256 is
`a0703576c46611f02ef16e75caec755d2ba0555c60e7671410b53c4f9dd542fe`.
It comes from the preserved real cold source build, not a public download URL.

Place all six locked inputs in an operator-owned directory. Run:

```sh
python3 prepare_inputs.py --downloads /absolute/downloads --output /absolute/inputs
```

The command verifies all bytes before creating the output and stages only
dependencies. Copy the resulting directory into the new owned runtime at
`/bench/kda-runtime-inputs`. Mount only that exact directory as an extra
foreground read path; never grant `/bench` as a whole. No compiled KDA
binary, gold source, test module or source history belongs there.

Materialize the source repository separately, using existing local Git objects:

```sh
python3 prepare_repository.py --source-repo /absolute/upstream-repository --output /absolute/base-only.git
```

This uses the complete historical tree and original commit object, verifies
`git fsck`, and rejects history/gold exposure. It does not change the upstream
checkout. Copy that bare repository to a backend-only path, for example
`/bench/source-kda`, and use that exact path in the remote configuration.
`prepare_runtime.py --runtime /vllm-workspace/vllm-ascend
--private-backup /bench/kda-private-runtime-backup` preserves the known runtime
source/tests/vendor outside foreground grants. It operates only inside a
dedicated owned container. Its result still requires an actual grant inventory
and a new immutable image fingerprint.

The public helper copies the complete current `csrc` tree to a new HOME output
directory and excludes only its top-level generated `build`, `build_out` and
`output` directories. It retains `cmake/third_party/build`, which contains
required upstream patches. It verifies and stages exact Catlass, Makeself,
JSON, Abseil and Protobuf inputs. It verifies every generic-prefix file and
rejects extra compiled objects, then uses the upstream native CMake
`ASCEND_PROTOBUF_SHARED_INCLUDE` and `ASCEND_PROTOC` variables to reuse only
the generic compiler/headers. A missing or corrupt prefix is an error.
The normal `csrc/build.sh --pkg` command
rebuilds both KDA native code and its selected dependency closure; the normal
installer rejects unresolved ELF dependencies. There is no fallback to a
prebuilt custom operator, network download or forced install.

An operator can bootstrap the generic artifact by explicitly running
`build_kda.py --cold-generic --source /absolute/base-source --inputs
/absolute/five-source-archives --output /absolute/fresh-cold-build`. This uses
the original normal CANN source build with the five source archives; there is
no silent cold fallback in normal trials. After it completes, run
`prepare_generic.py --source /absolute/fresh-cold-build/source --output
/absolute/generic-prefix`. The utility verifies the original CMake rule/patch
identities, preserves a complete file manifest, checks the real compiler
version and rejects target-operator material. A rebuild at a different path
may change executable/archive bytes; new bytes require a new locked recipe,
image identity and soundness proof rather than automatic acceptance.

Before admitting a repair task, create a new owned runtime from the recorded
base, fingerprint it, and satisfy these separate gates:

1. Preserve and then hide the old runtime custom vendor. Also hide the runtime
   repository's `csrc` and `tests`, Git histories, archives and other known
   fixed-source copies from the actual foreground read roots. The task's base
   worktree supplies its complete failing source. Keep the matching installed
   Python wrapper/native extension as an explicit ABI dependency.
2. Install a base-only, full-tree shallow source repository with no gold object,
   ancestry or alternates. The source repository is backend-only; only the
   currently allocated source worktree is bound into the foreground.
3. Admit exactly one idle physical device using a fresh DCMI readback and a
   cooperative device lease. Configure the actual Torch visible-device count
   to one. A3 uses `ascend910_93`; other hardware needs a newly versioned recipe
   and new proof. The earlier physical allocation is not a durable lease.
4. Run the existing RemoteSlot source-fail/gold-pass soundness flow against the
   exact new image, task hash, runtime read roots and complete source worktree.
   Its private verifier independently rebuilds edited source and runs the full
   historical test module. `assemble_verifier.py` regenerates its self-contained
   `test.sh`, because RemoteSlot transports that script rather than siblings.
5. Retain source/build/install/numerical and cleanup errors separately.
   Nonzero compile/import/driver errors without a structured business result
   are infrastructure failures. Do not retry an unknown operation.

Budget compilation realistically: the first investigation builds took about
fourteen minutes with initially serial generic compilation. A later complete
RemoteSlot cold-source revision took 386.92 and 391.48 seconds for its two
build/install stages. That revision passed source0/gold1 but included generated
local task bytecode and is retained as qualified historical evidence, not a
clean-checkout task proof. Those elapsed times include the real KDA compile,
so they do not measure Protobuf cost alone. The normal recipe now reuses the
explicit frozen generic dependency and still rebuilds the KDA code. Exact new
durations belong to its own proof. The task allows 3600 seconds foreground work
and 2400 seconds external verification; the remote deadline must also allow
the entire compile and judge.

The private oracle/test files must never be copied with public `environment`
assets. A passing known-source scan only covers its recorded source patterns,
files and granted roots. Arbitrary CANN/driver/vendor compiled objects remain
an explicit inspection boundary. This task does not claim full-model,
ACLGraph, distributed, performance or A2 acceptance.
