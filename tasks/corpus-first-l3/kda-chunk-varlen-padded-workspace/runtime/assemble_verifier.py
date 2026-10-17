"""Regenerate the single private script transported by the existing RemoteSlot."""

from pathlib import Path


def render(root: Path) -> str:
    assets = [
        ("build_kda.py", root / "environment" / "build_kda.py"),
        ("verify.py", root / "verifier" / "verify.py"),
        ("upstream_test.py", root / "verifier" / "upstream_test.py.txt"),
    ]
    text = """#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
test ! -e /vllm-workspace/vllm-ascend/csrc
test ! -e /vllm-workspace/vllm-ascend/tests
test ! -e /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
private=$(mktemp -d "$HOME/.kda-verifier.XXXXXX")
trap 'rm -rf -- "$private"' EXIT
"""
    for index, (name, path) in enumerate(assets):
        delimiter = f"PRIVATE_KDA_ASSET_{index}"
        content = path.read_text()
        if delimiter in content or not content.endswith("\n"):
            raise ValueError("private asset is not representable as an exact heredoc")
        text += f"cat > \"$private/{name}\" <<'{delimiter}'\n{content}{delimiter}\n"
    text += """python3 -I -B "$private/build_kda.py" --source "$PWD" --output "$private/build"
export ASCEND_CUSTOM_OPP_PATH="$private/build/opp/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=8
python3 -I -B "$private/verify.py" "$private"
"""
    return text


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    (root / "verifier" / "test.sh").write_text(render(root))
