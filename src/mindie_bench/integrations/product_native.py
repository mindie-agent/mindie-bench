"""[Role] Install a frozen trial corpus into the selected native product Store.
[Invariants] Runs in the selected product Python, uses that runtime's public
Store and consent authority, and reports actual installed component identities.
No task oracle, model request, external feed or contribution is accessed.
[Timeouts & Locks] The runner bounds this process; Store uses its native WAL lock.
[Side-Effect Policy] Writes only the supplied isolated native profile/state.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from importlib import metadata, util
from pathlib import Path

COMPONENTS = {
    "knowledge": ("mindie-knowledge", "mindie_knowledge"),
    "remote-dev": ("remote-dev", "remote_dev"),
    "diagnostics": ("mindie-diagnostics", "mindie_diagnostics"),
    "npu-top": ("npu-top", "npu_top"),
    "coordinator": ("mindie-coordinator", "mindie_coordinator"),
}


def identities():
    result = {}
    for name, (distribution, module) in COMPONENTS.items():
        installed = metadata.distribution(distribution)
        entry = Path(util.find_spec(module).origin).resolve(strict=True)
        files = {
            str(p.relative_to(entry.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(entry.parent.rglob("*.py"))
        }
        result[name] = {
            "distribution": distribution,
            "version": installed.version,
            "direct_url": json.loads(installed.read_text("direct_url.json") or "null"),
            "module_file": str(entry),
            "module_sources_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "module_source_file_count": len(files),
        }
    return result


def initialize(request):
    from mindie_knowledge import consent_store
    from mindie_knowledge.loop.store import Store

    corpus = Path(request["corpus"])
    root, domain = Path(request["root"]), request["domain"]
    consent_store.record_choice(request["consent"], "disabled")
    store = Store(root, domain)
    started = time.monotonic()
    cpu_started = time.process_time()
    stages = {}
    def progress(stage):
        print(json.dumps({"stage": stage, "elapsed_seconds": time.monotonic() - started,
                          "cpu_seconds": time.process_time() - cpu_started}),
              file=sys.stderr, flush=True)
    progress("store_open")
    try:
        contract = corpus / "publication-contract.json"
        graded = contract.is_file() and json.loads(contract.read_text()).get("schema") == (
            "mindie-publication-contract/2"
        )
        if graded:
            from mindie_knowledge.graded.index import read_units

            units = read_units(corpus)
            store.graded.install(units, source_identity=request["corpus_sha256"])
            count = len(units)
        else:
            packages = []
            for task in sorted((corpus / "tasks").iterdir()):
                files = {
                    p.relative_to(task).as_posix(): p.read_text() for p in sorted(task.rglob("*.md"))
                }
                # install_feed owns native validation; do not parse every YAML
                # document twice in the benchmark's preparation layer.
                packages.append({"files": files})
            stages["read_corpus_seconds"] = time.monotonic() - started
            progress("corpus_read")
            store.install_feed(
                packages,
                feed_ident="isolated-product-snapshot",
                source_revision=request["corpus_sha256"],
            )
            count = len(packages)
        stages["install_elapsed_seconds"] = time.monotonic() - started
        progress("native_install_completed")
        result = store.query(query="benchmark readiness", limit=1)
        stages["read_probe_elapsed_seconds"] = time.monotonic() - started
        progress("native_query_completed")
        return {
            "status": "completed",
            "corpus_format": "graded" if graded else "material_packages",
            "installed_units": count,
            "read_probe_sha256": hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
            "components": identities(),
            "python": sys.executable,
            "python_base_prefix": sys.base_prefix,
            "contribution_enabled": False,
            "model_calls": 0,
            "preparation_stages": stages,
        }
    finally:
        store.close()


def main():
    request = json.load(sys.stdin)
    if request["operation"] == "identities":
        result = {"components": identities(), "python_base_prefix": sys.base_prefix}
    elif request["operation"] == "initialize":
        result = initialize(request)
    else:
        raise ValueError("unsupported native product preparation operation")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
