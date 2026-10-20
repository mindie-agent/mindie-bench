"""[Role] Adapt the existing C5 Store to benchmark retrieval and exact readback.

[Invariants] Contract/2 and all authoritative units are validated by knowledge;
no legacy package or derived-index fallback supplies missing graded content.
[Timeouts & Locks] Reuses Store's WAL/5s busy timeout and single MCP owner thread.
[Side-Effect Policy] Installs a local reviewed snapshot into an owned Store;
does not publish or upgrade local import into a formal feed acceptance claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from mindie_bench.io import tree_hash
from mindie_bench.runner.graded_view import units


class GradedConsumer:
    def __init__(self, corpus: Path, state: Path):
        from mindie_knowledge.graded.units import GradedUnit
        from mindie_knowledge.loop.store import Store
        from mindie_knowledge.publication_contract import parse_contract

        raw = (corpus / "publication-contract.json").read_bytes()
        contract = parse_contract(raw, json.loads(raw)["domain"])["contract"]
        if contract["schema"] != "mindie-publication-contract/2":
            raise ValueError("graded consumer requires publication contract/2")
        self.store = Store(state, contract["domain"])
        try:
            incoming = []
            for path, _ in units(corpus).values():
                incoming.append(
                    GradedUnit.from_text(path.read_text(), path.relative_to(corpus).as_posix())
                )
            self.store.graded.install(incoming, source_identity=tree_hash(corpus))
        except BaseException as error:
            try:
                self.store.close()
            except Exception as cleanup:
                error.add_note(f"graded store cleanup also failed: {cleanup}")
            raise

    def search(
        self,
        query=None,
        limit=5,
        continuation=None,
        conditions=None,
        include_unknown=False,
        domain_path=None,
        token_budget=2400,
    ):
        return self.store.query(
            query=query,
            limit=limit,
            continuation=continuation,
            conditions=conditions,
            include_unknown=include_unknown,
            domain_path=domain_path,
            token_budget=token_budget,
        )

    def explain(self, ref):
        return self.store.explain(ref)

    def close(self):
        self.store.close()
