# mindie-bench 0.4.8: automatic native plugin release metadata

[中文](mindie-bench-0.4.8.md)

The shared publisher automatically prepares `mindie-plugin-release.json` for
Codex, Claude Code, Kimi Code, DSH, Antigravity and WorkBuddy. Repositories maintain
their source compatibility declarations; the verified package version, exact
commit and committed declaration determine the release metadata. Components
continue to use their existing release contract.

Preparation runs after packaging and before checksums and upgrade acceptance.
Missing metadata is generated. Existing metadata must match the candidate
version, source and compatibility declaration; valid bytes are preserved.
Missing, legacy or modified declarations fail before any release draft is created.

Repository identities, declaration paths and schema validation come directly
from the pinned published old updater. Regression tests generate assets from
six captured native declarations and exercise the old release selector. They
also cover incorrect existing assets, legacy declarations and source mismatch.
Real old-client upgrades, uploaded draft verification and a second upgrade from
the normal public feed remain part of the publication transaction.

This independent patch does not change the historical product channel, runtime
dependency pins or knowledge activation consent.
