# mindie-bench 0.4.9: native Windows & WSL2 host contracts, Job Object process-tree drain, and 3-OS CI gate

[中文](mindie-bench-0.4.9.md)

`mindie-bench` now gates pull requests and `main` pushes on `windows-latest` alongside
`ubuntu-latest` and `macos-latest` in `release-ready`, hardening host execution,
process cleanup, and path normalization across native Windows (including `CP936` OEM
code pages and hosts without WSL) and WSL2.

The runner and local verifier introduce `host_platform` primitives that reject WSL2
`/mnt/[a-z]` Windows `.exe`/`.cmd` interop shims on POSIX, filter zero-byte
`WindowsApps` Store stubs and `System32\bash.exe` WSL stubs on Windows, and discover
user-scoped Git for Windows and WinGet `uv` installations. Subprocess launches enforce
`CREATE_NO_WINDOW`, Win32 Job Objects (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) for
deterministic grandchild cleanup, and bounded pipe reader/writer threads instead of
POSIX pipe `select()`.

Target-access scoring, MCP brokering, and artifact I/O enforce explicit UTF-8 encoding
and Unix `LF` (`\n`) line endings, normalize `\\?\` extended-length path prefixes, and
parse `file:///C:/...` wheel URIs on Windows.

This independent patch release does not modify the historical product channel, runtime
dependency pins, or frozen task assets.
