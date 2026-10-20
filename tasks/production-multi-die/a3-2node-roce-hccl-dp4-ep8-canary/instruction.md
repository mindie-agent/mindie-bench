# A3 two-node RoCE HCCL DP=4 x EP=8 canary

You own a **two-node gang lease** across A3 hosts `192.168.13.165` and
`192.168.13.156`, 8 Dies per node. Both ends must expose the intended RoCE
interface as `HCCL_IF_IP` and `HCCL_SOCKET_IFNAME`, and `/dev/shm` must be free
of stale `hccl*` entries before the first collective. Run the real cross-node
`DP=4 x EP=8` AlltoAll canary.

The task exercises production failure modes: wrong NIC binding, a slow cross-node
link, `HCCL_EXEC_TIMEOUT`, and an orphaned peer after one end exits. A lease is
only valid when both hosts are reachable and assigned non-overlapping Dies.

Acceptance requires: (1) the gang lease records both hosts, HCCL interface
bindings and clean shared memory; (2) the cross-node AlltoAll completes with
rank-skew and SLA within the gates; (3) logits remain equivalent to the
single-node reference within tolerance; (4) after an injected one-end exit, the
surviving host's peer worker is reaped and both hosts end with clean
`/dev/shm/hccl*`; and (5) no rank exceeds the workspace or guard-workload
budget. Record the node pair, interface binding, timeout diagnosis and cleanup
evidence.

## Runtime prerequisites and current status

The standard L3 private oracle runs inside a single-host bubblewrap slot with
`--unshare-net`, so it cannot reach the second A3 host. This task therefore
requires an external two-host coordinator that exports
`MINDIE_BENCH_PEER_HOST`/`MINDIE_BENCH_HCCL_PEER` and a direct multi-node lease
(`MINDIE_BENCH_MULTINODE_DIRECT=1`). Without that coordinator the verifier exits
nonzero with `unsupported:multi_node_network_isolated` or
`unsupported:multi_node_coordinator_required`; a single-node sandbox is never
reported as a two-node acceptance.

## Blocked / incomplete

This task has no production acceptance path yet. The verifier preserves the
missing cross-node lease as an explicit report; it must not be promoted,
described as available, or counted as a completed benchmark.
