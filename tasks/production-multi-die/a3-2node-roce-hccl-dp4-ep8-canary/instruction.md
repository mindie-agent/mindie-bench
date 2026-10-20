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
