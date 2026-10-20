# A3 16-Die same-node PD disaggregation with Mooncake KV handoff

Allocate the full **16-Die A3 node** as two isolated instances: Prefill owns
`Phy-ID 0..7` and Decode owns `Phy-ID 8..15` (or the equivalent whole-package
split from the lease). Each process must receive a disjoint
`ASCEND_RT_VISIBLE_DEVICES`; they must not share HCCL communicators or
`/dev/shm` state. Prefill hands KV cache to Decode through the real
Mooncake/HCCL transfer path.

Reproduce a long-prompt Prefill followed by continuous-batching Decode. The
historical failure is a handoff alignment/timeout or a first-token precision
drift after the KV block transfer. Include a no-handoff local Decode control.

Acceptance requires: (1) Prefill and Decode instances use disjoint Die sets and
each is healthy; (2) the KV handoff completes without timeout and the first
token agrees with the local control within tolerance; (3) Prefill TTFT and
Decode TPOT satisfy both SLA gates; (4) communication ratio, overlap ratio and
per-rank workspace remain inside the guard limits; and (5) both instances exit
cleanly with no residual `/dev/shm/hccl*` or orphan worker. Report the Die split,
handoff timing, TTFT/TPOT percentiles and guard results.
