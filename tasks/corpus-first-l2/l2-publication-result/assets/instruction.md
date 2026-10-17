# Real-source CPU protocol repair

A local CPU publication protocol test has one quarantined draft and one valid draft. Its fixture transport returns a confirmed publication, followed by a local receipt or cleanup failure. Preserve the returned business result and the separate failure without replaying the transport. Genuine uncertain transport outcomes must remain unknown. No network publication is required.

Run `@L2_PYTHON@ reproduce.py` from the source root.
