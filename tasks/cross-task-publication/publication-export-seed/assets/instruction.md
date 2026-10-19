# Preserve partial export outcomes

A local CPU export contains a private endpoint observation and an unrelated
public observation. The current build stages the valid material but silently
quarantines the rejected entry, so the caller cannot tell that its request was
only partially completed.

Repair the real export path. Content rejection must reach the caller with the
rejected entry identities, while retaining the valid staged batch. An
all-rejected request must report rejection without creating a batch. An entirely
valid request must still return normally. A second idle build must not duplicate
the original staging. No remote publication is required.

Run `@L2_PYTHON@ reproduce.py` from the source root.
