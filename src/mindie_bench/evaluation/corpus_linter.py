# [Role] Rejects corpus contributions that violate publication quality rules.
# [Invariants] Fail closed: missing or uncertain evidence is never reported as success.
# [Timeouts & Locks] Callers bound external waits; no unbounded process or lock wait is owned here.
# [Side-Effect Policy] Writes only explicitly owned artifacts; no network or host user-directory mutation.

"""Reject unsafe new G3 advice without rewriting captured source material."""

from __future__ import annotations

import re

from mindie_bench.io import safe_relative

_GLOBAL_BAN = re.compile(
    r"(?:绝对|永远|一律|任何情况下).{0,16}(?:不要|禁止|不能|不得|避免)"
    r".{0,8}(?:使用|启用|调用|运行)"
    r"|\b(?:never|under no circumstances|always avoid)\s+(?:use|enable|call|run)\b",
    re.I,
)
_CONDITION = re.compile(
    r"\b(?:if|when|version|only with|only for|CANN\s*\d|torch[\w_-]*\s*[=v]?\d)"
    r"|(?:仅当|仅在|如果|当.{1,60}时|版本|触发条件)",
    re.I,
)
_VALIDATION = re.compile(r"\b(?:verify|test|reproduce|measure|profile|validate)\w*\b|验证|复现|测试|比对")
_ABSOLUTE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:ms\b|milliseconds?\b|毫秒|G(?:i)?B\s*/\s*s\b|GBps\b)",
    re.I,
)
_OPERATIONAL_WAIT = re.compile(
    r"timeout|deadline|poll|yield_time|wait|sleep|pytest|test|collect|init|startup|超时|轮询|等待预算|\d+(?:\.\d+)?\s*x\b",
    re.I,
)
_ROLE = re.compile(r"(?im)^###\s+(user|assistant)(?:\s*[:：].*)?\s*$")
_CHAIN = {
    "hypothesis": r"root cause(?: hypothesis)?|hypothesis|根因(?:假设)?|原因假设",
    "evidence": r"verification(?: evidence)?|validation(?: evidence)?|evidence|验证(?:证据)?|验证结果|证据",
    "boundary": r"applicability|scope|boundar(?:y|ies)|conditions|适用(?:范围|条件)?|生效边界|边界",
}
_PROSE_CHAIN = {
    "hypothesis": re.compile(
        r"\b(?:root\s+cause|because|caused\s+by|due\s+to|corrupt\w*|mismatch\w*|missing|ignores?|"
        r"falls?\s+back|unregistered|bug|defect|issue|fix(?:ed|es)?|updat(?:ed|es)|add(?:ed|s)|"
        r"register(?:ed|s)|replac(?:ed|es)|restor(?:ed|es)|preserv(?:ed|es))\b|根因|原因|由于|导致|修复",
        re.I,
    ),
    "evidence": re.compile(
        r"\b(?:verif\w*|validat\w*|reproduc\w*|tests?|pass(?:ed|es|ing)?|check(?:ed|s)?|"
        r"confirm\w*|pytest|bench\w*)\b|验证|复现|测试|通过",
        re.I,
    ),
    "boundary": re.compile(
        r"\b(?:when|if|for|in|under|with|only|version|CANN|Ascend|vLLM|DP|Eagle|Quarot|MoE|"
        r"MC2|GDN|RMSNorm|config\w*|layer\w*|model\w*|worker\w*|rank\w*|path\w*|mode\w*)\b"
        r"|适用|仅|当|在|版本",
        re.I,
    ),
}


class CorpusLintRejected(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            "G3 corpus rejected before publication/consumption: "
            + ", ".join(sorted({item["code"] for item in report["findings"]}))
        )


def _body(text):
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            return parts[1].lstrip("\n")
    return text


def _assistant_text(material):
    matches = list(_ROLE.finditer(material))
    if not matches:
        return ""
    return "\n".join(
        material[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        for index, match in enumerate(matches)
        if match[1].lower() == "assistant"
    )


def _causal_sections(text):
    labels = "|".join(f"(?P<{name}>{pattern})" for name, pattern in _CHAIN.items())
    pattern = re.compile(rf"(?im)^(?:#{{1,6}}\s*|[-*]\s*)?(?:{labels})\s*[:：]?\s*")
    matches = list(pattern.finditer(text))
    found = set()
    for index, match in enumerate(matches):
        value = text[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        value = re.split(r"(?m)^#{1,6}\s", value, maxsplit=1)[0].strip()
        if len(value) >= 12 and not re.match(
            r"(?i)^(?:TODO|please describe|fill in|describe here|请填写|待补|请描述)", value
        ):
            found.add(match.lastgroup)
    if not matches and len(text.strip()) >= 24:
        for name, prose_re in _PROSE_CHAIN.items():
            if prose_re.search(text):
                found.add(name)
    return found


def audit_packages(packages):
    """Audit newly exported native packages; never filter or modify their files.

    Causal sections must occur in an assistant-attributed source block. User
    instructions and a generated index alone cannot supply the required chain.
    The index limit concerns its rendered Markdown, excluding YAML metadata.
    """
    findings, checked = [], []

    def reject(identity, path, code, **details):
        findings.append(dict(task_id=identity, path=path, code=code, **details))

    if not isinstance(packages, list) or not packages:
        return dict(status="rejected", packages=[], findings=[dict(code="no_new_packages")])
    for package in packages:
        identity = package.get("task_id")
        files = package.get("files")
        if not isinstance(identity, str) or not isinstance(files, dict) or not files:
            reject(identity, None, "invalid_package")
            continue
        navigation, material = [], []
        for path, text in files.items():
            try:
                confined = safe_relative(path)
            except (ValueError, TypeError):
                reject(identity, None, "invalid_package_path")
                continue
            if not isinstance(text, str):
                reject(identity, confined, "non_text_material")
                continue
            body = _body(text)
            if confined in {"PACKAGE.md", "index.md", "INDEX.md"}:
                navigation.append(confined)
                if len(body.splitlines()) > 50:
                    reject(identity, confined, "index_line_budget", lines=len(body.splitlines()))
            elif confined.startswith(("materials/", "blocks/")) and confined.endswith(".md"):
                material.append(body)
            for paragraph in re.split(r"\n\s*\n", body):
                if _GLOBAL_BAN.search(paragraph) and not (
                    _CONDITION.search(paragraph) and _VALIDATION.search(paragraph)
                ):
                    reject(identity, confined, "unconditional_global_ban")
            for number, line in enumerate(body.splitlines(), 1):
                if _ABSOLUTE.search(line) and not _OPERATIONAL_WAIT.search(line):
                    reject(identity, confined, "absolute_performance_claim", line=number)
        if not navigation:
            reject(identity, None, "missing_native_index")
        source = _assistant_text("\n\n".join(material))
        missing = set(_CHAIN) - _causal_sections(source)
        if missing:
            reject(identity, None, "missing_causal_chain", missing=sorted(missing))
        checked.append(dict(task_id=identity, indexes=navigation))
    return dict(
        status="rejected" if findings else "passed",
        packages=checked,
        findings=findings,
        policy="g3-advisory-quality/1",
    )
