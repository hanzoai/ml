#!/usr/bin/env python3
"""Hanzo Security secret scanner and Hanzo Insights CI telemetry emitter.

Implements local 25-rule secret detection aligned with cloud/apps/security/detect
plus git commit-range diff scanning, staged-index scanning, working-tree scanning,
optional bounded read-only liveness verification, SHA-256 fingerprint allowlisting,
POST /v1/security/scans batch upload, and POST /v1/capture Hanzo Insights telemetry.
Never prints, logs, or transmits raw secret values outside in-memory scanning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

MAX_SCAN_FILES_PER_BATCH = 500
MAX_SCAN_BYTES_PER_BATCH = 8 * 1024 * 1024
MAX_SINGLE_FILE_BYTES = 1024 * 1024
ZERO_SHA = "0" * 40

SEVERITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".spv",
    ".wasm",
    ".safetensors",
    ".gguf",
    ".ggml",
    ".ot",
    ".pth",
    ".pt",
    ".npy",
    ".npz",
    ".tgz",
    ".gz",
    ".zip",
    ".xz",
    ".zst",
    ".parquet",
    ".bin",
    ".so",
    ".dylib",
    ".dll",
    ".rlib",
    ".rmeta",
    ".o",
    ".a",
    ".woff",
    ".woff2",
    ".ttf",
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    severity: str
    description: str
    pattern: re.Pattern[str]
    group: int = 0
    min_entropy: float = 0.0


@dataclass(frozen=True)
class Finding:
    rule_id: str
    rule_name: str
    severity: str
    path: str
    line: int
    preview: str
    fingerprint: str
    commit: str = "worktree"
    verified: Optional[bool] = None


def build_rules() -> List[Rule]:
    specs = [
        (
            "private-key",
            "Private key block",
            "critical",
            "PEM private key material (RSA/EC/OPENSSH/DSA/PGP).",
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
            0,
            0.0,
        ),
        (
            "aws-access-key-id",
            "AWS access key ID",
            "critical",
            "AWS access key ID (AKIA/ASIA).",
            r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
            0,
            0.0,
        ),
        (
            "aws-secret-access-key",
            "AWS secret access key",
            "critical",
            "Value assigned to an AWS secret access key field.",
            r"""(?i)aws_?secret_?access_?key["' ]*[:=]["' ]*["']([A-Za-z0-9/+=]{40})["']""",
            1,
            3.6,
        ),
        (
            "gcp-api-key",
            "Google API key",
            "high",
            "Google or GCP API key (AIza...).",
            r"\bAIza[0-9A-Za-z_\-]{35}\b",
            0,
            0.0,
        ),
        (
            "github-token",
            "GitHub token",
            "high",
            "GitHub personal, OAuth, user, server, or refresh token.",
            r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
            0,
            0.0,
        ),
        (
            "github-pat-fine",
            "GitHub fine-grained PAT",
            "high",
            "GitHub fine-grained personal access token.",
            r"\bgithub_pat_[A-Za-z0-9_]{82}\b",
            0,
            0.0,
        ),
        (
            "hanzo-api-key",
            "Hanzo IAM or Cloud API key",
            "critical",
            "Hanzo IAM or Cloud bearer key (hk-/sk-hanzo-).",
            r"\b(?:hk-[A-Za-z0-9_\-]{24,}|sk-hanzo-[A-Za-z0-9_\-]{24,})\b",
            0,
            0.0,
        ),
        (
            "openai-api-key",
            "OpenAI API key",
            "critical",
            "OpenAI project, service-account, or legacy secret key.",
            r"\bsk-(?:(?:proj|svcacct)-[A-Za-z0-9_\-]{24,}|[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20})\b",
            0,
            0.0,
        ),
        (
            "anthropic-api-key",
            "Anthropic API key",
            "critical",
            "Anthropic API key (sk-ant-api03-...).",
            r"\bsk-ant-api03-[A-Za-z0-9_\-]{40,}\b",
            0,
            0.0,
        ),
        (
            "digitalocean-pat",
            "DigitalOcean personal access token",
            "high",
            "DigitalOcean v1 personal access token (dop_v1_...).",
            r"\bdop_v1_[a-f0-9]{64}\b",
            0,
            0.0,
        ),
        (
            "digitalocean-oauth",
            "DigitalOcean OAuth token",
            "high",
            "DigitalOcean v1 OAuth token (doo_v1_...).",
            r"\bdoo_v1_[a-f0-9]{64}\b",
            0,
            0.0,
        ),
        (
            "digitalocean-genai-key",
            "DigitalOcean GenAI secret key",
            "high",
            "DigitalOcean GenAI key (sk-do-...).",
            r"\bsk-do-[A-Za-z0-9_\-]{24,}\b",
            0,
            0.0,
        ),
        (
            "huggingface-token",
            "HuggingFace access token",
            "high",
            "HuggingFace Hub access token (hf_...).",
            r"\bhf_[A-Za-z0-9]{34,}\b",
            0,
            3.5,
        ),
        (
            "crates-io-token",
            "crates.io API token",
            "critical",
            "crates.io registry publish token (cio...).",
            r"\bcio[A-Za-z0-9]{32}\b",
            0,
            3.8,
        ),
        (
            "pypi-upload-token",
            "PyPI upload token",
            "critical",
            "PyPI macaroon API token (pypi-...).",
            r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_\-]{50,}\b",
            0,
            0.0,
        ),
        (
            "slack-token",
            "Slack API token",
            "high",
            "Slack bot, app, user, or workspace token.",
            r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b",
            0,
            0.0,
        ),
        (
            "slack-webhook",
            "Slack webhook URL",
            "medium",
            "Slack incoming webhook URL.",
            r"https://hooks\.slack\.com/services/[A-Za-z0-9_/]+",
            0,
            0.0,
        ),
        (
            "stripe-secret-key",
            "Stripe live secret key",
            "critical",
            "Stripe live secret or restricted API key.",
            r"\b(?:sk|rk)_live_[0-9A-Za-z]{24,}\b",
            0,
            0.0,
        ),
        (
            "npm-token",
            "npm access token",
            "high",
            "npm registry access token (npm_...).",
            r"\bnpm_[A-Za-z0-9]{36}\b",
            0,
            0.0,
        ),
        (
            "sendgrid-api-key",
            "SendGrid API key",
            "high",
            "SendGrid mail API key (SG....).",
            r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b",
            0,
            0.0,
        ),
        (
            "square-oauth-secret",
            "Square OAuth secret",
            "high",
            "Square application secret (sq0csp-...).",
            r"\bsq0csp-[0-9A-Za-z_\-]{43}\b",
            0,
            0.0,
        ),
        (
            "age-secret-key",
            "Age secret key",
            "critical",
            "Age encryption private key (AGE-SECRET-KEY-1...).",
            r"\bAGE-SECRET-KEY-1[QPZRY9X8GF2TVDW0S3JN54KHCE6MUA7L]{58}\b",
            0,
            0.0,
        ),
        (
            "url-basic-auth",
            "URL embedded basic-auth secret",
            "high",
            "High-entropy password embedded in HTTP(S) basic-auth URL.",
            r"""https?://[A-Za-z0-9._\-]+:([A-Za-z0-9/+_\-.~%]{14,})@[A-Za-z0-9.\-]+""",
            1,
            3.6,
        ),
        (
            "jwt",
            "JSON Web Token",
            "medium",
            "Signed three-part JWT credential.",
            r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
            0,
            0.0,
        ),
        (
            "generic-secret",
            "Generic assigned secret",
            "medium",
            "High-entropy literal assigned to secret/password/token/api_key.",
            r"""(?i)\b(?:password|passwd|secret|api_?key|access_?token|auth_?token|client_?secret)\b["' ]*[:=]["' ]*["']([A-Za-z0-9/+_\-]{20,}={0,2})["']""",
            1,
            3.7,
        ),
    ]
    return [
        Rule(
            rule_id=rid,
            name=name,
            severity=sev,
            description=desc,
            pattern=re.compile(pat),
            group=grp,
            min_entropy=ent,
        )
        for (rid, name, sev, desc, pat, grp, ent) in specs
    ]


RULES: List[Rule] = build_rules()

BENIGN_VALUE_PATTERNS = (
    re.compile(r"^[A-Z][A-Z0-9_]+$"),
    re.compile(r"^[a-z][a-z0-9_.\-/]+$"),
    re.compile(r"(?i)(?:example|sample|dummy|placeholder|changeme|redacted|your[_-]|test[_-]token|fake[_-])"),
    re.compile(r"\$\{\{|\$\(|%[sd]"),
)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = float(len(s))
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def fingerprint_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8", errors="surrogateescape")).hexdigest()


def mask_secret(secret: str) -> str:
    n = len(secret)
    if n <= 8:
        return "*" * n
    masked_mid = min(n - 8, 24)
    return f"{secret[:4]}{'*' * masked_mid}{secret[-4:]}"


def is_benign_generic(secret: str) -> bool:
    for pat in BENIGN_VALUE_PATTERNS:
        if pat.search(secret):
            return True
    has_upper = any(c.isupper() for c in secret)
    has_lower = any(c.islower() for c in secret)
    has_digit = any(c.isdigit() for c in secret)
    if sum((has_upper, has_lower, has_digit)) < 2:
        return True
    return False


def load_allowlist(repo_root: Path) -> Set[str]:
    allowed: Set[str] = set()
    ignore_file = repo_root / ".hanzo-security-ignore"
    if not ignore_file.is_file():
        return allowed
    for raw in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            allowed.add(line)
    return allowed


def verify_secret_readonly(rule_id: str, secret: str) -> Optional[bool]:
    """Bounded 5s read-only liveness check without logging or persisting secret."""
    headers = {"User-Agent": "hanzo-security-ci/1.0"}
    url: Optional[str] = None
    if rule_id in ("github-token", "github-pat-fine"):
        url = "https://api.github.com/user"
        headers["Authorization"] = f"Bearer {secret}"
    elif rule_id == "huggingface-token":
        url = "https://huggingface.co/api/whoami-v2"
        headers["Authorization"] = f"Bearer {secret}"
    elif rule_id in ("digitalocean-pat", "digitalocean-oauth"):
        url = "https://api.digitalocean.com/v2/account"
        headers["Authorization"] = f"Bearer {secret}"
    elif rule_id == "hanzo-api-key":
        url = "https://api.hanzo.ai/v1/models"
        headers["Authorization"] = f"Bearer {secret}"
    elif rule_id == "openai-api-key":
        url = "https://api.openai.com/v1/models"
        headers["Authorization"] = f"Bearer {secret}"
    if url is None:
        return None
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        return None
    except Exception:
        return None


def scan_text_lines(
    path: str,
    numbered_lines: Iterable[Tuple[int, str]],
    allowlist: Set[str],
    commit: str = "worktree",
    do_verify: bool = False,
) -> List[Finding]:
    seen: Set[Tuple[str, int, str]] = set()
    findings: List[Finding] = []
    for line_no, line_text in numbered_lines:
        if "hanzo-security:ignore" in line_text:
            continue
        for rule in RULES:
            for match in rule.pattern.finditer(line_text):
                secret = match.group(rule.group) if rule.group > 0 else match.group(0)
                if not secret:
                    continue
                if rule.min_entropy > 0.0 and shannon_entropy(secret) < rule.min_entropy:
                    continue
                if rule.rule_id == "generic-secret" and is_benign_generic(secret):
                    continue
                fp = fingerprint_secret(secret)
                if fp in allowlist or f"{rule.rule_id}:{fp}" in allowlist:
                    continue
                dedupe_key = (rule.rule_id, line_no, fp)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                verified = verify_secret_readonly(rule.rule_id, secret) if do_verify else None
                findings.append(
                    Finding(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        path=path,
                        line=line_no,
                        preview=mask_secret(secret),
                        fingerprint=fp,
                        commit=commit,
                        verified=verified,
                    )
                )
    findings.sort(key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), f.path, f.line, f.rule_id))
    return findings


def is_scannable_path(rel_path: str) -> bool:
    ext = Path(rel_path).suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return False
    if rel_path.startswith("target/") or "/target/" in rel_path:
        return False
    return True


def git_tracked_files(repo_root: Path) -> List[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    out: List[str] = []
    for raw in proc.stdout.split(b"\x00"):
        if not raw:
            continue
        rel = raw.decode("utf-8", errors="replace")
        if is_scannable_path(rel):
            out.append(rel)
    return out


def scan_worktree(
    repo_root: Path,
    allowlist: Set[str],
    do_verify: bool,
) -> Tuple[List[Finding], List[Tuple[str, str]], int]:
    files = git_tracked_files(repo_root)
    all_findings: List[Finding] = []
    upload_candidates: List[Tuple[str, str]] = []
    scanned_count = 0
    for rel in files:
        fpath = repo_root / rel
        if not fpath.is_file():
            continue
        try:
            size = fpath.stat().st_size
            if size == 0 or size > MAX_SINGLE_FILE_BYTES:
                continue
            raw = fpath.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="replace")
        scanned_count += 1
        fnds = scan_text_lines(
            path=rel,
            numbered_lines=enumerate(text.splitlines(), start=1),
            allowlist=allowlist,
            commit="worktree",
            do_verify=do_verify,
        )
        all_findings.extend(fnds)
        if len(raw) <= 256 * 1024:
            upload_candidates.append((rel, text))
    return all_findings, upload_candidates, scanned_count


def parse_unified_diff_and_scan(
    diff_text: str,
    allowlist: Set[str],
    default_commit: str,
    do_verify: bool,
) -> Tuple[List[Finding], int]:
    current_commit = default_commit
    current_path: Optional[str] = None
    new_line_no = 0
    buffered: List[Tuple[int, str]] = []
    findings: List[Finding] = []
    files_touched: Set[str] = set()

    def flush() -> None:
        nonlocal buffered
        if buffered and current_path and is_scannable_path(current_path):
            files_touched.add(current_path)
            findings.extend(
                scan_text_lines(
                    path=current_path,
                    numbered_lines=buffered,
                    allowlist=allowlist,
                    commit=current_commit,
                    do_verify=do_verify,
                )
            )
        buffered = []

    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for line in diff_text.splitlines():
        if line.startswith("COMMIT_SHA:"):
            flush()
            current_commit = line.split(":", 1)[1].strip()
            current_path = None
        elif line.startswith("+++ b/"):
            flush()
            current_path = line[6:].strip()
        elif line.startswith("+++ /dev/null"):
            flush()
            current_path = None
        elif line.startswith("@@ "):
            flush()
            m = hunk_re.match(line)
            if m:
                new_line_no = int(m.group(1))
        elif current_path is not None:
            if line.startswith("+") and not line.startswith("+++"):
                buffered.append((new_line_no, line[1:]))
                new_line_no += 1
            elif line.startswith(" "):
                new_line_no += 1
    flush()
    return findings, len(files_touched)


def resolve_git_range(repo_root: Path, since_commit: Optional[str], branch: Optional[str]) -> str:
    head = branch if branch and branch != ZERO_SHA else "HEAD"
    if since_commit and since_commit != ZERO_SHA:
        chk = subprocess.run(
            ["git", "cat-file", "-e", f"{since_commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        )
        if chk.returncode == 0:
            return f"{since_commit}..{head}"
    return f"{head}~20..{head}"


def scan_git_history(
    repo_root: Path,
    since_commit: Optional[str],
    branch: Optional[str],
    allowlist: Set[str],
    do_verify: bool,
) -> Tuple[List[Finding], int]:
    rev_spec = resolve_git_range(repo_root, since_commit, branch)
    proc = subprocess.run(
        ["git", "log", "--format=COMMIT_SHA:%H", "-p", "--no-color", rev_spec],
        cwd=repo_root,
        capture_output=True,
    )
    if proc.returncode != 0:
        head = branch if branch and branch != ZERO_SHA else "HEAD"
        proc = subprocess.run(
            ["git", "log", "-n", "20", "--format=COMMIT_SHA:%H", "-p", "--no-color", head],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    diff_text = proc.stdout.decode("utf-8", errors="replace")
    return parse_unified_diff_and_scan(diff_text, allowlist, "HEAD", do_verify)


def scan_staged(repo_root: Path, allowlist: Set[str], do_verify: bool) -> Tuple[List[Finding], int]:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--no-color", "-U0"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    diff_text = proc.stdout.decode("utf-8", errors="replace")
    return parse_unified_diff_and_scan(diff_text, allowlist, "staged", do_verify)


def submit_security_scans(
    project: str,
    files: Sequence[Tuple[str, str]],
    api_key: str,
    host: str,
) -> Optional[dict]:
    if not api_key or not files:
        return None
    batch: List[dict] = []
    batch_bytes = 0
    for path, content in files:
        encoded_len = len(content.encode("utf-8", errors="replace"))
        if len(batch) >= MAX_SCAN_FILES_PER_BATCH or batch_bytes + encoded_len > MAX_SCAN_BYTES_PER_BATCH:
            break
        batch.append({"path": path, "content": content})
        batch_bytes += encoded_len
    if not batch:
        return None
    url = f"{host.rstrip('/')}/v1/security/scans"
    payload = json.dumps({"project": project, "files": batch}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "hanzo-ml-ci-security/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"[hanzo-security] remote /v1/security/scans skipped: {type(exc).__name__}", file=sys.stderr)
        return None


def emit_insights_events(
    events: Sequence[Tuple[str, dict]],
    api_key: str,
    host: str,
    distinct_id: str,
) -> bool:
    if not api_key or not events:
        return False
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {
        "api_key": api_key,
        "batch": [
            {
                "event": name,
                "distinct_id": distinct_id,
                "timestamp": now_iso,
                "properties": props,
            }
            for name, props in events
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    base = host.rstrip("/")
    for endpoint in (f"{base}/v1/capture", f"{base}/v1/event", "https://insights.hanzo.ai/capture/"):
        req = urllib.request.Request(
            endpoint,
            data=raw,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hanzo-ml-ci-insights/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                continue
            break
        except Exception:
            continue
    return False


def dedupe_findings(items: Sequence[Finding]) -> List[Finding]:
    seen: Set[Tuple[str, str, int, str]] = set()
    out: List[Finding] = []
    for f in items:
        key = (f.rule_id, f.path, f.line, f.fingerprint)
        if key not in seen:
            seen.add(key)
            out.append(f)
    out.sort(key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), f.path, f.line, f.rule_id))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hanzo Security scanner + Hanzo Insights CI telemetry")
    parser.add_argument("--project", default="hanzo-ml", help="Hanzo Security project slug")
    parser.add_argument("--git", action="store_true", help="Scan git commit history diff in addition to worktree")
    parser.add_argument("--staged", action="store_true", help="Scan staged git index only (pre-commit mode)")
    parser.add_argument("--since-commit", default="", help="Base commit SHA for --git range scan")
    parser.add_argument("--branch", default="", help="Head ref or commit SHA for --git range scan")
    parser.add_argument("--verify", action="store_true", help="Run bounded read-only provider liveness check")
    parser.add_argument("--github-actions", action="store_true", help="Emit masked GitHub Actions annotations")
    parser.add_argument("--gate-stage", default="security", help="CI gate stage label for Insights")
    parser.add_argument("--gate-verdict", choices=["auto", "pass", "fail"], default="auto")
    parser.add_argument("--skip-worktree", action="store_true", help="Emit gate verdict only without scanning")
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    repo_root = Path(__file__).resolve().parent.parent
    allowlist = load_allowlist(repo_root)

    all_findings: List[Finding] = []
    upload_files: List[Tuple[str, str]] = []
    scanned_files = 0

    if not args.skip_worktree:
        if args.staged:
            fnds, cnt = scan_staged(repo_root, allowlist, args.verify)
            all_findings.extend(fnds)
            scanned_files += cnt
        else:
            wt_fnds, wt_uploads, wt_cnt = scan_worktree(repo_root, allowlist, args.verify)
            all_findings.extend(wt_fnds)
            upload_files = wt_uploads
            scanned_files += wt_cnt
            if args.git:
                hist_fnds, hist_cnt = scan_git_history(
                    repo_root,
                    args.since_commit or os.environ.get("BEFORE", ""),
                    args.branch or os.environ.get("AFTER", ""),
                    allowlist,
                    args.verify,
                )
                all_findings.extend(hist_fnds)
                scanned_files += hist_cnt

    findings = dedupe_findings(all_findings)
    remote_summary: Optional[dict] = None
    security_key = os.environ.get("HANZO_API_KEY", "").strip()
    security_host = os.environ.get("HANZO_SECURITY_HOST", "https://api.hanzo.ai").strip()
    if security_key and upload_files:
        remote_summary = submit_security_scans(args.project, upload_files, security_key, security_host)

    remote_findings = int((remote_summary or {}).get("findings", 0) or 0)
    total_findings = max(len(findings), remote_findings)
    duration_ms = round((time.monotonic() - t0) * 1000.0, 2)

    for f in findings:
        ver_str = "" if f.verified is None else f" verified={str(f.verified).lower()}"
        msg = (
            f"[{f.severity.upper()}] rule={f.rule_id} path={f.path}:{f.line} "
            f"commit={f.commit} preview={f.preview} sha256={f.fingerprint[:16]}...{ver_str}"
        )
        if args.github_actions:
            print(f"::error file={f.path},line={f.line},title=Hanzo Security ({f.rule_id})::{msg}")
        else:
            print(msg, file=sys.stderr)

    repo = os.environ.get("GITHUB_REPOSITORY", "hanzoai/ml")
    sha = os.environ.get("GITHUB_SHA", "")
    if not sha:
        sha_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True)
        sha = sha_proc.stdout.decode("utf-8", errors="replace").strip()
    ref_name = os.environ.get("GITHUB_REF_NAME", "local")
    workflow = os.environ.get("GITHUB_WORKFLOW", "local")
    runner_name = os.environ.get("RUNNER_NAME", os.uname().nodename)

    verdict = args.gate_verdict
    if verdict == "auto":
        verdict = "fail" if total_findings > 0 else "pass"

    common_props = {
        "date": "2026-11-22",
        "project": args.project,
        "repo": repo,
        "commit": sha,
        "branch": ref_name,
        "workflow": workflow,
        "runner": runner_name,
        "rules_loaded": len(RULES),
        "files_scanned": scanned_files,
        "findings": total_findings,
        "duration_ms": duration_ms,
    }
    insights_events = [
        ("ci.insights.security_scan", dict(common_props, remote_scan_id=(remote_summary or {}).get("id"))),
        ("ci.security_scan", dict(common_props, remote_scan_id=(remote_summary or {}).get("id"))),
        ("ci.gate_verdict", dict(common_props, stage=args.gate_stage, verdict=verdict)),
    ]
    insights_key = (
        os.environ.get("HANZO_INSIGHTS_API_KEY", "").strip()
        or security_key
    )
    insights_host = os.environ.get("HANZO_INSIGHTS_HOST", "https://api.hanzo.ai").strip()
    shipped = emit_insights_events(insights_events, insights_key, insights_host, distinct_id=f"ci:{repo}")

    summary = {
        "status": "fail" if (total_findings > 0 or verdict == "fail") else "ok",
        "date": "2026-11-22",
        "project": args.project,
        "rules": len(RULES),
        "files_scanned": scanned_files,
        "findings": total_findings,
        "remote_scan_submitted": remote_summary is not None,
        "insights_emitted": shipped,
        "duration_ms": duration_ms,
    }
    print(json.dumps(summary))

    return 1 if (total_findings > 0 or verdict == "fail") else 0


if __name__ == "__main__":
    sys.exit(main())
