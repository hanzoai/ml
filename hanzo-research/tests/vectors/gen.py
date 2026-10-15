"""Write python.json and wal.jsonl: canonical JSON and a WAL as hanzo-research (Python) writes them.

tests/canonical.rs holds the crate to these bytes. Regenerate from hanzo-research/:

    uv run --no-project --with hanzo-research==0.2.1 python tests/vectors/gen.py tests/vectors
"""

import json
import os
import sys
import shutil
import tempfile
from importlib import metadata

from hanzo_research import canonical, wal
from hanzo_research.wal import ResearchWal

# Each number as JSON text, so both sides read the same literal: an integer stays an
# integer and 100.0 a float until the canonical form makes them doubles. The two after
# 0.8734610296764017 sit halfway between two shortest decimals, where ECMAScript takes the even.
NUMBERS = [
    "0",
    "-0",
    "-0.0",
    "1",
    "-1",
    "100.0",
    "-3.0",
    "4.50",
    "0.1",
    "0.3333333333333333",
    "123456.789",
    "333333333.33333329",
    "0.8734610296764017",
    "-683210194955870.25",
    "207492967965050.625",
    "1e20",
    "999999999999999900000",
    "1e21",
    "1.5e300",
    "1e30",
    "1.7976931348623157e308",
    "0.000001",
    "1e-7",
    "2e-3",
    "1e-27",
    "5e-324",
    "2.2250738585072014e-308",
    "9007199254740991",
    "9007199254740992",
    "9007199254740993",
    "-9007199254740993",
    "18446744073709551615",
]

# Payloads of the WAL, each holding numbers past the range the two sides agreed on before 0.2.2.
PAYLOADS = [
    {"step": 1, "loss": -0.0, "tokens": 9007199254740993},
    {"step": 2, "lr": 1e-7, "flops": 1e21, "seconds": 100.0},
    {"step": 3, "eps": 5e-324, "max": 1.7976931348623157e308, "split": "dev", "é": "😀"},
]


def main(out: str) -> None:
    numbers = []
    for text in NUMBERS:
        v = json.loads(text)
        numbers.append({"text": text, "canonical": canonical.canonicalize(v).decode(), "digest": canonical.digest(v)})
    payloads = []
    for p in PAYLOADS:
        payloads.append({"value": p, "canonical": canonical.canonicalize(p).decode(), "digest": canonical.digest(p)})
    run = tempfile.mkdtemp()
    try:
        w = ResearchWal.open(run, "run-1")
        for p in PAYLOADS:
            w.append("step", p)
        last = wal.verify(w.path)
        shutil.copy(w.path, os.path.join(out, "wal.jsonl"))
    finally:
        shutil.rmtree(run)
    doc = {
        "package": "hanzo-research",
        "version": metadata.version("hanzo-research"),
        "numbers": numbers,
        "payloads": payloads,
        "wal": {"last": last, "sequence": len(PAYLOADS)},
    }
    with open(os.path.join(out, "python.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
