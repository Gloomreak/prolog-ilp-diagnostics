"""Пересоздаёт демонстрационные наборы в data/.

Примеры:
    python scripts/generate_demo_dataset.py
    python scripts/generate_demo_dataset.py --mode hard --out data/demo_hard.json
    python scripts/generate_demo_dataset.py --mode partial --out data/demo_partial.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from core.generator import generate_document  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=500)
    ap.add_argument("seed", nargs="?", type=int, default=42)
    ap.add_argument("--mode", choices=["easy", "hard", "partial"], default="easy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        ROOT / "data" / ("demo_dataset.json" if args.mode == "easy" else f"demo_{args.mode}.json"))
    doc = generate_document(args.n, args.seed, mode=args.mode)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано {len(doc['projects'])} проектов ({args.mode}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
