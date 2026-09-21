"""Minimal CLI: match a name against a CSV list. Columns: id,name,<extra...>."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .matcher import Matcher
from .models import GoogleCloudAuth
from .tools import InMemorySearchTool
from .types import MatchConfig, Record


def load_csv(path: Path) -> list[Record]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        records = []
        for row in reader:
            rid = row.pop("id", None) or str(len(records) + 1)
            name = row.pop("name", "")
            extra = {k: v for k, v in row.items() if v not in (None, "")}
            records.append(Record(id=rid, name=name, extra=extra))
        return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="namematch", description=__doc__)
    parser.add_argument("name", help="Name to match")
    parser.add_argument("--list", required=True, type=Path, help="CSV with id,name columns")
    parser.add_argument(
        "--model", default=None, help="Model spec, e.g. google-cloud:gemini-3.8-flash"
    )
    parser.add_argument(
        "--service-account", type=Path, default=None, help="Service-account JSON key (Vertex AI)"
    )
    parser.add_argument("--project", default=None, help="Google Cloud project (Vertex AI)")
    parser.add_argument("--location", default=None, help="Vertex AI location, e.g. global")
    parser.add_argument(
        "--context", action="append", default=[], metavar="K=V", help="Context fields"
    )
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--trace", action="store_true", help="Print the search trace")
    args = parser.parse_args(argv)

    records = load_csv(args.list)
    extra_fields = sorted({k for r in records for k in r.extra})
    tool = InMemorySearchTool(records, extra_fields=extra_fields)
    config = MatchConfig()
    if args.max_searches:
        config = config.model_copy(update={"max_searches": args.max_searches})
    context = dict(kv.split("=", 1) for kv in args.context) or None

    google_cloud = None
    if args.service_account or args.project or args.location:
        google_cloud = GoogleCloudAuth(
            service_account_file=args.service_account,
            project=args.project,
            location=args.location,
        )

    matcher = Matcher(tool, model=args.model, config=config, google_cloud=google_cloud)
    run = matcher.match(args.name, context)
    out = run.model_dump(exclude={"seen_records"} if not args.trace else set())
    if not args.trace:
        out.pop("searches", None)
    out["match"] = run.match.model_dump() if run.match else None
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
