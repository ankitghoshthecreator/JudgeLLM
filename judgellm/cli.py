"""
Unified CLI for JudgeLLM.
Commands:
  - judgellm audit --traces traces.jsonl --labels labels.jsonl --config config.yaml
  - judgellm route --target-kappa-loss 0.02
  - judgellm serve --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from judgellm.audit import audit_judge


def cmd_audit(args: argparse.Namespace) -> None:
    print(f"Auditing judge on traces: {args.traces} using labels: {args.labels}")
    report = audit_judge(
        traces=args.traces,
        labels=args.labels,
        judge_config=args.config,
        rubric_path=args.rubric,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "audit_report.json"
    md_path = out_dir / "audit_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    print("\n" + report.to_markdown())
    print(f"\nAudit complete! Reports saved to:\n  - {json_path}\n  - {md_path}\n")


def cmd_route(args: argparse.Namespace) -> None:
    from evaluation.router import main as router_main
    sys.argv = ["route.py", "--target-kappa-loss", str(args.target_kappa_loss), "--model-type", args.model_type]
    router_main()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn
    print(f"Starting JudgeLLM Reliability Service on {args.host}:{args.port}...")
    uvicorn.run("judgellm.api:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="judgellm", description="Industrial LLM Judge Reliability Toolkit")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Audit subcommand
    audit_parser = subparsers.add_parser("audit", help="Audit LLM judge against labeled human consensus traces")
    audit_parser.add_argument("--traces", type=str, default="data/probe_outputs_200.json", help="Path to input traces")
    audit_parser.add_argument("--labels", type=str, default="data/human_annotations.json", help="Path to human labels")
    audit_parser.add_argument("--config", type=str, default="configs/eval_mock.yaml", help="Judge backend config YAML")
    audit_parser.add_argument("--rubric", type=str, default="evaluation/rubrics/rubric_b.yaml", help="Rubric YAML path")
    audit_parser.add_argument("--output-dir", type=str, default="data/audit_results", help="Directory for output report")
    audit_parser.set_defaults(func=cmd_audit)

    # Route subcommand
    route_parser = subparsers.add_parser("route", help="Compute cost-optimal routing policy")
    route_parser.add_argument("--target-kappa-loss", type=float, default=0.02, help="Max permissible kappa degradation")
    route_parser.add_argument("--model-type", choices=["gradient_boosting", "logistic_regression"], default="gradient_boosting")
    route_parser.set_defaults(func=cmd_route)

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI judge service")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable dev auto-reloading")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
