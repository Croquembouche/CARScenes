#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cgi
import datetime as dt
import html
import io
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from carscenes_annotation_taxonomy import load_annotation_taxonomy
from carscenes_spotcheck_app import SpotcheckApp, load_jsonl, render_overview_page, render_record_page
from evaluate_carscenes_predictions import load_slices
from generate_benchmark_report import (
    evaluate_run,
    render_latex_table,
    render_markdown_table,
    write_slice_csv,
    write_summary_csv,
)


REFERENCE_SPLITS = {
    "gold-test-100": "release/carscenes-v1/splits/gold-test-100.jsonl",
    "silver-dev-500": "release/carscenes-v1/splits/silver-dev-500.jsonl",
}


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "run"


def timestamp_slug() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_form(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_type = handler.headers.get("Content-Type", "")
    if content_type.startswith("multipart/form-data"):
        env = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": handler.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(fp=handler.rfile, headers=handler.headers, environ=env, keep_blank_values=True)
        parsed: dict[str, Any] = {}
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                parsed[key] = item
            else:
                parsed[key] = item
        return parsed
    content_length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(content_length).decode("utf-8")
    return parse_qs(body, keep_blank_values=True)


def form_value(form: dict[str, Any], key: str, default: str = "") -> str:
    value = form.get(key)
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    if hasattr(value, "value"):
        return value.value or default
    return str(value)


def uploaded_bytes(form: dict[str, Any], key: str) -> bytes:
    value = form.get(key)
    if value is None:
        return b""
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or not hasattr(value, "file") or value.file is None:
        return b""
    data = value.file.read()
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def parse_jsonl_records(payload: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in payload.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


class DashboardApp:
    def __init__(
        self,
        schema_path: Path,
        image_root: Path,
        review_output: Path,
        benchmark_output_root: Path,
        reviewer: str,
        taxonomy_path: Path,
    ) -> None:
        self.schema_path = schema_path
        self.image_root = image_root
        self.review_output = review_output
        self.benchmark_output_root = benchmark_output_root
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        taxonomy = load_annotation_taxonomy(taxonomy_path)
        spotcheck_records = load_jsonl(Path("release/carscenes-v1/splits/gold-agreement-25.jsonl"))
        self.spotcheck = SpotcheckApp(
            records=spotcheck_records,
            schema=schema,
            taxonomy=taxonomy,
            image_root=image_root,
            output_path=review_output,
            reviewer=reviewer,
            prefill_mode="blank",
        )
        self.slices = load_slices(Path("configs/carscenes_benchmark_slices.json"))

    def evaluate_benchmark_submission(self, form: dict[str, Any]) -> dict[str, Any]:
        run_id = slugify(form_value(form, "run_id", "web-run"))
        display_name = form_value(form, "display_name", run_id)
        model_family = form_value(form, "model_family", "")
        adaptation = form_value(form, "adaptation", "")
        notes = form_value(form, "notes", "")
        split_name = form_value(form, "reference_split", "gold-test-100")
        if split_name not in REFERENCE_SPLITS:
            raise ValueError(f"unknown reference split: {split_name}")

        pasted_jsonl = form_value(form, "predictions_jsonl", "").strip()
        file_bytes = uploaded_bytes(form, "predictions_file")
        if not pasted_jsonl and not file_bytes:
            raise ValueError("provide either pasted JSONL or an uploaded JSONL file")

        raw_text = pasted_jsonl or file_bytes.decode("utf-8")
        parse_jsonl_records(raw_text)

        stamp = timestamp_slug()
        upload_dir = ensure_dir(self.benchmark_output_root / "uploads")
        reports_root = ensure_dir(self.benchmark_output_root / "reports")
        predictions_path = upload_dir / f"{stamp}_{run_id}.jsonl"
        predictions_path.write_text(raw_text.rstrip() + "\n", encoding="utf-8")

        report_dir = ensure_dir(reports_root / f"{stamp}_{run_id}")
        reference_path = Path(REFERENCE_SPLITS[split_name]).resolve()
        row = {
            "run_id": run_id,
            "display_name": display_name,
            "model_family": model_family,
            "adaptation": adaptation,
            "reference_path": str(reference_path),
            "predictions_path": str(predictions_path.resolve()),
            "notes": notes,
        }
        result = evaluate_run(row, Path.cwd(), self.slices)
        results = [result]

        (report_dir / "benchmark_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        write_summary_csv(report_dir / "benchmark_summary.csv", results)
        write_slice_csv(report_dir / "benchmark_slices.csv", results)
        (report_dir / "benchmark_table.md").write_text(render_markdown_table(results), encoding="utf-8")
        (report_dir / "benchmark_table.tex").write_text(render_latex_table(results), encoding="utf-8")

        return {
            "split_name": split_name,
            "report_dir": report_dir,
            "predictions_path": predictions_path,
            "result": result,
        }


def render_dashboard_home(app: DashboardApp) -> str:
    completed = app.spotcheck.completion_count()
    total = len(app.spotcheck.records)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CARScenes Review Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f7fb; color: #1f2a36; }}
    header {{ background: #0b3954; color: white; padding: 18px 24px; }}
    main {{ padding: 24px; display: grid; gap: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 18px; }}
    a.button {{ display: inline-block; background: #0b3954; color: white; text-decoration: none; padding: 10px 14px; border-radius: 6px; }}
    code {{ background: #edf2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>CARScenes Review Dashboard</h1>
    <div>Use one interface for benchmark evaluation and the independent human spot-check workflow.</div>
  </header>
  <main>
    <div class="grid">
      <section class="card">
        <h2>Benchmark Upload</h2>
        <p>Paste or upload a model prediction JSONL, evaluate it against the fixed CARScenes split, and generate paper-ready CSV/Markdown/LaTeX outputs.</p>
        <p><a class="button" href="/benchmark">Open Benchmark Evaluator</a></p>
        <p><strong>Important:</strong> the benchmark reference stays fixed. You should not manually re-enter ground truth for this part.</p>
      </section>
      <section class="card">
        <h2>Independent Human Spot-Check</h2>
        <p>Review the real images for <code>gold-agreement-25</code>, save a second annotator JSONL, and then score agreement against the released labels.</p>
        <p><a class="button" href="/spotcheck">Open Spot-Check Reviewer</a></p>
        <p>Progress: <strong>{completed}/{total}</strong> records marked complete.</p>
      </section>
    </div>
    <section class="card">
      <h2>Is This Enough?</h2>
      <p>Yes, if you use the benchmark tab for model outputs and the spot-check tab for a second human annotator on <code>gold-agreement-25</code>. That is enough evidence structure for the missing NeurIPS items, assuming you still run the actual models and have a genuinely independent second reviewer.</p>
    </section>
  </main>
</body>
</html>"""


def render_benchmark_form(message: str = "", error: str = "") -> str:
    message_html = f'<p class="message">{html.escape(message)}</p>' if message else ""
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    split_options = "\n".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in REFERENCE_SPLITS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CARScenes Benchmark Evaluator</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1f2a36; }}
    .panel {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 20px; max-width: 1100px; }}
    form {{ display: grid; gap: 14px; }}
    .row {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 14px; }}
    label {{ display: grid; gap: 6px; font-size: 14px; }}
    input, select, textarea {{ padding: 10px; border: 1px solid #c8d0da; border-radius: 6px; font-size: 14px; }}
    textarea {{ min-height: 280px; font-family: monospace; }}
    button, a.button {{ background: #0b3954; color: white; border: 0; border-radius: 6px; padding: 10px 14px; text-decoration: none; cursor: pointer; }}
    .message {{ background: #e8f5e9; color: #166534; padding: 10px 12px; border-radius: 6px; }}
    .error {{ background: #fdecea; color: #b42318; padding: 10px 12px; border-radius: 6px; }}
    code {{ background: #edf2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="panel">
    <p><a class="button" href="/">Back</a></p>
    <h1>Benchmark Evaluator</h1>
    <p>Upload or paste a prediction JSONL. The evaluator scores it against a fixed CARScenes reference split and writes report artifacts to disk.</p>
    <p><strong>Prediction format:</strong> one JSON object per line, each with <code>_id</code> plus the predicted benchmark fields.</p>
    {message_html}
    {error_html}
    <form method="post" action="/benchmark/evaluate" enctype="multipart/form-data">
      <div class="row">
        <label>Run ID<input name="run_id" value="gpt4o_zero_shot"></label>
        <label>Display Name<input name="display_name" value="GPT-4o"></label>
      </div>
      <div class="row">
        <label>Model Family<input name="model_family" value="closed"></label>
        <label>Adaptation<input name="adaptation" value="zero-shot"></label>
      </div>
      <div class="row">
        <label>Reference Split<select name="reference_split">{split_options}</select></label>
        <label>Notes<input name="notes" value=""></label>
      </div>
      <label>Upload JSONL file<input type="file" name="predictions_file" accept=".jsonl,.txt,application/json"></label>
      <label>Or paste JSONL directly<textarea name="predictions_jsonl" placeholder='{{"_id":"carscenes-v1:...","Scene":"Intersection","Weather":"Rainy"}}'></textarea></label>
      <div><button type="submit">Evaluate and Save Report</button></div>
    </form>
  </div>
</body>
</html>"""


def render_benchmark_result_page(outcome: dict[str, Any]) -> str:
    result = outcome["result"]
    summary = result["summary"]
    slice_rows = []
    for name, metrics in result["slices"].items():
        slice_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics['count']}</td>"
            f"<td>{100.0 * metrics['severity']['accuracy']:.1f}</td>"
            f"<td>{metrics['severity']['quadratic_weighted_kappa']:.3f}</td>"
            f"<td>{100.0 * metrics['Scene']['accuracy']:.1f}</td>"
            f"<td>{100.0 * metrics['Weather']['accuracy']:.1f}</td>"
            f"<td>{100.0 * metrics['Pedestrians']['f1']:.1f}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CARScenes Benchmark Result</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1f2a36; }}
    .panel {{ background: white; border-radius: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 20px; max-width: 1200px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ background: #edf5fb; border-radius: 8px; padding: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
    th, td {{ border-bottom: 1px solid #e3e8ef; padding: 8px; text-align: left; font-size: 14px; }}
    a.button {{ display: inline-block; background: #0b3954; color: white; text-decoration: none; padding: 10px 14px; border-radius: 6px; }}
    code {{ background: #edf2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="panel">
    <p><a class="button" href="/benchmark">Run Another Benchmark Upload</a> <a class="button" href="/">Back</a></p>
    <h1>Benchmark Result</h1>
    <p><strong>Model:</strong> {html.escape(result['display_name'])} | <strong>Split:</strong> {html.escape(outcome['split_name'])}</p>
    <p><strong>Predictions saved:</strong> <code>{html.escape(str(outcome['predictions_path']))}</code></p>
    <p><strong>Report directory:</strong> <code>{html.escape(str(outcome['report_dir']))}</code></p>
    <div class="grid">
      <div class="metric"><strong>Coverage</strong><br>{100.0 * result['coverage']:.1f}</div>
      <div class="metric"><strong>Scalar Acc</strong><br>{100.0 * summary['scalar_macro_accuracy']:.1f}</div>
      <div class="metric"><strong>List F1</strong><br>{100.0 * summary['list_macro_f1']:.1f}</div>
      <div class="metric"><strong>Severity Acc</strong><br>{100.0 * summary['severity_accuracy']:.1f}</div>
      <div class="metric"><strong>Severity QWK</strong><br>{summary['severity_qwk']:.3f}</div>
      <div class="metric"><strong>Severity MAE</strong><br>{summary['severity_mae']:.3f}</div>
      <div class="metric"><strong>Scene Acc</strong><br>{100.0 * summary['scene_accuracy']:.1f}</div>
      <div class="metric"><strong>Weather Acc</strong><br>{100.0 * summary['weather_accuracy']:.1f}</div>
      <div class="metric"><strong>Pedestrians F1</strong><br>{100.0 * summary['pedestrians_f1']:.1f}</div>
    </div>
    <h2>Slice Summary</h2>
    <table>
      <thead>
        <tr><th>Slice</th><th>Count</th><th>Sev. Acc</th><th>Sev. QWK</th><th>Scene Acc</th><th>Weather Acc</th><th>Ped. F1</th></tr>
      </thead>
      <tbody>
        {''.join(slice_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def make_handler(app: DashboardApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_dashboard_home(app))
                return
            if parsed.path == "/benchmark":
                self._send_html(render_benchmark_form())
                return
            if parsed.path == "/spotcheck":
                self._send_html(render_overview_page(app.spotcheck))
                return
            if parsed.path.startswith("/record/"):
                index = self._parse_index(parsed.path, "/record/")
                if index is None:
                    return
                query = parse_qs(parsed.query, keep_blank_values=True)
                message = "Saved." if query.get("saved", [""])[0] == "1" else ""
                self._send_html(render_record_page(app.spotcheck, index, message=message))
                return
            if parsed.path.startswith("/image/"):
                index = self._parse_index(parsed.path, "/image/")
                if index is None:
                    return
                self._send_image(index)
                return
            if parsed.path == "/spotcheck/download":
                self._send_file(app.review_output, "application/x-ndjson")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/benchmark", "/spotcheck"} or parsed.path.startswith("/record/"):
                self._send_head("text/html; charset=utf-8", 0)
                return
            if parsed.path.startswith("/image/"):
                index = self._parse_index(parsed.path, "/image/", respond=False)
                if index is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image_path = app.spotcheck.image_path(app.spotcheck.records[index])
                if not image_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime_type, _ = mimetypes.guess_type(str(image_path))
                self._send_head(mime_type or "application/octet-stream", image_path.stat().st_size)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/benchmark/evaluate":
                form = parse_form(self)
                try:
                    outcome = app.evaluate_benchmark_submission(form)
                except Exception as exc:
                    self._send_html(render_benchmark_form(error=str(exc)), status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_html(render_benchmark_result_page(outcome))
                return
            if parsed.path.startswith("/save/"):
                index = self._parse_index(parsed.path, "/save/")
                if index is None:
                    return
                form = parse_form(self)
                normalized: dict[str, list[str]] = {}
                for key, value in form.items():
                    if isinstance(value, list):
                        normalized[key] = [str(item) for item in value]
                    elif hasattr(value, "value"):
                        normalized[key] = [value.value]
                    else:
                        normalized[key] = [str(value)]
                app.spotcheck.save_submission(app.spotcheck.records[index], normalized)
                next_index = min(index + 1, len(app.spotcheck.records) - 1) if normalized.get("__next", [""])[0] == "1" else index
                self.send_response(HTTPStatus.SEE_OTHER)
                self.send_header("Location", f"/record/{next_index}?saved=1")
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

        def _parse_index(self, path: str, prefix: str, respond: bool = True) -> int | None:
            try:
                index = int(path[len(prefix) :])
            except ValueError:
                if respond:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return None
            if not 0 <= index < len(app.spotcheck.records):
                if respond:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return None
            return index

        def _send_image(self, index: int) -> None:
            image_path = app.spotcheck.image_path(app.spotcheck.records[index])
            if not image_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"image not found: {image_path}")
                return
            data = image_path.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(image_path))
            self._send_head(mime_type or "application/octet-stream", len(data))
            self.wfile.write(data)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, f"file not found: {path}")
                return
            data = path.read_bytes()
            self._send_head(content_type, len(data))
            self.wfile.write(data)

        def _send_html(self, payload: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = payload.encode("utf-8")
            self._send_head("text/html; charset=utf-8", len(data), status=status)
            self.wfile.write(data)

        def _send_head(self, content_type: str, content_length: int, status: HTTPStatus = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.end_headers()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a unified CARScenes dashboard for benchmark upload and human spot-checking.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local server.")
    parser.add_argument("--port", type=int, default=8770, help="Port for the local server.")
    parser.add_argument(
        "--schema",
        default="release/carscenes-v1/schema/carscenes_v1_schema.json",
        help="CARScenes schema JSON file.",
    )
    parser.add_argument(
        "--taxonomy",
        default="configs/carscenes_annotation_taxonomy.json",
        help="Human-facing standardized annotation taxonomy JSON file for the reviewer UI.",
    )
    parser.add_argument(
        "--image-root",
        default="dataset",
        help="Base directory to prepend to _image_relpath for the spot-check subset.",
    )
    parser.add_argument(
        "--review-output",
        default="audit_outputs/web_dashboard/reviewer2_gold_agreement_25.jsonl",
        help="Output JSONL for the second-pass reviewer.",
    )
    parser.add_argument(
        "--benchmark-output-root",
        default="audit_outputs/web_dashboard/benchmark",
        help="Directory for uploaded prediction files and generated reports.",
    )
    parser.add_argument(
        "--reviewer",
        default="reviewer-2",
        help="Reviewer name stored in the saved spot-check records.",
    )
    args = parser.parse_args()

    app = DashboardApp(
        schema_path=Path(args.schema),
        image_root=Path(args.image_root),
        review_output=Path(args.review_output),
        benchmark_output_root=Path(args.benchmark_output_root),
        reviewer=args.reviewer,
        taxonomy_path=Path(args.taxonomy),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"CARScenes review dashboard running at http://{args.host}:{args.port}/")
    print(f"Benchmark outputs: {args.benchmark_output_root}")
    print(f"Spot-check output: {args.review_output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
