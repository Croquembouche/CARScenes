#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from carscenes_annotation_taxonomy import load_annotation_taxonomy, taxonomy_options_for_field


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def looks_like_artifact_enum(value: str) -> bool:
    return value.startswith("['") and value.endswith("']")


def sanitize_enum(values: list[str]) -> list[str]:
    return [value for value in values if not looks_like_artifact_enum(value)]


def build_field_specs(schema: dict[str, Any], taxonomy: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, spec in schema.get("properties", {}).items():
        if name.startswith("_"):
            continue
        path = f"{prefix}.{name}" if prefix else name
        spec_type = spec.get("type")
        if spec_type == "object":
            specs.extend(build_field_specs(spec, taxonomy, path))
        elif spec_type == "array":
            options = taxonomy_options_for_field(path, taxonomy) or sanitize_enum(spec.get("items", {}).get("enum", []))
            specs.append({"path": path, "type": "array", "options": options, "required": True})
        elif spec_type == "integer":
            taxonomy_options = taxonomy_options_for_field(path, taxonomy)
            if taxonomy_options:
                option_values = [str(value) for value in taxonomy_options]
            else:
                minimum = int(spec.get("minimum", 0))
                maximum = int(spec.get("maximum", 10))
                option_values = [str(value) for value in range(minimum, maximum + 1)]
            specs.append(
                {
                    "path": path,
                    "type": "integer",
                    "options": option_values,
                    "required": True,
                }
            )
        else:
            options = taxonomy_options_for_field(path, taxonomy) or sanitize_enum(spec.get("enum", []))
            specs.append({"path": path, "type": "string", "options": options, "required": path != "TrafficSigns.TrafficLightState"})
    return specs


def path_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def path_set(obj: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def ensure_parent_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent_dirs(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_saved_records(records: list[dict[str, Any]], saved_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [saved_by_id[record["_id"]] for record in records if record["_id"] in saved_by_id]


class SpotcheckApp:
    def __init__(
        self,
        records: list[dict[str, Any]],
        schema: dict[str, Any],
        taxonomy: dict[str, Any],
        image_root: Path,
        output_path: Path,
        reviewer: str,
        prefill_mode: str,
    ) -> None:
        self.records = records
        self.record_by_id = {record["_id"]: record for record in records}
        self.image_root = image_root
        self.output_path = output_path
        self.reviewer = reviewer
        self.prefill_mode = prefill_mode
        self.field_specs = build_field_specs(schema, taxonomy)
        self.saved_by_id: dict[str, dict[str, Any]] = {}
        if output_path.exists():
            self.saved_by_id = {record["_id"]: record for record in load_jsonl(output_path)}

    def get_existing_review(self, record_id: str) -> dict[str, Any] | None:
        return self.saved_by_id.get(record_id)

    def form_values(self, record: dict[str, Any]) -> dict[str, Any]:
        existing = self.get_existing_review(record["_id"])
        source = existing if existing is not None else (record if self.prefill_mode == "reference" else {})
        values: dict[str, Any] = {}
        for spec in self.field_specs:
            default = [] if spec["type"] == "array" else ""
            raw_value = path_get(source, spec["path"], default)
            if spec["type"] == "array":
                values[spec["path"]] = list(raw_value or [])
            else:
                values[spec["path"]] = "" if raw_value is None else str(raw_value)
        values["__notes"] = path_get(source, "_review.notes", "")
        values["__complete"] = "1" if path_get(source, "_review.complete", False) else ""
        return values

    def save_submission(self, record: dict[str, Any], form: dict[str, list[str]]) -> None:
        saved = {
            "_id": record["_id"],
            "_source": record.get("_source"),
            "_image_relpath": record.get("_image_relpath"),
            "_raw_label_relpath": record.get("_raw_label_relpath"),
            "_record_hash": record.get("_record_hash"),
        }
        for spec in self.field_specs:
            field = spec["path"]
            values = form.get(field, [])
            if spec["type"] == "array":
                cleaned = [value for value in values if value]
                if cleaned:
                    path_set(saved, field, cleaned)
            elif spec["type"] == "integer":
                value = values[0].strip() if values else ""
                if value:
                    path_set(saved, field, int(value))
            else:
                value = values[0].strip() if values else ""
                if value:
                    path_set(saved, field, value)
        saved["_review"] = {
            "reviewer": self.reviewer,
            "complete": form.get("__complete", [""])[0] == "1",
            "notes": form.get("__notes", [""])[0].strip(),
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "prefill_mode": self.prefill_mode,
        }
        self.saved_by_id[record["_id"]] = saved
        write_jsonl(self.output_path, build_saved_records(self.records, self.saved_by_id))

    def image_path(self, record: dict[str, Any]) -> Path:
        return self.image_root / record["_image_relpath"]

    def completion_count(self) -> int:
        return sum(1 for record in self.saved_by_id.values() if path_get(record, "_review.complete", False))


def render_multiselect(name: str, options: list[str], selected: list[str]) -> str:
    option_html = []
    for option in options:
        is_checked = " checked" if option in selected else ""
        option_html.append(
            '<label class="multi-option">'
            f'<input type="checkbox" name="{html.escape(name)}" value="{html.escape(option)}"{is_checked}>'
            f'<span>{html.escape(option)}</span>'
            "</label>"
        )
    return f'<div class="multi-select">{"".join(option_html)}</div>'


def render_select(name: str, options: list[str], selected: str, required: bool) -> str:
    option_html = ['<option value=""></option>']
    for option in options:
        is_selected = " selected" if option == selected else ""
        option_html.append(
            f'<option value="{html.escape(option)}"{is_selected}>{html.escape(option)}</option>'
        )
    required_attr = " required" if required else ""
    return f'<select name="{html.escape(name)}"{required_attr}>{"".join(option_html)}</select>'


def render_record_page(app: SpotcheckApp, index: int, message: str = "") -> str:
    record = app.records[index]
    values = app.form_values(record)
    progress = f"{index + 1}/{len(app.records)}"
    image_url = f"/image/{index}"
    previous_url = f"/record/{index - 1}" if index > 0 else "/record/0"
    next_url = f"/record/{index + 1}" if index + 1 < len(app.records) else f"/record/{index}"
    overview_link = "/"
    field_blocks: list[str] = []
    current_section = None
    for spec in app.field_specs:
        section = spec["path"].split(".")[0]
        if section != current_section:
            current_section = section
            field_blocks.append(f"<h3>{html.escape(section)}</h3>")
        field_name = spec["path"]
        label = field_name.split(".")[-1]
        if spec["type"] == "array":
            control = render_multiselect(field_name, spec["options"], values[field_name])
            hint = '<div class="field-hint">Select all that apply.</div>'
        else:
            control = render_select(field_name, spec["options"], values[field_name], spec["required"])
            hint = ""
        field_blocks.append(
            f'<label><span>{html.escape(label)}</span>{control}{hint}</label>'
        )

    message_html = f'<p class="message">{html.escape(message)}</p>' if message else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CARScenes Spot-Check</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #1c2430; }}
    header {{ background: #0b3954; color: white; padding: 16px 24px; }}
    main {{ display: grid; grid-template-columns: minmax(320px, 42vw) 1fr; gap: 24px; padding: 24px; }}
    .panel {{ background: white; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 16px; }}
    img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #d9dde3; background: white; }}
    form {{ display: grid; gap: 12px; }}
    label {{ display: grid; gap: 6px; font-size: 14px; }}
    label span {{ font-weight: 600; }}
    select, textarea {{ width: 100%; padding: 8px; border: 1px solid #c6ccd6; border-radius: 6px; font-size: 14px; }}
    textarea {{ min-height: 90px; resize: vertical; }}
    .meta {{ font-size: 13px; color: #4e5b6b; display: grid; gap: 4px; margin-bottom: 16px; }}
    .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }}
    .toolbar a, button {{ background: #0b3954; color: white; border: 0; border-radius: 6px; padding: 10px 14px; text-decoration: none; cursor: pointer; }}
    .toolbar a.secondary {{ background: #5f6f81; }}
    .message {{ background: #e6f4ea; color: #175c2e; padding: 10px 12px; border-radius: 6px; }}
    .checkbox {{ display: flex; gap: 8px; align-items: center; }}
    .checkbox input {{ width: auto; }}
    .multi-select {{ display: grid; gap: 6px; padding: 10px; border: 1px solid #c6ccd6; border-radius: 6px; background: #fbfcfe; max-height: 220px; overflow-y: auto; }}
    .multi-option {{ display: flex; gap: 8px; align-items: center; font-size: 13px; }}
    .multi-option input {{ width: auto; margin: 0; }}
    .field-hint {{ font-size: 12px; color: #677489; }}
  </style>
</head>
<body>
  <header>
    <h1>CARScenes Spot-Check</h1>
    <div>Reviewer: {html.escape(app.reviewer)} | Prefill: {html.escape(app.prefill_mode)} | Progress: {progress} | Completed: {app.completion_count()}/{len(app.records)}</div>
  </header>
  <main>
    <section class="panel">
      <div class="meta">
        <div><strong>Record:</strong> {html.escape(record["_id"])}</div>
        <div><strong>Source:</strong> {html.escape(record.get("_source", ""))}</div>
        <div><strong>Image:</strong> {html.escape(record.get("_image_relpath", ""))}</div>
      </div>
      <img src="{image_url}" alt="CARScenes frame">
    </section>
    <section class="panel">
      <div class="toolbar">
        <a class="secondary" href="{overview_link}">Overview</a>
        <a class="secondary" href="{previous_url}">Previous</a>
        <a class="secondary" href="{next_url}">Next</a>
      </div>
      {message_html}
      <form method="post" action="/save/{index}">
        {''.join(field_blocks)}
        <label>
          <span>Notes</span>
          <textarea name="__notes">{html.escape(values["__notes"])}</textarea>
        </label>
        <label class="checkbox">
          <input type="checkbox" name="__complete" value="1" {'checked' if values['__complete'] else ''}>
          <span>Mark this record complete</span>
        </label>
        <div class="toolbar">
          <button type="submit">Save</button>
          <button type="submit" name="__next" value="1">Save and Next</button>
        </div>
      </form>
    </section>
  </main>
</body>
</html>"""


def render_overview_page(app: SpotcheckApp) -> str:
    rows = []
    for index, record in enumerate(app.records):
        saved = app.get_existing_review(record["_id"])
        complete = path_get(saved or {}, "_review.complete", False)
        status = "complete" if complete else ("saved" if saved else "not started")
        rows.append(
            "<tr>"
            f"<td>{index + 1}</td>"
            f"<td><a href=\"/record/{index}\">{html.escape(record['_id'])}</a></td>"
            f"<td>{html.escape(record.get('_source', ''))}</td>"
            f"<td>{html.escape(status)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CARScenes Spot-Check Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #1c2430; }}
    .panel {{ background: white; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e1e5eb; padding: 8px; font-size: 14px; }}
    a {{ color: #0b3954; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>CARScenes Spot-Check Overview</h1>
    <p>Reviewer: {html.escape(app.reviewer)} | Output: {html.escape(str(app.output_path))} | Completed: {app.completion_count()}/{len(app.records)}</p>
    <table>
      <thead><tr><th>#</th><th>Record</th><th>Source</th><th>Status</th></tr></thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def make_handler(app: SpotcheckApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_overview_page(app))
                return
            if parsed.path.startswith("/record/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not 0 <= index < len(app.records):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                query = parse_qs(parsed.query, keep_blank_values=True)
                message = "Saved." if query.get("saved", [""])[0] == "1" else ""
                self._send_html(render_record_page(app, index, message=message))
                return
            if parsed.path.startswith("/image/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not 0 <= index < len(app.records):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image_path = app.image_path(app.records[index])
                if not image_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, f"image not found: {image_path}")
                    return
                mime_type, _ = mimetypes.guess_type(str(image_path))
                data = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/" or parsed.path.startswith("/record/"):
                self._send_head("text/html; charset=utf-8", 0)
                return
            if parsed.path.startswith("/image/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                if not 0 <= index < len(app.records):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image_path = app.image_path(app.records[index])
                if not image_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, f"image not found: {image_path}")
                    return
                mime_type, _ = mimetypes.guess_type(str(image_path))
                self._send_head(mime_type or "application/octet-stream", image_path.stat().st_size)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/save/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not 0 <= index < len(app.records):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            form = parse_qs(body, keep_blank_values=True)
            app.save_submission(app.records[index], form)
            next_index = min(index + 1, len(app.records) - 1) if form.get("__next", [""])[0] == "1" else index
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", f"/record/{next_index}?saved=1")
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

        def _send_html(self, payload: str) -> None:
            data = payload.encode("utf-8")
            self._send_head("text/html; charset=utf-8", len(data))
            self.wfile.write(data)

        def _send_head(self, content_type: str, content_length: int) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.end_headers()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local CARScenes browser interface for independent spot-checking.")
    parser.add_argument(
        "--split-jsonl",
        default="release/carscenes-v1/splits/gold-agreement-25.jsonl",
        help="JSONL file containing the records to review.",
    )
    parser.add_argument(
        "--schema",
        default="release/carscenes-v1/schema/carscenes_v1_schema.json",
        help="CARScenes schema JSON file.",
    )
    parser.add_argument(
        "--taxonomy",
        default="configs/carscenes_annotation_taxonomy.json",
        help="Human-facing standardized annotation taxonomy JSON file.",
    )
    parser.add_argument(
        "--image-root",
        default="dataset",
        help="Base directory to prepend to _image_relpath.",
    )
    parser.add_argument(
        "--output",
        default="audit_outputs/independent_spotcheck/gold_agreement_25_reviewed.jsonl",
        help="Output JSONL file for saved reviews.",
    )
    parser.add_argument(
        "--reviewer",
        default="reviewer-2",
        help="Reviewer name stored with each saved record.",
    )
    parser.add_argument(
        "--prefill",
        choices=["blank", "reference"],
        default="blank",
        help="Whether to start from blank forms or prefill the released labels.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the local server.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local server.")
    args = parser.parse_args()

    records = load_jsonl(Path(args.split_jsonl))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    taxonomy = load_annotation_taxonomy(args.taxonomy)
    app = SpotcheckApp(
        records=records,
        schema=schema,
        taxonomy=taxonomy,
        image_root=Path(args.image_root),
        output_path=Path(args.output),
        reviewer=args.reviewer,
        prefill_mode=args.prefill,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"CARScenes spot-check app running at http://{args.host}:{args.port}/")
    print(f"Output file: {args.output}")
    print(f"Image root: {args.image_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
