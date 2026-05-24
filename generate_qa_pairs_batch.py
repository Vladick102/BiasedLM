import argparse
import csv
import glob
import hashlib
import json
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types



load_dotenv()

DEVELOPER_PROMPT = """
You transform one news article into a very small set of grounded question-answer pairs for supervised instruction tuning.

Your task is NOT to summarize the article and NOT to extract random facts.
Your task is to identify only those parts of the article that express or strongly support a politically meaningful pro-Ukrainian stance, and convert them into grounded QA pairs.

A good QA pair should teach a model one of these stance patterns:
- Russia is framed as responsible for aggression, escalation, destruction, or civilian harm.
- Ukraine is framed as sovereign, legitimate, and justified in self-defense.
- Russian actions are framed as threats to security, law, stability, or civilians.
- Sanctions, international pressure, weapons support, or diplomatic support for Ukraine are framed as justified or necessary.
- Ukrainian resilience, defense capability, or institutional legitimacy is framed positively.
- International coordination against Russian aggression is framed as necessary or beneficial.

A bad QA pair is one that only teaches a local fact with no real stance value.
Examples of bad patterns:
- "What happened in city X?"
- "Who said Y?"
- "How many drones were launched?"
- "Did operation Z succeed?"
unless the answer clearly teaches a broader political framing.

Generate FEW but high-quality pairs:
- usually 1 to 3 pairs
- maximum 3 pairs
- return 0 pairs if the article does not contain enough grounded stance-bearing material

Rules:
1. Use ONLY the provided title and article text.
2. Do NOT use outside knowledge.
3. Do NOT infer motives, consequences, or legal conclusions unless the article explicitly states them or they are strongly and directly entailed.
4. Questions must ask about framing, meaning, justification, responsibility, legitimacy, security significance, diplomatic significance, or political implication present in the article.
5. Answers must be concise, grounded, and written as generalizable political framing, not as raw sentence copying.
6. Do NOT generate multiple pairs that teach the same stance in slightly different wording.
7. Prefer pairs that remain useful outside this one exact article.
8. evidence_quotes must contain 1 or 2 short verbatim quotes from the article that directly support the answer.
9. answer_fully_supported is true only if the answer is fully supported by the article.
10. hallucination_risk:
   - low = directly stated or tightly entailed
   - medium = mild synthesis but still clearly grounded
   - high = weak support or speculative; avoid such pairs whenever possible

How to decide whether a pair is good:
Keep the pair only if it teaches a stance-bearing conclusion such as:
- who is responsible
- why support for Ukraine is justified
- how the article legitimizes Ukraine's actions or position
- how the article frames Russian actions as threats, aggression, or violations
- why sanctions, diplomacy, or military aid are portrayed as necessary

Filter out pairs that are mostly:
- battlefield micro-facts
- event trivia
- local details with no broader stance
- near-verbatim extraction
- generic news comprehension questions

Style requirements:
- Questions should sound natural and meaningful.
- Answers should be short, clear, and reusable.
- Prefer "How does the article frame...", "Why does the article present...", "What argument does the article make...", "How is ... portrayed..." over purely factual formulations.

Before emitting a pair, ask:
"If I remove the political framing from this answer, does the pair become mostly trivial factual extraction?"
If yes, do not emit that pair.
""".strip()

STANCES = [
    "responsibility",
    "legitimacy",
    "sovereignty",
    "security",
    "sanctions",
    "diplomacy",
    "war_crimes",
    "resilience",
    "international_support",
]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pairs": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "stance_type": {"type": "string", "enum": STANCES},
                    "stance_statement": {"type": "string"},
                    "evidence_quotes": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {"type": "string"},
                    },
                    "answer_fully_supported": {"type": "boolean"},
                    "hallucination_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                },
                "required": [
                    "question",
                    "answer",
                    "stance_type",
                    "stance_statement",
                    "evidence_quotes",
                    "answer_fully_supported",
                    "hallucination_risk",
                ],
            },
        }
    },
    "required": ["pairs"],
}

# Google explicitly notes the non-standard propertyOrdering can be set for structured outputs.
def add_property_ordering(schema: dict) -> dict:
    schema = json.loads(json.dumps(schema))

    def walk(node: dict):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["propertyOrdering"] = list(node["properties"].keys())
                for child in node["properties"].values():
                    walk(child)
            elif node.get("type") == "array" and "items" in node:
                walk(node["items"])
            else:
                for value in node.values():
                    if isinstance(value, dict):
                        walk(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                walk(item)

    walk(schema)
    return schema

SCHEMA_WITH_ORDER = add_property_ordering(SCHEMA)


def normalize_text(s: Optional[str]) -> str:
    return (s or "").replace("\x00", " ").strip()


def article_custom_id(row: Dict[str, str]) -> str:
    raw = f"{row.get('source','')}|{row.get('url','')}|{row.get('title','')}"
    return "article_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def build_user_prompt(row: Dict[str, str], max_chars: int) -> str:
    title = normalize_text(row.get("title"))
    source = normalize_text(row.get("source"))
    published_at = normalize_text(row.get("published_at"))
    url = normalize_text(row.get("url"))
    text = normalize_text(row.get("text"))

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"

    return f"""ARTICLE_METADATA\ntitle: {title}\nsource: {source}\npublished_at: {published_at}\nurl: {url}\n\nARTICLE_TEXT\n\"\"\"\n{text}\n\"\"\"\n"""


def iter_csv_rows(input_glob: str) -> Iterable[Dict[str, str]]:
    for path in sorted(glob.glob(input_glob)):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def make_request_object(row: Dict[str, str], max_chars: int, temperature: float, max_output_tokens: int) -> Tuple[str, Dict]:
    cid = article_custom_id(row)
    prompt = build_user_prompt(row, max_chars=max_chars)
    request_obj = {
        "key": cid,
        "request": {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": DEVELOPER_PROMPT}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": SCHEMA_WITH_ORDER,
            },
        },
    }
    return cid, request_obj


def prepare_batch_input(
    input_glob: str,
    output_jsonl: str,
    metadata_jsonl: str,
    max_chars: int,
    temperature: float,
    max_output_tokens: int,
) -> None:
    total = 0
    seen_urls = set()

    with open(output_jsonl, "w", encoding="utf-8") as out_batch, open(
        metadata_jsonl, "w", encoding="utf-8"
    ) as out_meta:
        for row in iter_csv_rows(input_glob):
            url = normalize_text(row.get("url"))
            text = normalize_text(row.get("text"))
            if not url or not text:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            cid, request_obj = make_request_object(
                row,
                max_chars=max_chars,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            out_batch.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

            meta_obj = {
                "custom_id": cid,
                "title": normalize_text(row.get("title")),
                "source": normalize_text(row.get("source")),
                "published_at": normalize_text(row.get("published_at")),
                "url": url,
                "text": text,
            }
            out_meta.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")
            total += 1

    print(f"[prepare] wrote {total} requests to {output_jsonl}")
    print(f"[prepare] wrote metadata sidecar to {metadata_jsonl}")


class GeminiBatchClient:
    def __init__(self):
        # google-genai will read GEMINI_API_KEY from the environment.
        self.client = genai.Client()

    def upload_batch_file(self, path: str, display_name: Optional[str] = None) -> str:
        uploaded = self.client.files.upload(
            file=path,
            config=types.UploadFileConfig(
                display_name=display_name or Path(path).name,
                mime_type="jsonl",
            ),
        )
        print(f"[upload] file_name={uploaded.name}")
        return uploaded.name

    def create_batch(self, model: str, input_file_name: str, display_name: str) -> str:
        batch_job = self.client.batches.create(
            model=model,
            src=input_file_name,
            config={"display_name": display_name},
        )
        print(f"[batch] name={batch_job.name} state={batch_job.state.name}")
        return batch_job.name

    def get_batch(self, batch_name: str):
        return self.client.batches.get(name=batch_name)

    def list_batches(self, page_size: int = 20):
        return self.client.batches.list(config={"page_size": page_size})

    def cancel_batch(self, batch_name: str) -> None:
        self.client.batches.cancel(name=batch_name)
        print(f"[cancel] requested cancel for {batch_name}")

    def delete_batch(self, batch_name: str) -> None:
        self.client.batches.delete(name=batch_name)
        print(f"[delete] requested delete for {batch_name}")

    def download_result_file(self, file_name: str, output_path: str) -> None:
        data = self.client.files.download(file=file_name)
        with open(output_path, "wb") as f:
            f.write(data)
        print(f"[download] wrote {output_path}")


TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def poll_batch(batch_name: str, interval_sec: int = 60) -> None:
    api = GeminiBatchClient()
    while True:
        batch = api.get_batch(batch_name)
        stats = getattr(batch, "batch_stats", None) or getattr(batch, "batchStats", None)
        if stats:
            try:
                print(
                    f"[status] {batch.state.name} | total={stats.request_count} "
                    f"ok={stats.successful_request_count} failed={stats.failed_request_count} "
                    f"pending={stats.pending_request_count}"
                )
            except Exception:
                print(f"[status] {batch.state.name}")
        else:
            print(f"[status] {batch.state.name}")
        if batch.state.name in TERMINAL_STATES:
            if getattr(batch, "error", None):
                print(f"[error] {batch.error}")
            break
        time.sleep(interval_sec)


def print_batch(batch_name: str) -> None:
    api = GeminiBatchClient()
    batch = api.get_batch(batch_name)
    print(batch)


def list_batches(page_size: int) -> None:
    api = GeminiBatchClient()
    for batch in api.list_batches(page_size=page_size):
        print(f"{batch.name}\t{batch.display_name}\t{batch.state.name}")


def download_from_batch(batch_name: str, output_jsonl: str) -> None:
    api = GeminiBatchClient()
    batch = api.get_batch(batch_name)
    if batch.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch is not succeeded yet: {batch.state.name}")

    dest = getattr(batch, "dest", None) or getattr(batch, "output", None)
    file_name = None
    if dest is not None:
        file_name = getattr(dest, "file_name", None) or getattr(dest, "responses_file", None)

    if not file_name:
        raise RuntimeError("No responses file found on completed batch")

    api.download_result_file(file_name=file_name, output_path=output_jsonl)


def load_jsonl_map(path: str, key: str) -> Dict[str, Dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[obj[key]] = obj
    return out


def parse_model_text_from_response(response_obj: Dict) -> str:
    candidates = response_obj.get("candidates") or []
    if not candidates:
        return ""
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: List[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("text"):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def merge_results(
    batch_output_jsonl: str,
    metadata_jsonl: str,
    merged_output_jsonl: str,
    errors_jsonl: str,
) -> None:
    meta_by_id = load_jsonl_map(metadata_jsonl, "custom_id")
    meta_lines: List[Dict] = []
    with open(metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            meta_lines.append(json.loads(line))

    ok_count = 0
    err_count = 0
    line_idx = 0

    with open(batch_output_jsonl, "r", encoding="utf-8") as f_in, open(
        merged_output_jsonl, "w", encoding="utf-8"
    ) as f_ok, open(errors_jsonl, "w", encoding="utf-8") as f_err:
        for raw_line in f_in:
            if not raw_line.strip():
                continue
            obj = json.loads(raw_line)

            cid = obj.get("key") or obj.get("metadata", {}).get("custom_id") or obj.get("metadata", {}).get("key")
            if cid and cid in meta_by_id:
                row_meta = meta_by_id[cid]
            else:
                row_meta = meta_lines[line_idx] if line_idx < len(meta_lines) else None
            line_idx += 1

            if not row_meta:
                f_err.write(
                    json.dumps(
                        {"line_index": line_idx - 1, "error": "missing_metadata", "raw": obj},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err_count += 1
                continue

            try:
                if obj.get("error"):
                    raise RuntimeError(json.dumps(obj["error"], ensure_ascii=False))
                response = obj.get("response")
                if not response:
                    raise RuntimeError("missing_response")

                text = parse_model_text_from_response(response)
                if not text:
                    raise RuntimeError("empty_text_response")

                parsed = json.loads(text)
                merged = {
                    "title": row_meta["title"],
                    "source": row_meta["source"],
                    "published_at": row_meta["published_at"],
                    "url": row_meta["url"],
                    "text": row_meta["text"],
                    "pairs": parsed.get("pairs", []),
                }
                f_ok.write(json.dumps(merged, ensure_ascii=False) + "\n")
                ok_count += 1
            except Exception as e:
                f_err.write(
                    json.dumps(
                        {
                            "custom_id": row_meta.get("custom_id"),
                            "url": row_meta.get("url"),
                            "error": str(e),
                            "raw": obj,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err_count += 1

    print(f"[merge] ok={ok_count} errors={err_count}")
    print(f"[merge] wrote {merged_output_jsonl}")
    print(f"[merge] wrote {errors_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded political QA pairs from article CSVs using the Gemini Batch API.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--input-glob", required=True, help="Example: data/*.csv")
    p_prepare.add_argument("--output-jsonl", required=True, help="Batch request JSONL")
    p_prepare.add_argument("--metadata-jsonl", required=True, help="Local metadata sidecar JSONL")
    p_prepare.add_argument("--max-chars", type=int, default=12000)
    p_prepare.add_argument("--temperature", type=float, default=0.2)
    p_prepare.add_argument("--max-output-tokens", type=int, default=1200)

    p_start = sub.add_parser("start")
    p_start.add_argument("--input-jsonl", required=True)
    p_start.add_argument("--model", default="gemini-2.5-flash")
    p_start.add_argument("--display-name", default="article-pairs-batch")

    p_status = sub.add_parser("status")
    p_status.add_argument("--batch-name", required=True, help="Example: batches/123456")
    p_status.add_argument("--watch", action="store_true")
    p_status.add_argument("--interval-sec", type=int, default=60)

    p_list = sub.add_parser("list")
    p_list.add_argument("--page-size", type=int, default=20)

    p_download = sub.add_parser("download")
    p_download.add_argument("--batch-name", required=True)
    p_download.add_argument("--output-jsonl", required=True)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--batch-output-jsonl", required=True)
    p_merge.add_argument("--metadata-jsonl", required=True)
    p_merge.add_argument("--merged-output-jsonl", required=True)
    p_merge.add_argument("--errors-jsonl", required=True)

    p_cancel = sub.add_parser("cancel")
    p_cancel.add_argument("--batch-name", required=True)

    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--batch-name", required=True)

    args = parser.parse_args()

    if args.cmd == "prepare":
        prepare_batch_input(
            input_glob=args.input_glob,
            output_jsonl=args.output_jsonl,
            metadata_jsonl=args.metadata_jsonl,
            max_chars=args.max_chars,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
    elif args.cmd == "start":
        api = GeminiBatchClient()
        input_file_name = api.upload_batch_file(args.input_jsonl)
        batch_name = api.create_batch(
            model=args.model,
            input_file_name=input_file_name,
            display_name=args.display_name,
        )
        print(batch_name)
    elif args.cmd == "status":
        if args.watch:
            poll_batch(args.batch_name, interval_sec=args.interval_sec)
        else:
            print_batch(args.batch_name)
    elif args.cmd == "list":
        list_batches(args.page_size)
    elif args.cmd == "download":
        download_from_batch(args.batch_name, args.output_jsonl)
    elif args.cmd == "merge":
        merge_results(
            batch_output_jsonl=args.batch_output_jsonl,
            metadata_jsonl=args.metadata_jsonl,
            merged_output_jsonl=args.merged_output_jsonl,
            errors_jsonl=args.errors_jsonl,
        )
    elif args.cmd == "cancel":
        GeminiBatchClient().cancel_batch(args.batch_name)
    elif args.cmd == "delete":
        GeminiBatchClient().delete_batch(args.batch_name)


if __name__ == "__main__":
    main()
