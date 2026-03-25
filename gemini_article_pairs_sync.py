import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional
from dotenv import load_dotenv

from google import genai
from google.genai import errors, types

load_dotenv()


DEFAULT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_REQUEST_TIMEOUT_MS = 120000
DEFAULT_REQUEST_ATTEMPTS = 5
DEFAULT_REQUEST_BACKOFF_SEC = 2.0
DEFAULT_REQUEST_BACKOFF_MAX_SEC = 30.0


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
- if the article is mostly local event detail, battlefield trivia, casualty counts, or thin operational updates, return 0 pairs
- when in doubt, prefer returning 0 pairs over weak, redundant, or overly article-specific pairs

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
11. Before emitting a pair, ask:
   "If I remove the political framing from this answer, does the pair become mostly trivial factual extraction?"
   If yes, do not emit that pair.
12. Do NOT ask questions about "the article" itself. Ask standalone questions about the claim, framing, justification, responsibility, threat, legitimacy, or political implication.
13. If the only possible questions would be article-specific recap questions, output no pairs.
14. Never emit placeholder or empty pair objects. If nothing qualifies, return {"pairs": []}.

Style requirements:
- Questions should sound natural, meaningful, and reusable outside the source article.
- Answers should be short, clear, and reusable.
- Do NOT use question forms like "How does the article frame...", "According to the article...", "What does the article say about...", or similar article-referential wording.
- Prefer direct, standalone formulations such as "Why is support for Ukraine portrayed as necessary?", "How is Russia portrayed as a threat?", "What makes Ukraine's position legitimate?", or "Why are sanctions framed as justified?"

Return valid JSON matching the provided schema.
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


FOOTER_MARKERS = [
    "We use cookies to provide you with the best experience on our website.",
    "By continuing to use the site, you agree to the use of cookies.",
    "Learn more about our cookie policy.",
]


class GeminiResponseFormatError(ValueError):
    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def add_property_ordering(schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = json.loads(json.dumps(schema))

    def walk(node: Dict[str, Any]) -> None:
        if node.get("type") == "object" and "properties" in node:
            node["propertyOrdering"] = list(node["properties"].keys())
            for child in node["properties"].values():
                if isinstance(child, dict):
                    walk(child)
        elif node.get("type") == "array" and isinstance(node.get("items"), dict):
            walk(node["items"])

    walk(schema)
    return schema


SCHEMA_WITH_ORDER = add_property_ordering(SCHEMA)


def normalize_text(value: Optional[str]) -> str:
    return (value or "").replace("\x00", " ").strip()


def clean_article_text(value: Optional[str]) -> str:
    text = normalize_text(value)
    lowered = text.lower()

    cut_positions = []
    for marker in FOOTER_MARKERS:
        pos = lowered.find(marker.lower())
        if pos != -1:
            cut_positions.append(pos)
    if cut_positions:
        text = text[: min(cut_positions)].rstrip()

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lowered_line = stripped.lower()
        if not stripped:
            cleaned_lines.append("")
            continue
        if lowered_line.startswith("we use cookies"):
            continue
        if lowered_line.startswith("by continuing to use the site"):
            continue
        if lowered_line.startswith("learn more about our cookie policy"):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_article_prompt(row: Dict[str, str], max_chars: int) -> str:
    title = normalize_text(row.get("title"))
    source = normalize_text(row.get("source"))
    published_at = normalize_text(row.get("published_at"))
    url = normalize_text(row.get("url"))
    text = clean_article_text(row.get("text"))

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED]"

    return f'''ARTICLE_METADATA
source: {source}
title: {title}
published_at: {published_at}
url: {url}

ARTICLE_TEXT
"""
{text}
"""
'''


def iter_csv_rows(path: str) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def extract_json_from_text(raw: str) -> Dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        fragments = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                fragments.append(part_text)
        if fragments:
            return "\n".join(fragments).strip()

    return ""


def extract_finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    reasons = []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            reasons.append(str(reason))
    return ", ".join(reasons)


def parse_response_payload(response: Any) -> Dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed

    raw_text = extract_response_text(response)
    if raw_text:
        try:
            return extract_json_from_text(raw_text)
        except json.JSONDecodeError as exc:
            finish_reason = extract_finish_reason(response)
            message = f"invalid JSON from Gemini: {exc}"
            if finish_reason:
                message = f"{message} (finish_reason={finish_reason})"
            raise GeminiResponseFormatError(
                message,
                raw_text=raw_text,
            ) from exc

    raise GeminiResponseFormatError(
        "Gemini returned neither parsed output nor text",
        raw_text=raw_text,
    )


def is_retryable_request_error(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError) and exc.code in {429, 503}:
        return True

    timeout_types = (
        errors.httpx.TimeoutException,
        TimeoutError,
    )
    if isinstance(exc, timeout_types):
        return True

    return False


def format_retryable_error(exc: Exception) -> str:
    if isinstance(exc, errors.APIError):
        status = f" {exc.status}" if exc.status else ""
        return f"{exc.code}{status}"
    return type(exc).__name__


def generate_content_with_retries(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_ms: int,
    request_attempts: int,
    request_backoff_sec: float,
    request_backoff_max_sec: float,
) -> Any:
    attempts = max(1, request_attempts)

    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=DEVELOPER_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=SCHEMA_WITH_ORDER,
                    temperature=temperature,
                    candidate_count=1,
                    max_output_tokens=max_output_tokens,
                    http_options=types.HttpOptions(
                        timeout=request_timeout_ms,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                ),
            )
        except Exception as exc:
            if not is_retryable_request_error(exc) or attempt >= attempts:
                raise

            delay = min(request_backoff_sec * (2 ** (attempt - 1)), request_backoff_max_sec)
            print(
                f"  -> retryable error: {format_retryable_error(exc)}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{attempts})",
                flush=True,
            )
            time.sleep(delay)


def call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    request_timeout_ms: int,
    request_attempts: int,
    request_backoff_sec: float,
    request_backoff_max_sec: float,
    retry_invalid_json: int = 2,
    retry_delay_sec: float = 1.0,
) -> Dict[str, Any]:
    attempts = retry_invalid_json + 1
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        request_prompt = prompt
        request_temperature = temperature
        if attempt > 1:
            request_prompt = (
                prompt
                + "\n\nIMPORTANT: Return only one complete JSON object matching the schema. "
                "Do not include markdown. Do not stop mid-object."
            )
            request_temperature = max(temperature, 0.2)

        response = generate_content_with_retries(
            client=client,
            model=model,
            prompt=request_prompt,
            temperature=request_temperature,
            max_output_tokens=max_output_tokens,
            request_timeout_ms=request_timeout_ms,
            request_attempts=request_attempts,
            request_backoff_sec=request_backoff_sec,
            request_backoff_max_sec=request_backoff_max_sec,
        )

        try:
            return parse_response_payload(response)
        except GeminiResponseFormatError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_delay_sec)

    if last_error is not None:
        raise GeminiResponseFormatError(
            f"{last_error} after {attempts} attempts",
            raw_text=getattr(last_error, "raw_text", ""),
        )

    raise RuntimeError("Gemini call failed without returning a parseable response")


def process_rows(
    input_csv: str,
    output_jsonl: str,
    model: str,
    max_rows: Optional[int],
    start_row: int,
    max_chars: int,
    temperature: float,
    max_output_tokens: int,
    request_timeout_ms: int,
    request_attempts: int,
    request_backoff_sec: float,
    request_backoff_max_sec: float,
    sleep_sec: float,
    overwrite: bool,
) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")

    attempted = 0
    processed = 0
    failures = 0
    mode = "w" if overwrite else "a"

    with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client, open(
        output_jsonl, mode, encoding="utf-8"
    ) as fout:
        for idx, row in enumerate(iter_csv_rows(input_csv), start=1):
            if idx < start_row:
                continue

            url = normalize_text(row.get("url"))
            text = normalize_text(row.get("text"))
            if not url or not text:
                print(f"[skip row {idx}] missing url or text")
                continue
            if max_rows is not None and attempted >= max_rows:
                break

            attempted += 1

            prompt = build_article_prompt(row, max_chars=max_chars)
            print(f"[row {idx}] {url}")

            try:
                parsed = call_gemini(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    request_timeout_ms=request_timeout_ms,
                    request_attempts=request_attempts,
                    request_backoff_sec=request_backoff_sec,
                    request_backoff_max_sec=request_backoff_max_sec,
                )

                result = {
                    "title": normalize_text(row.get("title")),
                    "source": normalize_text(row.get("source")),
                    "published_at": normalize_text(row.get("published_at")),
                    "url": url,
                    "text": text,
                    "pairs": parsed.get("pairs", []),
                }
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
                print(f"  -> ok, pairs={len(result['pairs'])}")
            except Exception as e:
                failures += 1
                error_record = {
                    "title": normalize_text(row.get("title")),
                    "source": normalize_text(row.get("source")),
                    "published_at": normalize_text(row.get("published_at")),
                    "url": url,
                    "text": text,
                    "error": str(e),
                    "pairs": [],
                }
                raw_model_text = getattr(e, "raw_text", "")
                if raw_model_text:
                    error_record["raw_model_text"] = raw_model_text
                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"  -> error: {e}")

            if sleep_sec > 0:
                time.sleep(sleep_sec)

    print(f"[done] processed={processed} failures={failures} output={output_jsonl}")


def process_single_article(
    title: str,
    text: str,
    source: str,
    published_at: str,
    url: str,
    model: str,
    max_chars: int,
    temperature: float,
    max_output_tokens: int,
    request_timeout_ms: int,
    request_attempts: int,
    request_backoff_sec: float,
    request_backoff_max_sec: float,
) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")

    row = {
        "title": title,
        "text": text,
        "source": source,
        "published_at": published_at,
        "url": url,
    }
    prompt = build_article_prompt(row, max_chars=max_chars)

    with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client:
        parsed = call_gemini(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            request_timeout_ms=request_timeout_ms,
            request_attempts=request_attempts,
            request_backoff_sec=request_backoff_sec,
            request_backoff_max_sec=request_backoff_max_sec,
        )

    result = {
        "title": title,
        "source": source,
        "published_at": published_at,
        "url": url,
        "text": text,
        "pairs": parsed.get("pairs", []),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronous Gemini extractor for grounded pro-Ukrainian QA pairs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_csv = sub.add_parser("csv", help="Process a CSV file row by row.")
    p_csv.add_argument("--input-csv", required=True)
    p_csv.add_argument("--output-jsonl", required=True)
    p_csv.add_argument("--model", default="gemini-2.5-flash")
    p_csv.add_argument("--max-rows", type=int, default=None)
    p_csv.add_argument("--start-row", type=int, default=1)
    p_csv.add_argument("--max-chars", type=int, default=12000)
    p_csv.add_argument("--temperature", type=float, default=0.0)
    p_csv.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p_csv.add_argument("--request-timeout-ms", type=int, default=DEFAULT_REQUEST_TIMEOUT_MS)
    p_csv.add_argument("--request-attempts", type=int, default=DEFAULT_REQUEST_ATTEMPTS)
    p_csv.add_argument("--request-backoff-sec", type=float, default=DEFAULT_REQUEST_BACKOFF_SEC)
    p_csv.add_argument("--request-backoff-max-sec", type=float, default=DEFAULT_REQUEST_BACKOFF_MAX_SEC)
    p_csv.add_argument("--sleep-sec", type=float, default=0.0)
    p_csv.add_argument("--overwrite", action="store_true")

    p_one = sub.add_parser("one", help="Process one article passed directly on the command line.")
    p_one.add_argument("--title", required=True)
    p_one.add_argument("--text", required=True)
    p_one.add_argument("--source", default="")
    p_one.add_argument("--published-at", default="")
    p_one.add_argument("--url", default="")
    p_one.add_argument("--model", default="gemini-2.5-flash")
    p_one.add_argument("--max-chars", type=int, default=12000)
    p_one.add_argument("--temperature", type=float, default=0.0)
    p_one.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    p_one.add_argument("--request-timeout-ms", type=int, default=DEFAULT_REQUEST_TIMEOUT_MS)
    p_one.add_argument("--request-attempts", type=int, default=DEFAULT_REQUEST_ATTEMPTS)
    p_one.add_argument("--request-backoff-sec", type=float, default=DEFAULT_REQUEST_BACKOFF_SEC)
    p_one.add_argument("--request-backoff-max-sec", type=float, default=DEFAULT_REQUEST_BACKOFF_MAX_SEC)

    args = parser.parse_args()

    if args.cmd == "csv":
        process_rows(
            input_csv=args.input_csv,
            output_jsonl=args.output_jsonl,
            model=args.model,
            max_rows=args.max_rows,
            start_row=args.start_row,
            max_chars=args.max_chars,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            request_timeout_ms=args.request_timeout_ms,
            request_attempts=args.request_attempts,
            request_backoff_sec=args.request_backoff_sec,
            request_backoff_max_sec=args.request_backoff_max_sec,
            sleep_sec=args.sleep_sec,
            overwrite=args.overwrite,
        )
    elif args.cmd == "one":
        process_single_article(
            title=args.title,
            text=args.text,
            source=args.source,
            published_at=args.published_at,
            url=args.url,
            model=args.model,
            max_chars=args.max_chars,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            request_timeout_ms=args.request_timeout_ms,
            request_attempts=args.request_attempts,
            request_backoff_sec=args.request_backoff_sec,
            request_backoff_max_sec=args.request_backoff_max_sec,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
