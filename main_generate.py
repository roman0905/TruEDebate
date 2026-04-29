"""
TruEDebate (TED) - Stage 1 offline debate generation.

This script generates one complete debate record per news sample and saves it
as JSON. It is designed to be resilient to transient API failures and partial
outputs by using:
1. Per-sample retries with backoff.
2. Validation of both existing and newly generated JSON files.
3. Atomic writes through temporary files.
4. A final serial rescue pass for failed samples.
5. A failure manifest for any samples that still fail.
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
from debate_flow.model import DebateModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.ROOT_DIR / "generate.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


EXPECTED_NODE_COUNT = 7
EXPECTED_EDGE_COUNT = 22
COMPLETE_END_CHARS = '.!?"\')]}。！？；：”’）】》0123456789'
REQUIRED_KEYS = {
    "id",
    "news_text",
    "label",
    "dataset",
    "split",
    "nodes",
    "edge_index",
    "synthesis",
    "claim_texts",
    "claims",
    "rationale_cards",
    "synthesis_structured",
}


def load_dataset(dataset: str, split: str) -> list[dict]:
    """Load the source news dataset."""
    data_dir = config.DATA_DIR
    possible_paths = [
        data_dir / dataset / f"{split}.json",
        data_dir / dataset / f"{split}.jsonl",
        data_dir / f"{dataset}_{split}.json",
        data_dir / f"{dataset}_{split}.jsonl",
    ]

    file_path = None
    for candidate in possible_paths:
        if candidate.exists():
            file_path = candidate
            break

    if file_path is None:
        raise FileNotFoundError(
            f"Could not find dataset file in {data_dir}. "
            f"Tried: {[str(p) for p in possible_paths]}"
        )

    logger.info("Loading dataset from %s", file_path)

    records: list[dict] = []
    if file_path.suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item.setdefault("id", i)
                records.append(item)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Dataset JSON must be a list, got {type(data)} from {file_path}"
            )
        for i, item in enumerate(data):
            item.setdefault("id", i)
            records.append(item)

    for item in records:
        if "text" not in item and "content" in item:
            item["text"] = item["content"]
        if "label" in item:
            item["label"] = config.LABEL_MAP.get(item["label"], item["label"])

    logger.info("Loaded %s records", len(records))
    return records


def load_evidence_file(evidence_file: str | Path | None) -> dict[int, list[dict]]:
    """Load optional EviTED evidence cards aligned by sample id."""
    if evidence_file is None:
        return {}

    path = Path(evidence_file)
    if not path.exists():
        raise FileNotFoundError(f"Evidence file not found: {path}")

    rows: list[dict] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("id", key)
                else:
                    row = {"id": key, "evidence_cards": value}
                rows.append(row)
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported evidence JSON payload: {type(payload)}")

    evidence_map: dict[int, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            continue
        cards = row.get("evidence_cards", row.get("retrieval_evidence", []))
        if not isinstance(cards, list):
            cards = []
        evidence_map[int(row["id"])] = cards
    logger.info("Loaded evidence for %s samples from %s", len(evidence_map), path)
    return evidence_map
def _output_file_for(
    item_id: int, output_dir: Path, dataset: str, split: str
) -> Path:
    return output_dir / f"{dataset}_{split}_{item_id:06d}.json"


def _temp_file_for(output_file: Path) -> Path:
    return output_file.with_suffix(f"{output_file.suffix}.tmp")


def _text_looks_incomplete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] not in COMPLETE_END_CHARS:
        return True
    return False


def _validate_output_record(
    record: dict,
    item: dict,
    dataset: str,
    split: str,
    require_evidence: bool = False,
) -> str | None:
    missing_keys = REQUIRED_KEYS - set(record.keys())
    if missing_keys:
        return f"missing_keys={sorted(missing_keys)}"

    if record["id"] != item["id"]:
        return f"id_mismatch={record['id']}!={item['id']}"
    if record["dataset"] != dataset:
        return f"dataset_mismatch={record['dataset']}!={dataset}"
    if record["split"] != split:
        return f"split_mismatch={record['split']}!={split}"

    nodes = record["nodes"]
    if not isinstance(nodes, list) or len(nodes) != EXPECTED_NODE_COUNT:
        return f"bad_nodes={type(nodes)}:{len(nodes) if isinstance(nodes, list) else 'na'}"

    expected_role_ids = list(range(EXPECTED_NODE_COUNT))
    role_ids = [node.get("role_id") for node in nodes]
    if role_ids != expected_role_ids:
        return f"bad_role_ids={role_ids}"

    for node in nodes:
        text = node.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return f"empty_text={node.get('role_name')}"
        if _text_looks_incomplete(text):
            return f"incomplete_text={node.get('role_name')}"
        if node.get("role_id") != config.ROLE_IDS["synthesis"]:
            argument = str(node.get("argument", "")).strip()
            reasoning = str(node.get("reasoning", "")).strip()
            if not argument:
                return f"empty_argument={node.get('role_name')}"
            if not reasoning:
                return f"empty_reasoning={node.get('role_name')}"

    edge_index = record["edge_index"]
    if (
        not isinstance(edge_index, list)
        or len(edge_index) != 2
        or not all(isinstance(x, list) for x in edge_index)
        or len(edge_index[0]) != EXPECTED_EDGE_COUNT
        or len(edge_index[1]) != EXPECTED_EDGE_COUNT
    ):
        return "bad_edge_index"

    synthesis = record.get("synthesis", "")
    if not isinstance(synthesis, str) or not synthesis.strip():
        return "empty_synthesis"
    if _text_looks_incomplete(synthesis):
        return "incomplete_synthesis"

    claim_texts = record.get("claim_texts")
    if not isinstance(claim_texts, list) or len(claim_texts) == 0:
        return "empty_claim_texts"
    if not all(isinstance(x, str) and x.strip() for x in claim_texts):
        return "bad_claim_texts"

    claims = record.get("claims")
    if not isinstance(claims, list) or len(claims) == 0:
        return "empty_claims"
    for claim in claims:
        if not isinstance(claim, dict):
            return "bad_claim_obj"
        if not str(claim.get("id", "")).strip() or not str(claim.get("content", "")).strip():
            return "bad_claim_fields"

    rationale_cards = record.get("rationale_cards")
    if not isinstance(rationale_cards, list):
        return "bad_rationale_cards"
    for card in rationale_cards:
        if not isinstance(card, dict):
            return "bad_rationale_card_obj"
        if not str(card.get("id", "")).strip() or not str(card.get("content", "")).strip():
            return "bad_rationale_card_fields"

    evidence_cards = record.get("evidence_cards", [])
    if require_evidence:
        if not isinstance(evidence_cards, list) or len(evidence_cards) == 0:
            return "empty_evidence_cards"
    if evidence_cards:
        if not isinstance(evidence_cards, list):
            return "bad_evidence_cards"
        for card in evidence_cards:
            if not isinstance(card, dict):
                return "bad_evidence_card_obj"
            if not str(card.get("id", "")).strip():
                return "bad_evidence_card_id"
            if not str(card.get("evidence_text", "")).strip():
                return "bad_evidence_card_text"

    synthesis_structured = record.get("synthesis_structured")
    if not isinstance(synthesis_structured, dict):
        return "bad_synthesis_structured"
    if not str(synthesis_structured.get("final_debate_tendency", "")).strip():
        return "bad_synthesis_tendency"

    return None


def _load_existing_output(output_file: Path) -> dict | None:
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read existing output %s: %s", output_file.name, e)
        return None


def _cleanup_invalid_output(output_file: Path) -> None:
    temp_file = _temp_file_for(output_file)
    for path in (output_file, temp_file):
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                logger.warning("Failed to remove invalid file %s: %s", path, e)


def _write_output_atomic(output: dict, output_file: Path) -> None:
    temp_file = _temp_file_for(output_file)
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    temp_file.replace(output_file)


def _build_output(item: dict, dataset: str, split: str, debate_record: dict) -> dict:
    return {
        "id": item["id"],
        "news_text": item["text"],
        "label": item["label"],
        "dataset": dataset,
        "split": split,
        "nodes": debate_record["nodes"],
        "edge_index": debate_record["edge_index"],
        "synthesis": debate_record["synthesis"],
        "claim_texts": debate_record["claim_texts"],
        "claims": debate_record["claims"],
        "rationale_cards": debate_record["rationale_cards"],
        "evidence_cards": debate_record.get("evidence_cards", []),
        "synthesis_structured": debate_record["synthesis_structured"],
        "td_rationale": item.get("td_rationale", ""),
        "cs_rationale": item.get("cs_rationale", ""),
        "td_pred": item.get("td_pred", -1),
        "cs_pred": item.get("cs_pred", -1),
        "td_acc": item.get("td_acc", -1),
        "cs_acc": item.get("cs_acc", -1),
        "time": item.get("time", ""),
        "source_id": item.get("source_id", -1),
    }


def _generate_one_record(
    item: dict,
    dataset: str,
    split: str,
    evidence_cards: list[dict] | None = None,
) -> dict:
    debate_model = DebateModel(
        item["text"],
        lang=dataset,
        td_rationale=item.get("td_rationale", ""),
        cs_rationale=item.get("cs_rationale", ""),
        td_pred=item.get("td_pred", -1),
        cs_pred=item.get("cs_pred", -1),
        evidence_cards=evidence_cards,
    )
    debate_model.step()
    debate_record = debate_model.get_debate_record()
    return _build_output(item, dataset, split, debate_record)


def process_single_news(
    item: dict,
    output_dir: Path,
    dataset: str,
    split: str,
    item_retries: int,
    retry_backoff_s: float,
    evidence_cards: list[dict] | None = None,
    require_evidence: bool = False,
) -> dict:
    """Generate one sample with validation, retries, and atomic writes."""
    item_id = item["id"]
    output_file = _output_file_for(item_id, output_dir, dataset, split)

    if output_file.exists():
        existing = _load_existing_output(output_file)
        if existing is not None:
            validation_error = _validate_output_record(
                existing,
                item,
                dataset,
                split,
                require_evidence=require_evidence,
            )
            if validation_error is None:
                return {
                    "status": "SKIP",
                    "item_id": item_id,
                    "file": output_file.name,
                    "attempts": 0,
                }
            logger.warning(
                "Existing output %s is invalid (%s); regenerating",
                output_file.name,
                validation_error,
            )
        _cleanup_invalid_output(output_file)

    last_error = "unknown"
    for attempt in range(1, item_retries + 1):
        try:
            output = _generate_one_record(item, dataset, split, evidence_cards=evidence_cards)
            validation_error = _validate_output_record(
                output,
                item,
                dataset,
                split,
                require_evidence=require_evidence,
            )
            if validation_error is not None:
                raise ValueError(f"generated_invalid_output: {validation_error}")

            _write_output_atomic(output, output_file)
            logger.info(
                "Generated sample id=%s file=%s on attempt=%s",
                item_id,
                output_file.name,
                attempt,
            )
            return {
                "status": "OK",
                "item_id": item_id,
                "file": output_file.name,
                "attempts": attempt,
            }
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Generation failed for id=%s attempt=%s/%s: %s",
                item_id,
                attempt,
                item_retries,
                e,
            )
            _cleanup_invalid_output(output_file)
            if attempt < item_retries:
                time.sleep(retry_backoff_s * attempt)

    logger.error("Generation failed permanently for id=%s: %s", item_id, last_error)
    return {
        "status": "ERROR",
        "item_id": item_id,
        "file": output_file.name,
        "attempts": item_retries,
        "error": last_error,
    }


def _run_generation_pass(
    records: list[dict],
    output_dir: Path,
    dataset: str,
    split: str,
    max_workers: int,
    item_retries: int,
    retry_backoff_s: float,
    desc: str,
    evidence_map: dict[int, list[dict]] | None = None,
    require_evidence: bool = False,
) -> tuple[dict, list[dict]]:
    results = {"OK": 0, "SKIP": 0, "ERROR": 0}
    failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_news,
                item,
                output_dir,
                dataset,
                split,
                item_retries,
                retry_backoff_s,
                (evidence_map or {}).get(int(item["id"]), []),
                require_evidence,
            ): item["id"]
            for item in records
        }

        with tqdm(total=len(futures), desc=desc) as pbar:
            for future in as_completed(futures):
                result = future.result()
                status = result["status"]
                results[status] = results.get(status, 0) + 1
                if status == "ERROR":
                    failures.append(result)
                pbar.set_postfix(
                    ok=results["OK"],
                    skip=results["SKIP"],
                    err=results["ERROR"],
                )
                pbar.update(1)

    return results, failures


def _cleanup_temp_files(output_dir: Path) -> int:
    removed = 0
    for temp_file in output_dir.glob("*.json.tmp"):
        try:
            temp_file.unlink()
            removed += 1
        except OSError as e:
            logger.warning("Failed to remove temp file %s: %s", temp_file, e)
    return removed


def _write_failure_manifest(
    output_dir: Path,
    dataset: str,
    split: str,
    failures: list[dict],
) -> Path | None:
    if not failures:
        return None

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        config.CHECKPOINT_DIR / f"generate_failures_{dataset}_{split}.jsonl"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TruEDebate - Stage 1 multi-threaded debate generation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Dataset language",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=config.MAX_WORKERS,
        help="Maximum worker threads",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional sample cap for debugging",
    )
    parser.add_argument(
        "--item_retries",
        type=int,
        default=4,
        help="Retry count per sample before marking it failed",
    )
    parser.add_argument(
        "--retry_backoff_s",
        type=float,
        default=2.0,
        help="Base backoff seconds between per-sample retries",
    )
    parser.add_argument(
        "--final_retry_rounds",
        type=int,
        default=1,
        help="Additional serial rescue rounds for failed samples",
    )
    parser.add_argument(
        "--evidence_file",
        type=str,
        default=None,
        help="Optional EviTED evidence JSON/JSONL file aligned by sample id",
    )
    args = parser.parse_args()

    output_dir = config.OUTPUT_DIR / f"{args.dataset}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    removed_temp_files = _cleanup_temp_files(output_dir)
    if removed_temp_files:
        logger.info("Removed %s stale temp files from %s", removed_temp_files, output_dir)

    records = load_dataset(args.dataset, args.split)
    evidence_map = load_evidence_file(args.evidence_file)
    require_evidence = args.evidence_file is not None
    if args.max_samples is not None:
        records = records[: args.max_samples]
        logger.info("Truncated to %s samples", args.max_samples)

    logger.info(
        "Starting generation: dataset=%s split=%s samples=%s workers=%s "
        "item_retries=%s final_retry_rounds=%s",
        args.dataset,
        args.split,
        len(records),
        args.max_workers,
        args.item_retries,
        args.final_retry_rounds,
    )

    all_results = {"OK": 0, "SKIP": 0, "ERROR": 0}
    first_pass_results, first_pass_failures = _run_generation_pass(
        records=records,
        output_dir=output_dir,
        dataset=args.dataset,
        split=args.split,
        max_workers=args.max_workers,
        item_retries=args.item_retries,
        retry_backoff_s=args.retry_backoff_s,
        desc="Generating debates",
        evidence_map=evidence_map,
        require_evidence=require_evidence,
    )
    for key, value in first_pass_results.items():
        all_results[key] = all_results.get(key, 0) + value

    failed_ids = {row["item_id"] for row in first_pass_failures}
    failed_records = [item for item in records if item["id"] in failed_ids]

    for round_idx in range(1, args.final_retry_rounds + 1):
        if not failed_records:
            break
        logger.info(
            "Starting final rescue round %s for %s failed samples",
            round_idx,
            len(failed_records),
        )
        round_results, round_failures = _run_generation_pass(
            records=failed_records,
            output_dir=output_dir,
            dataset=args.dataset,
            split=args.split,
            max_workers=1,
            item_retries=args.item_retries,
            retry_backoff_s=args.retry_backoff_s,
            desc=f"Rescue round {round_idx}",
            evidence_map=evidence_map,
            require_evidence=require_evidence,
        )
        for key, value in round_results.items():
            if key != "SKIP":
                all_results[key] = all_results.get(key, 0) + value
        failed_ids = {row["item_id"] for row in round_failures}
        failed_records = [item for item in failed_records if item["id"] in failed_ids]

    final_failures = []
    for item in failed_records:
        output_file = _output_file_for(item["id"], output_dir, args.dataset, args.split)
        final_failures.append(
            {
                "status": "ERROR",
                "item_id": item["id"],
                "file": output_file.name,
                "error": "failed_after_all_retries",
            }
        )

    manifest_path = _write_failure_manifest(
        output_dir=output_dir,
        dataset=args.dataset,
        split=args.split,
        failures=final_failures,
    )

    logger.info("=" * 60)
    logger.info("Generation summary:")
    logger.info("  OK: %s", all_results["OK"])
    logger.info("  SKIP: %s", all_results["SKIP"])
    logger.info("  ERROR: %s", len(final_failures))
    logger.info("  Output dir: %s", output_dir)
    if manifest_path is not None:
        logger.info("  Failure manifest: %s", manifest_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
