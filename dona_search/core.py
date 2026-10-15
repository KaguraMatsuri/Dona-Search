import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CORE_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "core"
BASE_MODEL = "BAAI/bge-reranker-v2-m3"
BASE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MAX_LENGTH = 512
QUERY_LENGTH = 128
STRIDE = 64


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Keep offsets into the extracted Document.text, including final punctuation."""
    result = []
    boundaries = [(match.start(), match.end()) for match in re.finditer(r"\r?\n[ \t]*\r?\n+", text)]
    starts = [0, *(end for _, end in boundaries)]
    ends = [*(start for start, _ in boundaries), len(text)]
    for start, end in zip(starts, ends):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            result.append((start, end))
    return result


def candidate_spans(tokenizer, query: str, text: str, *, paragraphs=True) -> list[tuple[int, int]]:
    """Use the formal tokenizer overflow/offset API only for long paragraphs."""
    query_ids = tokenizer(query, add_special_tokens=False)["input_ids"][:QUERY_LENGTH]
    query = tokenizer.decode(query_ids, skip_special_tokens=True)
    result = []
    for start, end in paragraph_spans(text) if paragraphs else [(0, len(text))] if text.strip() else []:
        paragraph = text[start:end]
        if len(tokenizer(query, paragraph, verbose=False)["input_ids"]) <= MAX_LENGTH:
            result.append((start, end))
            continue
        encoded = tokenizer(
            query,
            paragraph,
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        for i, offsets in enumerate(encoded["offset_mapping"]):
            document_offsets = [
                offset
                for offset, sequence in zip(offsets, encoded.sequence_ids(i))
                if sequence == 1 and offset[1] > offset[0]
            ]
            if document_offsets:
                span = (start + document_offsets[0][0], start + document_offsets[-1][1])
                # A character slice can retokenize differently at a subword boundary.
                # Trim with the tokenizer's formal offsets so scoring never drops a tail
                # that the displayed evidence or positive training label includes.
                while len(tokenizer(query, text[span[0]:span[1]], verbose=False)["input_ids"]) > MAX_LENGTH:
                    trimmed = tokenizer(
                        query, text[span[0]:span[1]], truncation="only_second",
                        max_length=MAX_LENGTH, return_offsets_mapping=True,
                    )
                    ends = [b for (_, b), seq in zip(trimmed["offset_mapping"], trimmed.sequence_ids()) if seq == 1 and b > 0]
                    # Several byte/subword tokens can share one Unicode character offset.
                    new_end = span[0] + max((b for b in ends if b < span[1] - span[0]), default=0)
                    if new_end <= span[0]:
                        raise ValueError("window_offset_invalid")
                    span = (span[0], new_end)
                if not result or span != result[-1]:
                    result.append(span)
    return result


class CoreModel:
    """One fixed cross-encoder shared by ranking and verbatim evidence selection."""

    @staticmethod
    def validate_checkpoint(path: Path) -> dict:
        manifest_path = path / "core_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("core_manifest_missing", str(manifest_path))
        manifest = json.loads(manifest_path.read_text())
        threshold = manifest.get("threshold")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("base_model") != BASE_MODEL
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
            or manifest.get("max_length") != MAX_LENGTH
            or not manifest.get("files")
        ):
            raise ValueError("core_manifest_invalid")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or file_hash(path / name) != digest:
                raise ValueError("core_checkpoint_hash_mismatch")
        if not {"model.safetensors", "config.json", "tokenizer.json"} <= set(manifest["files"]):
            raise ValueError("core_checkpoint_incomplete")
        return manifest

    def __init__(self, checkpoint: Path = CORE_CHECKPOINT, batch_size: int = 8):
        self.manifest = self.validate_checkpoint(checkpoint)
        if not torch.backends.mps.is_available():
            raise RuntimeError("mps_required")
        if batch_size < 1:
            raise ValueError("batch_size_invalid")
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        self.device = "mps"
        self.batch_size = batch_size
        self.threshold = float(self.manifest["threshold"])
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=torch.float16,
            device_map={"": self.device},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        queries_by_text = {
            query: self.tokenizer.decode(
                self.tokenizer(query, add_special_tokens=False)["input_ids"][:QUERY_LENGTH],
                skip_special_tokens=True,
            ) for query in dict.fromkeys(q for q, _ in pairs)
        }
        with torch.inference_mode():
            for offset in range(0, len(pairs), self.batch_size):
                batch = pairs[offset : offset + self.batch_size]
                queries = [queries_by_text[q] for q, _ in batch]
                encoded = self.tokenizer(
                    queries,
                    [text for _, text in batch],
                    padding=True,
                    truncation="only_second",
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits.squeeze(-1).float()
                if not torch.isfinite(logits).all().item():
                    raise ValueError("core_nonfinite_logits")
                values = torch.sigmoid(logits).cpu().tolist()
                if any(not math.isfinite(score) for score in values):
                    raise ValueError("core_nonfinite_score")
                scores.extend(values)
                del encoded, logits
        return scores

    def warmup(self) -> None:
        self.score_pairs([("What stays loaded?", "The model weights stay loaded.")])

    def cleanup(self) -> None:
        torch.mps.synchronize()
        torch.mps.empty_cache()

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.cleanup()
