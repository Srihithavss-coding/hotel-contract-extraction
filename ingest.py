"""PDF ingestion: selectable text first, OCR only when a page has too little text.

This module does not drop pages, call an LLM, or decide what is relevant.
"""

from __future__ import annotations

import ctypes
import difflib
import multiprocessing
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pymupdf as fitz
from dotenv import load_dotenv
from PIL import Image, ImageChops, ImageFilter, ImageOps

load_dotenv()

_WORD_RE = re.compile(r"[A-Za-z0-9]")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def _env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip() == "":
        return None
    return int(raw)


@dataclass(frozen=True)
class IngestConfig:
    """Tunable scan-detection and OCR settings. Override via constructor or env vars."""

    min_text_chars: int = 80
    min_alnum_ratio: float = 0.25
    render_dpi: int = 300
    tesseract_cmd: Optional[str] = None
    tessdata_dir: Optional[str] = None
    tesseract_lang: str = "eng"
    tesseract_psm: int = 11
    extra_space_gap_ratio: float = 1.6
    ocr_crop_content: bool = True
    ocr_crop_padding: int = 60
    ocr_grayscale: bool = False
    ocr_upscale_factor: float = 1.0
    ocr_adaptive_threshold: bool = False
    ocr_adaptive_radius: int = 17
    ocr_adaptive_offset: int = 6
    ocr_refinement_psm: Optional[int] = 6

    @classmethod
    def from_env(cls) -> "IngestConfig":
        cmd = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
        tessdata = (
            os.environ.get("TESSDATA_DIR")
            or os.environ.get("TESSDATA_PREFIX")
            or os.environ.get("INGEST_TESSDATA_DIR")
        )
        return cls(
            min_text_chars=_env_int("INGEST_MIN_TEXT_CHARS", 80),
            min_alnum_ratio=_env_float("INGEST_MIN_ALNUM_RATIO", 0.25),
            render_dpi=_env_int("INGEST_RENDER_DPI", 300),
            tesseract_cmd=cmd.strip() if cmd else None,
            tessdata_dir=tessdata.strip() if tessdata else None,
            tesseract_lang=os.environ.get("INGEST_TESSERACT_LANG", "eng"),
            tesseract_psm=_env_int("INGEST_TESSERACT_PSM", 11),
            extra_space_gap_ratio=_env_float("INGEST_EXTRA_SPACE_GAP_RATIO", 1.6),
            ocr_crop_content=_env_bool("INGEST_OCR_CROP_CONTENT", True),
            ocr_crop_padding=_env_int("INGEST_OCR_CROP_PADDING", 60),
            ocr_grayscale=_env_bool("INGEST_OCR_GRAYSCALE", False),
            ocr_upscale_factor=_env_float("INGEST_OCR_UPSCALE_FACTOR", 1.0),
            ocr_adaptive_threshold=_env_bool("INGEST_OCR_ADAPTIVE_THRESHOLD", False),
            ocr_refinement_psm=_env_optional_int("INGEST_OCR_REFINEMENT_PSM", 6),
        )


@dataclass
class IngestionTimings:
    pdf_discovery_seconds: float = 0.0
    pdf_ingestion_seconds: float = 0.0
    rendering_seconds: float = 0.0
    ocr_seconds: float = 0.0


@dataclass(frozen=True)
class OCRWord:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: Optional[float]


@dataclass(frozen=True)
class PageLine:
    text: str
    words: List[OCRWord] = field(default_factory=list)


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: Optional[float]
    words: List[OCRWord]
    lines: List[PageLine]


@dataclass(frozen=True)
class _PreparedOCRImage:
    image: Image.Image
    offset_x: int
    offset_y: int
    scale: float


@dataclass
class PageContent:
    file: str
    page: int
    text: str
    extraction_method: str
    ocr_confidence: Optional[float]
    unreadable: bool = False
    warning: Optional[str] = None
    ocr_words: List[OCRWord] = field(default_factory=list, repr=False)
    lines: List[PageLine] = field(default_factory=list)
    document_id: str = field(default="", repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "page": self.page,
            "text": self.text,
            "extraction_method": self.extraction_method,
            "ocr_confidence": self.ocr_confidence,
            "lines": [
                {
                    "text": line.text,
                    "words": [asdict(word) for word in line.words],
                }
                for line in self.lines
            ],
            "ocr_words": [asdict(word) for word in self.ocr_words],
        }


class TesseractUnavailable(RuntimeError):
    """Tesseract or its required language data is unavailable."""


class IngestionTimeoutError(TimeoutError):
    """The ingestion deadline expired before every page was processed."""


class IngestionWorkerError(RuntimeError):
    """The isolated PDF ingestion worker failed."""


class NoPDFDocumentsError(RuntimeError):
    """The contract folder contains no supported PDF documents."""


_OCR_CLEANUP_GRACE_SECONDS = 2.0


def ingest_folder(
    folder: str | Path,
    config: Optional[IngestConfig] = None,
    *,
    deadline: Optional[float] = None,
    timings: Optional[IngestionTimings] = None,
) -> List[PageContent]:
    """Read every PDF in a contract folder, page by page. Pages are never discarded."""
    _check_deadline(deadline, "opening the contract folder")
    path = Path(folder)
    if not path.is_dir():
        raise FileNotFoundError(f"Contract folder not found: {path}")
    cfg = config or IngestConfig.from_env()
    discovery_started = time.perf_counter()
    pdf_paths = discover_pdfs(path, deadline=deadline)
    if timings is not None:
        timings.pdf_discovery_seconds += time.perf_counter() - discovery_started
    ingestion_started = time.perf_counter()
    if timings is None:
        pages = _ingest_pdf_paths(pdf_paths, cfg, deadline)
    else:
        pages = _ingest_pdf_paths(pdf_paths, cfg, deadline, timings=timings)
    if timings is not None:
        timings.pdf_ingestion_seconds += time.perf_counter() - ingestion_started
    return pages


def ingest_folder_bounded(
    folder: str | Path,
    deadline: float,
    config: Optional[IngestConfig] = None,
) -> List[PageContent]:
    """Ingest in a killable worker so PDF rendering is covered by the deadline."""
    _check_deadline(deadline, "starting PDF ingestion")
    cfg = config or IngestConfig.from_env()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_ingest_worker,
        args=(str(folder), cfg, deadline, sender),
        name="hotel-contract-ingest",
    )
    try:
        worker.start()
        sender.close()
        while True:
            remaining = _check_deadline(deadline, "ingesting PDF pages")
            if receiver.poll(min(remaining, 0.05)):
                message = receiver.recv()
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
                _check_deadline(deadline, "receiving ingested pages")
                if worker.is_alive():
                    _stop_ingest_worker(worker)
                    raise IngestionWorkerError(
                        "The ingestion worker returned data but did not exit."
                    )
                if message[0] == "ok":
                    return message[1]
                _raise_ingestion_worker_error(message[1], message[2])
            if not worker.is_alive():
                if receiver.poll():
                    continue
                raise IngestionWorkerError(
                    f"The ingestion worker exited with code {worker.exitcode} "
                    "without returning a result."
                )
    finally:
        receiver.close()
        sender.close()
        if worker.pid is not None:
            if worker.is_alive():
                _stop_ingest_worker(worker)
            worker.close()


def _ingest_worker(folder, config, deadline, connection) -> None:
    try:
        path = Path(folder)
        pdf_paths = discover_pdfs(path, deadline=deadline)
        if not pdf_paths:
            raise NoPDFDocumentsError(
                f"No PDF documents found in contract folder: {path}"
            )
        pages = _ingest_pdf_paths(pdf_paths, config, deadline)
        connection.send(("ok", pages))
    except Exception as exc:
        try:
            connection.send(("error", type(exc).__name__, str(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _raise_ingestion_worker_error(error_name: str, message: str) -> None:
    if error_name == "IngestionTimeoutError":
        raise IngestionTimeoutError(message)
    if error_name == "TesseractUnavailable":
        raise TesseractUnavailable(message)
    if error_name == "NoPDFDocumentsError":
        raise NoPDFDocumentsError(message)
    if error_name == "FileNotFoundError":
        raise FileNotFoundError(message)
    raise IngestionWorkerError(f"{error_name}: {message}")


def _stop_ingest_worker(worker) -> None:
    worker.terminate()
    worker.join(timeout=0.25)
    if worker.is_alive():
        worker.kill()
        worker.join()


def _check_deadline(deadline: Optional[float], stage: str) -> Optional[float]:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise IngestionTimeoutError(
            f"The runtime deadline expired while {stage}; no partial page set "
            "was returned."
        )
    return remaining


def _ocr_timeout_seconds(deadline: Optional[float]) -> Optional[float]:
    remaining = _check_deadline(deadline, "preparing OCR")
    if remaining is None:
        return None
    timeout_seconds = remaining - _OCR_CLEANUP_GRACE_SECONDS
    if timeout_seconds <= 0:
        raise IngestionTimeoutError(
            "The runtime deadline is too close to safely start another Tesseract "
            "process; no partial page set was returned."
        )
    return timeout_seconds


def discover_pdfs(
    folder: Path,
    *,
    deadline: Optional[float] = None,
) -> List[Path]:
    _check_deadline(deadline, f"listing PDFs in {folder}")
    paths = []
    for path in folder.iterdir():
        _check_deadline(deadline, f"listing PDFs in {folder}")
        if path.is_file() and path.suffix.lower() == ".pdf":
            paths.append(path)
    _check_deadline(deadline, f"ordering PDFs in {folder}")
    return sorted(paths)


def _ingest_pdf_paths(
    pdf_paths: Sequence[Path],
    config: IngestConfig,
    deadline: Optional[float],
    *,
    timings: Optional[IngestionTimings] = None,
) -> List[PageContent]:
    pages: List[PageContent] = []
    for pdf_path in pdf_paths:
        _check_deadline(deadline, f"opening {pdf_path.name}")
        if timings is None:
            pages.extend(ingest_pdf(pdf_path, config, deadline=deadline))
        else:
            pages.extend(
                ingest_pdf(pdf_path, config, deadline=deadline, timings=timings)
            )
    return pages


def ingest_pdf(
    pdf_path: str | Path,
    config: Optional[IngestConfig] = None,
    *,
    deadline: Optional[float] = None,
    timings: Optional[IngestionTimings] = None,
) -> List[PageContent]:
    _check_deadline(deadline, f"opening {Path(pdf_path).name}")
    cfg = config or IngestConfig.from_env()
    path = Path(pdf_path)
    pages: List[PageContent] = []
    with fitz.open(path) as doc:
        for index in range(doc.page_count):
            _check_deadline(deadline, f"processing {path.name}, page {index + 1}")
            if timings is None:
                page_content = _ingest_page(
                    doc,
                    index,
                    path.name,
                    cfg,
                    document_id=str(path.resolve()),
                    deadline=deadline,
                )
            else:
                page_content = _ingest_page(
                    doc,
                    index,
                    path.name,
                    cfg,
                    document_id=str(path.resolve()),
                    deadline=deadline,
                    timings=timings,
                )
            pages.append(page_content)
    return pages


def needs_ocr(text: str, config: IngestConfig) -> bool:
    stripped = text.strip()
    if len(stripped) < config.min_text_chars:
        return True
    letters = _WORD_RE.findall(stripped)
    ratio = len(letters) / len(stripped) if stripped else 0.0
    return ratio < config.min_alnum_ratio


def _ingest_page(
    doc: fitz.Document,
    index: int,
    filename: str,
    config: IngestConfig,
    document_id: str = "",
    *,
    deadline: Optional[float] = None,
    timings: Optional[IngestionTimings] = None,
) -> PageContent:
    _check_deadline(deadline, f"reading {filename}, page {index + 1}")
    page = doc[index]
    page_number = index + 1
    selectable = page.get_text("text") or ""
    if not needs_ocr(selectable, config):
        text = _normalise_text(selectable)
        return PageContent(
            file=filename,
            page=page_number,
            text=text,
            extraction_method="text",
            ocr_confidence=None,
            lines=_text_lines(text),
            document_id=document_id,
        )

    _check_deadline(deadline, f"rendering {filename}, page {page_number}")
    rendering_started = time.perf_counter()
    image = render_page(page, config.render_dpi)
    if timings is not None:
        timings.rendering_seconds += time.perf_counter() - rendering_started
    _check_deadline(deadline, f"starting OCR for {filename}, page {page_number}")
    ocr_started = time.perf_counter()
    ocr_result = ocr_image(image, config, deadline=deadline)
    if timings is not None:
        timings.ocr_seconds += time.perf_counter() - ocr_started
    _check_deadline(deadline, f"finishing {filename}, page {page_number}")

    combined = ocr_result.text.strip() or _normalise_text(selectable)
    unreadable = not combined.strip()
    warning = "OCR returned no usable text" if unreadable else None
    return PageContent(
        file=filename,
        page=page_number,
        text=combined,
        extraction_method="ocr",
        ocr_confidence=ocr_result.confidence,
        unreadable=unreadable,
        warning=warning,
        ocr_words=ocr_result.words,
        lines=ocr_result.lines,
        document_id=document_id,
    )


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _prepare_ocr_image(
    image: Image.Image,
    config: IngestConfig,
    crop_content: bool,
) -> _PreparedOCRImage:
    prepared = image
    offset_x = 0
    offset_y = 0
    if crop_content and config.ocr_crop_content:
        crop = _content_bounds(image, config)
        if crop is not None:
            offset_x, offset_y = crop[0], crop[1]
            prepared = image.crop(crop)

    if config.ocr_grayscale or config.ocr_adaptive_threshold:
        prepared = ImageOps.grayscale(prepared)
    if config.ocr_adaptive_threshold:
        if config.ocr_adaptive_radius < 1:
            raise ValueError("ocr_adaptive_radius must be at least 1")
        if not 0 <= config.ocr_adaptive_offset <= 255:
            raise ValueError("ocr_adaptive_offset must be between 0 and 255")
        prepared = _adaptive_threshold(
            prepared,
            config.ocr_adaptive_radius,
            config.ocr_adaptive_offset,
        )

    scale = config.ocr_upscale_factor
    if scale <= 0:
        raise ValueError("ocr_upscale_factor must be greater than 0")
    if scale != 1.0:
        prepared = prepared.resize(
            (round(prepared.width * scale), round(prepared.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return _PreparedOCRImage(prepared, offset_x, offset_y, scale)


def _content_bounds(image: Image.Image, config: IngestConfig) -> Optional[tuple[int, int, int, int]]:
    gray = ImageOps.grayscale(image)
    denoised = gray.filter(ImageFilter.MedianFilter(size=5))
    mask = denoised.point(lambda value: 255 if value < 80 else 0)
    bounds = mask.getbbox()
    if bounds is None:
        return None

    scale = config.render_dpi / 300.0
    padding = max(0, round(config.ocr_crop_padding * scale))
    left = max(0, bounds[0] - padding)
    top = max(0, bounds[1] - padding)
    right = min(image.width, bounds[2] + padding)
    bottom = min(image.height, bounds[3] + padding)
    if left == 0 and top == 0 and right == image.width and bottom == image.height:
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _adaptive_threshold(image: Image.Image, radius: int, offset: int) -> Image.Image:
    gray = ImageOps.grayscale(image)
    local_mean = gray.filter(ImageFilter.BoxBlur(radius))
    difference = ImageChops.subtract(
        gray,
        local_mean.point(lambda value: max(0, value - offset)),
        offset=128,
    )
    return difference.point(lambda value: 255 if value > 128 else 0)


def _ocr_words_from_data(data: dict, prepared: _PreparedOCRImage) -> List[OCRWord]:
    words: List[OCRWord] = []
    texts = data.get("text") or []
    for index, raw_text in enumerate(texts):
        text = (raw_text or "").strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            confidence = None

        try:
            left = int(data["left"][index])
            top = int(data["top"][index])
            right = left + int(data["width"][index])
            bottom = top + int(data["height"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        x = prepared.offset_x + round(left / prepared.scale)
        y = prepared.offset_y + round(top / prepared.scale)
        words.append(
            OCRWord(
                text=text,
                x=x,
                y=y,
                width=max(0, round(right / prepared.scale) - round(left / prepared.scale)),
                height=max(0, round(bottom / prepared.scale) - round(top / prepared.scale)),
                confidence=confidence,
            )
        )
    return words


def _merge_ocr_words(
    primary: List[OCRWord],
    candidates: List[OCRWord],
    add_missing: bool,
    minimum_confidence: float = 0,
    minimum_add_confidence: float = 75,
) -> List[OCRWord]:
    merged = list(primary)
    for candidate in candidates:
        if candidate.confidence is None or not re.search(r"[A-Za-z0-9]", candidate.text):
            continue
        matches = [
            index
            for index, existing in enumerate(merged)
            if _overlap_ratio(existing, candidate) >= 0.5
        ]
        if matches:
            target_index = max(
                matches,
                key=lambda index: _overlap_ratio(merged[index], candidate),
            )
            target = merged[target_index]
            if (
                candidate.confidence >= minimum_confidence
                and candidate.confidence > (target.confidence or -1) + 3
                and _compatible_ocr_tokens(target.text, candidate.text)
            ):
                merged[target_index] = candidate
        elif (
            add_missing
            and candidate.confidence >= minimum_add_confidence
            and not any(_overlap_ratio(existing, candidate) >= 0.2 for existing in primary)
        ):
            merged.append(candidate)
    return merged


def _refine_low_confidence_rate_words(
    pytesseract,
    image: Image.Image,
    words: List[OCRWord],
    config: IngestConfig,
    *,
    deadline: Optional[float] = None,
) -> List[OCRWord]:
    lines = _ocr_lines_from_words(words, config.extra_space_gap_ratio)
    price_rows = []
    for line in lines:
        tokens = {
            token
            for word in line.words
            for token in re.findall(r"[a-z]+", word.text.casefold())
        }
        price_words = [
            word for word in line.words
            if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", word.text.strip())
        ]
        if not tokens.intersection({"ep", "cp", "map", "ap", "bb"}) or not price_words:
            continue
        if any(
            word.confidence is None
            or word.confidence < 50
            or word.width <= 0
            or word.height <= 0
            for word in price_words
        ):
            price_rows.append(line)
    if not price_rows:
        return words

    heights = sorted(word.height for word in words if word.height > 0)
    typical_height = heights[len(heights) // 2] if heights else 20
    padding = max(24, round(typical_height * 3))
    for line in price_rows:
        _check_deadline(deadline, "preparing targeted rate-row OCR")
        left = max(0, min(word.x for word in line.words) - padding)
        top = max(0, min(word.y for word in line.words) - padding)
        right = min(
            image.width,
            max(word.x + word.width for word in line.words) + padding,
        )
        bottom = min(
            image.height,
            max(word.y + word.height for word in line.words) + padding,
        )
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        scale = 2.0
        enlarged = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
        prepared = _PreparedOCRImage(enlarged, left, top, scale)
        data = _image_to_data(
            pytesseract,
            enlarged,
            config,
            11,
            timeout_seconds=_ocr_timeout_seconds(deadline),
        )
        refined_words = _ocr_words_from_data(data, prepared)
        words = _merge_ocr_words(
            words,
            refined_words,
            add_missing=False,
            minimum_confidence=50,
        )
    return words


def _overlap_ratio(first: OCRWord, second: OCRWord) -> float:
    overlap_width = max(
        0,
        min(first.x + first.width, second.x + second.width) - max(first.x, second.x),
    )
    overlap_height = max(
        0,
        min(first.y + first.height, second.y + second.height) - max(first.y, second.y),
    )
    overlap_area = overlap_width * overlap_height
    smaller_area = min(first.width * first.height, second.width * second.height)
    return overlap_area / smaller_area if smaller_area > 0 else 0.0


def _compatible_ocr_tokens(first: str, second: str) -> bool:
    left = re.sub(r"[^a-z0-9]", "", first.lower())
    right = re.sub(r"[^a-z0-9]", "", second.lower())
    if not left or not right:
        return False
    if left.isdigit() and right.isdigit():
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    similarity = difflib.SequenceMatcher(None, left, right).ratio()
    return similarity >= 0.65 or (min(len(left), len(right)) <= 2 and similarity >= 0.5)


def ocr_image(
    image: Image.Image,
    config: IngestConfig,
    *,
    deadline: Optional[float] = None,
) -> OCRResult:
    _check_deadline(deadline, "loading OCR support")
    pytesseract = _load_pytesseract(config, deadline=deadline)
    _check_deadline(deadline, "preparing the OCR image")
    primary_image = _prepare_ocr_image(image, config, crop_content=False)
    _check_deadline(deadline, "running primary OCR")
    primary_data = _image_to_data(
        pytesseract,
        primary_image.image,
        config,
        config.tesseract_psm,
        timeout_seconds=_ocr_timeout_seconds(deadline),
    )
    words = _ocr_words_from_data(primary_data, primary_image)

    _check_deadline(deadline, "preparing refined OCR")
    refined_image = _prepare_ocr_image(image, config, crop_content=True)
    has_content_crop = (
        refined_image.offset_x != 0
        or refined_image.offset_y != 0
        or refined_image.image.size != primary_image.image.size
    )
    if config.ocr_crop_content and has_content_crop:
        _check_deadline(deadline, "running cropped OCR")
        cropped_data = _image_to_data(
            pytesseract,
            refined_image.image,
            config,
            config.tesseract_psm,
            timeout_seconds=_ocr_timeout_seconds(deadline),
        )
        words = _merge_ocr_words(
            words,
            _ocr_words_from_data(cropped_data, refined_image),
            add_missing=True,
            minimum_confidence=60,
            minimum_add_confidence=75,
        )
    else:
        refined_image = primary_image

    if (
        config.ocr_refinement_psm is not None
        and config.ocr_refinement_psm != config.tesseract_psm
    ):
        _check_deadline(deadline, "running refinement OCR")
        refinement_data = _image_to_data(
            pytesseract,
            refined_image.image,
            config,
            config.ocr_refinement_psm,
            timeout_seconds=_ocr_timeout_seconds(deadline),
        )
        words = _merge_ocr_words(
            words,
            _ocr_words_from_data(refinement_data, refined_image),
            add_missing=False,
            minimum_confidence=90,
        )

    words = _refine_low_confidence_rate_words(
        pytesseract,
        image,
        words,
        config,
        deadline=deadline,
    )
    lines = _ocr_lines_from_words(words, config.extra_space_gap_ratio)
    text = "\n".join(line.text for line in lines)
    confidence = _mean_confidence([word.confidence for word in words])
    if not text.strip():
        _check_deadline(deadline, "running text OCR")
        try:
            text = pytesseract.image_to_string(
                primary_image.image,
                lang=config.tesseract_lang,
                config=_tesseract_config(config),
                timeout=_ocr_timeout_seconds(deadline) or 0,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise TesseractUnavailable(
                "Tesseract is not installed or not on PATH. "
                "Install it and optionally set TESSERACT_CMD."
            ) from exc
        except pytesseract.TesseractError as exc:
            raise TesseractUnavailable(
                f"Tesseract could not read this page ({exc}). "
                "Set TESSDATA_DIR to your tessdata folder."
            ) from exc
        except RuntimeError as exc:
            if "tesseract process timeout" in str(exc).casefold():
                raise IngestionTimeoutError(
                    "Tesseract exceeded its OCR timeout; the OCR subprocess was "
                    "terminated and no partial page set was returned."
                ) from exc
            raise
        lines = _text_lines(text)
    return OCRResult(_normalise_text(text), confidence, words, lines)


def _image_to_data(
    pytesseract,
    image: Image.Image,
    config: IngestConfig,
    psm: int,
    *,
    timeout_seconds: Optional[float] = None,
) -> dict:
    try:
        return pytesseract.image_to_data(
            image,
            lang=config.tesseract_lang,
            config=_tesseract_config(config, psm),
            output_type=pytesseract.Output.DICT,
            timeout=timeout_seconds or 0,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractUnavailable(
            "Tesseract is not installed or not on PATH. "
            "Install it and optionally set TESSERACT_CMD."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise TesseractUnavailable(
            f"Tesseract could not read this page ({exc}). "
            "Set TESSDATA_DIR to your tessdata folder."
        ) from exc
    except RuntimeError as exc:
        if "tesseract process timeout" in str(exc).casefold():
            raise IngestionTimeoutError(
                "Tesseract exceeded its OCR timeout; the OCR subprocess was "
                "terminated and no partial page set was returned."
            ) from exc
        raise


def _tesseract_config(config: IngestConfig, psm: Optional[int] = None) -> str:
    parts = [f"--psm {config.tesseract_psm if psm is None else psm}"]
    tessdata = _resolve_tessdata_dir(config)
    if not tessdata or not (Path(tessdata) / "eng.traineddata").is_file():
        location = tessdata or "the configured tessdata directory"
        raise TesseractUnavailable(
            f"English language data (eng.traineddata) was not found in {location}. "
            "Set TESSDATA_DIR to the folder containing eng.traineddata."
        )
    if os.name == "nt":
        tessdata = _windows_short_path(tessdata)
        parts.append(f"--tessdata-dir {tessdata}")
    else:
        parts.append(f'--tessdata-dir "{tessdata}"')
    return " ".join(parts)


def _windows_short_path(path: str) -> str:
    if " " not in path:
        return path
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(path, buffer, len(buffer))
    if not length or length >= len(buffer):
        raise TesseractUnavailable(
            f"Cannot pass tessdata directory with spaces to Tesseract: {path}. "
            "Move it to a path without spaces or enable Windows 8.3 short names."
        )
    return buffer.value


def _load_pytesseract(
    config: IngestConfig,
    *,
    deadline: Optional[float] = None,
):
    import pytesseract

    cmd = config.tesseract_cmd or _guess_tesseract_cmd()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    if deadline is not None:
        _check_deadline(deadline, "checking Tesseract availability")
        return pytesseract
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractUnavailable(
            "Tesseract is not installed or not on PATH. "
            "Install it and optionally set TESSERACT_CMD."
        ) from exc
    return pytesseract


def _resolve_tessdata_dir(config: IngestConfig) -> Optional[str]:
    if config.tessdata_dir:
        return _normalise_tessdata_dir(config.tessdata_dir) or config.tessdata_dir
    return _guess_tessdata_dir()


def _normalise_tessdata_dir(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    path = Path(raw)
    if (path / "eng.traineddata").is_file():
        return str(path)
    nested = path / "tessdata"
    if (nested / "eng.traineddata").is_file():
        return str(nested)
    return str(path) if path.is_dir() else None


def _guess_tessdata_dir() -> Optional[str]:
    candidates = (
        os.environ.get("TESSDATA_DIR"),
        os.environ.get("TESSDATA_PREFIX"),
        os.environ.get("INGEST_TESSDATA_DIR"),
        str(Path(sys.prefix) / "share" / "tessdata"),
        str(Path(sys.prefix) / "share"),
        str(Path(sys.prefix) / "Library" / "share" / "tessdata"),
        r"C:\Program Files\Tesseract-OCR\tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
        "/opt/homebrew/share/tessdata",
    )
    for candidate in candidates:
        resolved = _normalise_tessdata_dir(candidate)
        if resolved and (Path(resolved) / "eng.traineddata").is_file():
            return resolved
    return None


def _guess_tesseract_cmd() -> Optional[str]:
    conda_bin = Path(sys.prefix) / "Library" / "bin" / "tesseract.exe"
    candidates = (
        os.environ.get("TESSERACT_CMD"),
        str(conda_bin),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def tesseract_available(config: Optional[IngestConfig] = None) -> bool:
    cfg = config or IngestConfig.from_env()
    try:
        _load_pytesseract(cfg)
        _tesseract_config(cfg)
    except TesseractUnavailable:
        return False
    return True


def unreadable_pages(pages: Sequence[PageContent]) -> List[PageContent]:
    return [p for p in pages if p.unreadable or not (p.text or "").strip()]


def _ocr_lines_from_words(
    words: Sequence[OCRWord],
    extra_space_gap_ratio: float,
) -> List[PageLine]:
    display_words: List[OCRWord] = []
    ordered = sorted(words, key=lambda word: (word.y + word.height / 2, word.x))
    for word in ordered:
        if (
            word.width < 4
            or word.height < 4
            or not re.search(r"[A-Za-z0-9]", word.text)
        ):
            continue
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(display_words)
                if _overlap_ratio(existing, word) >= 0.7
                and _compatible_ocr_tokens(existing.text, word.text)
            ),
            None,
        )
        if duplicate_index is None:
            display_words.append(word)
        elif (word.confidence or -1) > (display_words[duplicate_index].confidence or -1):
            display_words[duplicate_index] = word

    lines: List[List[OCRWord]] = []
    for word in display_words:
        center_y = word.y + word.height / 2
        best_line: Optional[List[OCRWord]] = None
        best_distance: Optional[float] = None
        for line in lines:
            line_center = sum(item.y + item.height / 2 for item in line) / len(line)
            median_height = sorted(item.height for item in line)[len(line) // 2]
            tolerance = max(8, min(word.height, median_height) * 0.55)
            distance = abs(center_y - line_center)
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best_line = line
                best_distance = distance
        if best_line is None:
            lines.append([word])
        else:
            best_line.append(word)

    lines.sort(
        key=lambda line: sum(item.y + item.height / 2 for item in line) / len(line)
    )
    return [
        PageLine(
            text=_join_line(
                [(word.x, word.width, word.text) for word in line],
                extra_space_gap_ratio,
            ),
            words=sorted(line, key=lambda word: word.x),
        )
        for line in lines
    ]


def _layout_from_ocr_words(words: Sequence[OCRWord], extra_space_gap_ratio: float) -> str:
    return "\n".join(
        line.text for line in _ocr_lines_from_words(words, extra_space_gap_ratio)
    )


def _text_lines(text: str) -> List[PageLine]:
    return [PageLine(text=line) for line in _normalise_text(text).splitlines()]


def _join_line(words: Iterable[tuple[int, int, str]], extra_space_gap_ratio: float) -> str:
    ordered = sorted(words, key=lambda item: item[0])
    if not ordered:
        return ""
    widths = [w for _, w, token in ordered if len(token) > 0]
    avg_char = (sum(widths) / max(sum(len(t) for _, _, t in ordered), 1)) if widths else 10.0
    gap_limit = avg_char * extra_space_gap_ratio
    parts = [ordered[0][2]]
    for prev, current in zip(ordered, ordered[1:]):
        prev_right = prev[0] + prev[1]
        gap = current[0] - prev_right
        if gap > gap_limit * 3:
            parts.append("   ")
        elif gap > gap_limit:
            parts.append("  ")
        else:
            parts.append(" ")
        parts.append(current[2])
    return "".join(parts)


def _mean_confidence(values: Sequence) -> Optional[float]:
    scores = []
    for value in values:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score >= 0:
            scores.append(score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def pages_as_dicts(pages: Sequence[PageContent]) -> List[dict]:
    return [page.to_dict() for page in pages]


def dump_pages(pages: Sequence[PageContent]) -> List[dict]:
    return [page.to_dict() for page in pages]
