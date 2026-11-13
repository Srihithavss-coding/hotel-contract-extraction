import re
import os
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

import ingest
from ingest import (
    IngestConfig,
    IngestionTimeoutError,
    TesseractUnavailable,
    _image_to_data,
    _tesseract_config,
    dump_pages,
    ingest_folder,
    ingest_folder_bounded,
    ingest_pdf,
    tesseract_available,
    unreadable_pages,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def _price_tokens(text: str) -> str:
    return text.replace(",", "").replace(" ", "")


def _line_tokens(line: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+|[\d,]+", line)]


def test_c01_reads_selectable_pdf_text():
    pages = ingest_folder(CONTRACTS / "c01")
    assert [p.file for p in pages] == ["rate-letter.pdf", "rate-letter.pdf"]
    assert [p.page for p in pages] == [1, 2]
    assert all(p.extraction_method == "text" for p in pages)
    assert all(p.ocr_confidence is None for p in pages)
    evidence = pages[0].to_dict()
    assert evidence["text"] == pages[0].text
    assert evidence["lines"]
    assert evidence["ocr_words"] == []
    assert evidence["file"] == pages[0].file
    assert evidence["page"] == pages[0].page
    combined = "\n".join(p.text for p in pages)
    assert "Hotel Saffron Courtyard" in combined
    assert "Heritage Room" in pages[0].text
    assert "3,200" in pages[0].text or "3200" in pages[0].text
    assert "Cancellation" in pages[1].text
    assert unreadable_pages(pages) == []


def test_c02_detects_scanned_page_and_recovers_rate_table():
    if not tesseract_available():
        pytest.skip("Tesseract is not installed")

    pages = ingest_folder(CONTRACTS / "c02")
    assert len(pages) == 2
    page1, page2 = pages
    assert page1.file == page2.file == "rate-letter.pdf"
    assert page1.page == 1
    assert page2.page == 2
    assert page1.extraction_method == "text"
    assert "Coral Sands Resort" in page1.text
    assert page2.extraction_method == "ocr"
    assert page2.ocr_confidence is not None
    assert page2.ocr_confidence >= 0
    compact = _price_tokens(page2.text)
    assert "Garden Villa" in page2.text
    assert "Sea View Villa" in page2.text
    assert "7000" in compact
    assert "7800" in compact
    lines = [_line_tokens(line) for line in page2.text.splitlines()]
    expected_rows = [
        ["garden", "villa", "cp", "7,000", "7,800"],
        ["garden", "villa", "map", "8,200", "10,200"],
        ["sea", "view", "villa", "cp", "9,500", "10,300"],
        ["sea", "view", "villa", "map", "10,700", "12,700"],
        ["garden", "villa", "cp", "4,800", "5,400"],
        ["garden", "villa", "map", "6,000", "7,800"],
        ["sea", "view", "villa", "cp", "6,500", "7,100"],
        ["sea", "view", "villa", "map", "7,700", "9,500"],
    ]
    assert all(row in lines for row in expected_rows)
    assert [
        "peak", "season", "1", "october", "2026", "to", "15", "january", "2027"
    ] in lines
    assert [
        "lean", "season", "16", "january", "2027", "to", "30", "september", "2027"
    ] in lines
    assert any("single" in line and "double" in line for line in lines)
    assert page2.ocr_words
    assert all(word.x >= 0 and word.y >= 0 for word in page2.ocr_words)
    assert all(word.width >= 0 and word.height >= 0 for word in page2.ocr_words)
    assert any(word.confidence is not None for word in page2.ocr_words)
    evidence = page2.to_dict()
    assert evidence["text"] == page2.text
    assert evidence["file"] == "rate-letter.pdf"
    assert evidence["page"] == 2
    assert evidence["ocr_confidence"] == page2.ocr_confidence
    assert len(evidence["ocr_words"]) == len(page2.ocr_words)
    assert all(
        {"text", "x", "y", "width", "height", "confidence"} <= word.keys()
        for word in evidence["ocr_words"]
    )
    structured_rows = [line for line in evidence["lines"] if "7,000" in line["text"]]
    assert len(structured_rows) == 1
    assert _line_tokens(
        " ".join(word["text"] for word in structured_rows[0]["words"])
    ) == ["garden", "villa", "cp", "7,000", "7,800"]
    assert dump_pages([page2])[0]["ocr_words"] == evidence["ocr_words"]
    assert unreadable_pages(pages) == []


def test_c03_reads_both_pdfs_independently():
    pages = ingest_folder(CONTRACTS / "c03")
    files = [p.file for p in pages]
    assert files.count("email.pdf") >= 1
    assert files.count("rate-letter.pdf") >= 1
    by_file = {p.file: p for p in pages if p.page == 1}
    assert "Misty Ridge Retreat" in by_file["rate-letter.pdf"].text
    assert "Extra bed" in by_file["email.pdf"].text
    assert all(p.extraction_method == "text" for p in pages)
    assert {p.page for p in pages if p.file == "rate-letter.pdf"} == {1, 2}
    assert {p.page for p in pages if p.file == "email.pdf"} == {1}


def test_c05_reads_all_seven_pages_with_numbers():
    pages = ingest_folder(CONTRACTS / "c05")
    assert [p.file for p in pages] == ["contract.pdf"] * 7
    assert [p.page for p in pages] == [1, 2, 3, 4, 5, 6, 7]
    assert all(p.extraction_method == "text" for p in pages)
    assert "Hotel Banyan Grove" in pages[0].text
    assert "Contracted net rates" in pages[3].text
    assert "Commercial terms" in pages[5].text
    assert "Force majeure" in pages[6].text


def test_ocr_thresholds_are_configurable():
    config = IngestConfig(min_text_chars=10_000, min_alnum_ratio=1.0)
    if tesseract_available():
        pages = ingest_folder(CONTRACTS / "c01", config=config)
        assert all(p.extraction_method == "ocr" for p in pages)
    else:
        with pytest.raises(TesseractUnavailable):
            ingest_folder(CONTRACTS / "c01", config=config)


def test_tesseract_config_requires_english_data(tmp_path):
    with pytest.raises(TesseractUnavailable, match="eng.traineddata"):
        _tesseract_config(IngestConfig(tessdata_dir=str(tmp_path)))

    traineddata = tmp_path / "eng.traineddata"
    traineddata.touch()
    config = _tesseract_config(IngestConfig(tessdata_dir=str(tmp_path)))
    assert "--tessdata-dir" in config
    if os.name == "nt":
        assert '"' not in config
    else:
        assert f'--tessdata-dir "{tmp_path}"' in config


def test_tesseract_paths_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("TESSERACT_CMD", r"C:\tools\tesseract.exe")
    monkeypatch.setenv("TESSDATA_DIR", r"C:\tools\tessdata")

    config = IngestConfig.from_env()

    assert config.tesseract_cmd == r"C:\tools\tesseract.exe"
    assert config.tessdata_dir == r"C:\tools\tessdata"


def test_ocr_required_but_unavailable_raises(monkeypatch):
    def unavailable(image, config, *, deadline=None):
        raise TesseractUnavailable("Tesseract unavailable")

    monkeypatch.setattr(ingest, "ocr_image", unavailable)
    with pytest.raises(TesseractUnavailable, match="Tesseract unavailable"):
        ingest_folder(
            CONTRACTS / "c01",
            config=IngestConfig(min_text_chars=10_000, min_alnum_ratio=1.0),
        )


def test_tesseract_timeout_is_bounded_and_reported_as_ingestion_timeout(
    monkeypatch,
):
    captured = {}

    class FakePytesseract:
        Output = SimpleNamespace(DICT="dict")
        TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
        TesseractError = type("TesseractError", (Exception,), {})

        @staticmethod
        def image_to_data(image, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(ingest, "_tesseract_config", lambda config, psm: "test")

    with pytest.raises(IngestionTimeoutError, match="OCR timeout"):
        _image_to_data(
            FakePytesseract,
            image=None,
            config=IngestConfig(),
            psm=11,
            timeout_seconds=3.0,
        )

    assert captured["timeout"] == 3.0
    assert captured["output_type"] == "dict"


def test_deadline_is_checked_between_pdf_pages(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        ingest, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    processed_pages = []

    def ingest_page(doc, index, filename, config, document_id="", *, deadline=None):
        processed_pages.append(index)
        clock[0] = 6.0
        return ingest.PageContent(
            file=filename,
            page=index + 1,
            text="page",
            extraction_method="text",
            ocr_confidence=None,
            document_id=document_id,
        )

    monkeypatch.setattr(ingest, "_ingest_page", ingest_page)

    with pytest.raises(IngestionTimeoutError, match="page 2"):
        ingest_pdf(
            CONTRACTS / "c01" / "rate-letter.pdf",
            deadline=5.0,
        )

    assert processed_pages == [0]


def test_bounded_ingestion_worker_returns_only_after_all_pages():
    pages = ingest_folder_bounded(
        CONTRACTS / "c01",
        deadline=time.monotonic() + 15,
    )

    assert [page.page for page in pages] == [1, 2]
    assert all(page.text.strip() for page in pages)


def test_bounded_ingestion_worker_applies_deadline_to_scanned_pages():
    if not tesseract_available():
        pytest.skip("Tesseract is not installed")

    pages = ingest_folder_bounded(
        CONTRACTS / "c02",
        deadline=time.monotonic() + 60,
    )

    assert len(pages) == 2
    assert pages[1].extraction_method == "ocr"
    assert pages[1].text.strip()


def test_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        ingest_folder(CONTRACTS / "does-not-exist")
