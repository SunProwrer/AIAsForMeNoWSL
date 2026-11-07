"""Конвейер обработки PDF: текстовый слой или OCR для сканов."""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf as fitz

from .detect import has_text_layer
from .extract import extract_text_page
from .models import DocumentResult, PageResult
from .ocr_engine import ocr_image
from .preprocess import preprocess
from .render import render_page

logger = logging.getLogger(__name__)


def process_pdf(path: str | Path, lang: str = "rus+eng") -> DocumentResult:
    """Обработать PDF-документ: текстовый слой или OCR для сканов.

    Для каждой страницы проверяется наличие текстового слоя через
    ``has_text_layer``. Если слой есть — текст извлекается через
    ``extract_text_page``. Если страниц считается сканом — она рендерится
    через ``render_page``, предобрабатывается через ``preprocess`` и
    распознаётся через ``ocr_image`` с записью ``source="ocr"`` и уверенности.

    Ошибка на одной странице не ломает весь документ: записывается пустой
    ``PageResult`` и выводится предупреждение в лог.

    Args:
        path: Путь к PDF-файлу.
        lang: Языки OCR, например "rus+eng".

    Returns:
        DocumentResult с результатами по всем страницам.

    Raises:
        FileNotFoundError: Если файл не найден.
        ValueError: Если путь не к файлу, не PDF или файл повреждён.
    """
    pdf_path = Path(path).expanduser()

    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Нужно указать путь к файлу, а не к папке: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Допускаются только файлы с расширением .pdf.")

    try:
        doc = fitz.open(pdf_path)
    except Exception as error:
        raise ValueError(f"Повреждённый PDF-файл: {pdf_path}") from error

    pages: list[PageResult] = []
    for page_index in range(len(doc)):
        page_number = page_index + 1
        page = doc[page_index]

        try:
            if has_text_layer(page, min_chars=30):
                pages.append(extract_text_page(page, page_number=page_number))
            else:
                image = render_page(page)
                processed = preprocess(image)
                text, confidence = ocr_image(processed, lang=lang)
                pages.append(
                    PageResult(
                        page_number=page_number,
                        text=text,
                        source="ocr",
                        confidence=confidence,
                        char_count=len(text),
                    )
                )
        except Exception as error:
            logger.warning(
                "Ошибка обработки страницы %d: %s", page_number, error
            )
            pages.append(
                PageResult(
                    page_number=page_number,
                    text="",
                    source="ocr",
                    confidence=None,
                    char_count=0,
                )
            )

    doc.close()

    return DocumentResult(file_name=pdf_path.name, pages=pages)


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Обработать PDF: текстовый слой или OCR для сканов."
    )
    parser.add_argument("path", help="Путь к PDF-файлу.")
    parser.add_argument(
        "--out", help="Путь к выходному JSON-файлу (по умолчанию в stdout)."
    )
    parser.add_argument(
        "--lang", default="rus+eng", help="Языки OCR (по умолчанию rus+eng)."
    )
    args = parser.parse_args()

    try:
        result = process_pdf(args.path, lang=args.lang)
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1

    output = result.to_json(indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
