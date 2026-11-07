"""Извлечение текста из PDF с текстовым слоем через библиотеку pypdf."""

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def extract_text(pdf_path: str | Path) -> str:
    """Извлечь текст из PDF и вернуть его в виде одной строки.

    Страницы разделяются пустой строкой. Если в PDF нет текстового слоя
    (текста нет или почти нет), выбрасывается ValueError с пояснением,
    чтобы позже можно было подключить OCR.

    Args:
        pdf_path: путь к PDF-файлу.

    Returns:
        Извлечённый текст, страницы разделены пустой строкой.

    Raises:
        FileNotFoundError: если файл не найден.
        ValueError: если файл не PDF, повреждён или не содержит текстового слоя.
        PdfReadError: если PDF повреждён и не может быть прочитан.
    """
    path = Path(pdf_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Нужно указать путь к файлу, а не к папке: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Допускаются только файлы с расширением .pdf.")

    try:
        reader = PdfReader(path)
    except PdfReadError as error:
        raise ValueError(f"Повреждённый PDF-файл: {path}") from error

    pages_text = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages_text.append(page_text.strip())

    text = "\n\n".join(pages_text)

    if not text.strip() or not any(ch.isalnum() for ch in text):
        raise ValueError(
            f"PDF не содержит текстового слоя: {path}. "
            "Текст не извлёкся, потребуется OCR для распознавания."
        )

    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Извлечь текст из PDF с текстовым слоем и вывести в терминал."
    )
    parser.add_argument("path", help="Путь к PDF-файлу.")
    args = parser.parse_args()

    try:
        text = extract_text(args.path)
    except (OSError, ValueError) as error:
        print(f"Ошибка извлечения текста: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nОтменено.", file=sys.stderr)
        return 130

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
