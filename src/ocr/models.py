"""Модели результатов извлечения текста из PDF."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PageResult:
    """Результат обработки одной страницы PDF.

    Attributes:
        page_number: Номер страницы (нумерация с 1).
        text: Извлечённый текст страницы.
        source: Источник текста — "text_layer" (текстовый слой) или "ocr" (распознавание).
        confidence: Уверенность OCR (0-100), None для текстового слоя.
        char_count: Количество символов в тексте.
    """

    page_number: int
    text: str
    source: Literal["text_layer", "ocr"]
    confidence: float | None = None
    char_count: int = 0

    def __post_init__(self) -> None:
        if self.char_count == 0:
            object.__setattr__(self, "char_count", len(self.text))


@dataclass
class DocumentResult:
    """Результат обработки всего PDF-документа.

    Attributes:
        file_name: Имя исходного файла.
        pages: Список результатов по страницам.
    """

    file_name: str
    pages: list[PageResult] = field(default_factory=list)

    def full_text(self) -> str:
        """Вернуть текст всех страниц, разделённый пустыми строками."""
        return "\n\n".join(page.text for page in self.pages)

    def to_dict(self) -> dict:
        """Преобразовать результат в словарь."""
        return {
            "file_name": self.file_name,
            "pages": [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "source": p.source,
                    "confidence": p.confidence,
                    "char_count": p.char_count,
                }
                for p in self.pages
            ],
        }

    def to_json(self, **kwargs) -> str:
        """Сериализовать результат в JSON-строку.

        Принимает те же аргументы, что и ``json.dumps``.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
