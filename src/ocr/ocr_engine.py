"""Движок OCR: распознавание текста на изображениях через pytesseract (Tesseract)."""

from __future__ import annotations

from PIL import Image

import pytesseract


class OcrEngineError(RuntimeError):
    """Исключение при ошибках OCR-движка (Tesseract не установлен, нет языка и т.п.)."""


def ocr_image(
    image: Image.Image, lang: str = "rus+eng", min_conf: float = 30.0
) -> tuple[str, float]:
    """Распознать текст на изображении через Tesseract.

    Args:
        image: Изображение PIL для распознавания.
        lang: Языки распознавания, например "rus+eng".
        min_conf: Минимальная уверенность слова (0-100). Слова с более низкой
            уверенностью не включаются ни в текст, ни в среднюю уверенность.

    Returns:
        Кортеж (распознанный текст, средняя уверенность по словам в процентах).
        Уверенность вычисляется из image_to_data, игнорируя значения -1
        и слова с уверенностью ниже min_conf.
        Если после фильтрации слов не осталось, возвращается ("", 0.0).

    Raises:
        OcrEngineError: Если Tesseract не установлен, не найден указанный язык
            или произошла другая ошибка при распознавании. В сообщении есть
            подсказка, что установить.
    """
    try:
        data = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as error:
        raise OcrEngineError(
            "Tesseract OCR не установлен. Установите системный пакет:\n"
            "  sudo apt install tesseract-ocr\n"
            "  sudo apt install tesseract-ocr-rus   # для русского языка\n"
        ) from error
    except Exception as error:
        msg = str(error)
        if "language" in msg.lower() or "lang" in msg.lower():
            raise OcrEngineError(
                f"Язык OCR '{lang}' не установлен. Установите языковой пакет:\n"
                f"  sudo apt install tesseract-ocr-{lang.split('+')[0]}\n"
                f"Пример для русского: sudo apt install tesseract-ocr-rus\n"
            ) from error
        raise OcrEngineError(
            f"Ошибка при распознавании текста: {error}"
        ) from error

    words: list[tuple[str, float]] = []
    for word, conf in zip(data["text"], data["conf"]):
        if word is None or word.strip() == "":
            continue
        conf_value = float(conf)
        if conf_value < 0:
            continue
        if conf_value < min_conf:
            continue
        words.append((word.strip(), conf_value))

    text = " ".join(word for word, _ in words)
    avg_confidence = (
        sum(conf for _, conf in words) / len(words) if words else 0.0
    )

    return text, avg_confidence
