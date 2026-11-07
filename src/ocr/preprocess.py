"""Предобработка изображений перед OCR через OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def preprocess(
    image: Image.Image,
    *,
    grayscale: bool = True,
    denoise: bool = True,
    binarize: bool = True,
    deskew: bool = True,
) -> Image.Image:
    """Предобработка изображения перед OCR.

    Применяет шаги по порядку: серый -> шумоподавление -> бинаризация ->
    выравнивание наклона. Каждый шаг можно отключить флагом. Размер
    изображения не меняется.

    Args:
        image: Исходное изображение PIL.
        grayscale: Перевести в оттенки серого.
        denoise: Убрать шум (fastNlMeansDenoising / медианный фильтр).
        binarize: Бинаризовать (Otsu или адаптивный порог).
        deskew: Выровнять наклон (minAreaRect по тёмным пикселям, углы
            больше 15° не исправляются).

    Returns:
        Обработанное изображение PIL того же размера.

    Raises:
        ValueError: Если изображение пустое или некорректное.
    """
    if image is None or image.size[0] == 0 or image.size[1] == 0:
        raise ValueError("Изображение пустое или некорректное.")

    img = np.array(image)

    if grayscale:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            pass
        else:
            raise ValueError(f"Некорректное количество каналов: {img.shape}")

    if denoise:
        if len(img.shape) == 2:
            img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        else:
            img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    if binarize:
        if len(img.shape) == 2:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if deskew:
        img = _deskew(img)

    return Image.fromarray(img)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Выровнять наклон изображения через оценку угла minAreaRect.

    Углы больше 15° не исправляются. Размер изображения сохраняется.

    Args:
        img: Изображение (серое или бинарное) в виде numpy массива.

    Returns:
        Выровненное изображение того же размера.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Бинаризуем для поиска тёмных пикселей (текста)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return img

    rect = cv2.minAreaRect(coords)
    angle = rect[2]

    # OpenCV minAreaRect возвращает угол в диапазоне [-90, 0]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 15:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if len(img.shape) == 2:
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    return img
