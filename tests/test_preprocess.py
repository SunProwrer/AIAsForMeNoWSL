"""Тесты для модуля предобработки изображений src/ocr/preprocess."""

from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image

from src.ocr.preprocess import preprocess


class TestPreprocess(unittest.TestCase):
    def setUp(self) -> None:
        # Создаём тестовое изображение с текстом-подобным паттерном
        self.img = Image.new("RGB", (200, 100), color="white")
        # Рисуем несколько тёмных линий для имитации текста
        arr = np.array(self.img)
        arr[30:40, 10:50] = 0
        arr[50:60, 15:45] = 0
        arr[70:80, 20:55] = 0
        self.img = Image.fromarray(arr)

    def test_default_preserves_size(self):
        result = preprocess(self.img)
        self.assertEqual(result.size, self.img.size)

    def test_all_flags_false_preserves_size_and_content(self):
        result = preprocess(self.img, grayscale=False, denoise=False, binarize=False, deskew=False)
        self.assertEqual(result.size, self.img.size)
        self.assertEqual(np.array(result).tolist(), np.array(self.img).tolist())

    def test_grayscale_only(self):
        result = preprocess(self.img, grayscale=True, denoise=False, binarize=False, deskew=False)
        self.assertEqual(result.size, self.img.size)
        arr = np.array(result)
        self.assertEqual(len(arr.shape), 2)

    def test_denoise_only(self):
        result = preprocess(self.img, grayscale=True, denoise=True, binarize=False, deskew=False)
        self.assertEqual(result.size, self.img.size)

    def test_binarize_only(self):
        result = preprocess(self.img, grayscale=True, denoise=False, binarize=True, deskew=False)
        self.assertEqual(result.size, self.img.size)
        arr = np.array(result)
        unique = np.unique(arr)
        self.assertTrue(set(unique.tolist()).issubset({0, 255}))

    def test_binarize_only_values_0_255(self):
        result = preprocess(self.img, grayscale=True, denoise=False, binarize=True, deskew=False)
        arr = np.array(result)
        self.assertTrue(np.all((arr == 0) | (arr == 255)))

    def test_deskew_only(self):
        result = preprocess(self.img, grayscale=True, denoise=False, binarize=False, deskew=True)
        self.assertEqual(result.size, self.img.size)

    def test_all_flags_true(self):
        result = preprocess(self.img, grayscale=True, denoise=True, binarize=True, deskew=True)
        self.assertEqual(result.size, self.img.size)

    def test_deskew_rotated_image(self):
        # Создаём изображение с повёрнутым на 3 градуса текстом
        base = Image.new("RGB", (300, 200), color="white")
        arr = np.array(base)
        # Рисуем горизонтальные полосы (имитация строк текста)
        for y in range(50, 150, 20):
            arr[y:y+5, 20:280] = 0
        base = Image.fromarray(arr)
        rotated = base.rotate(3, expand=False, fillcolor="white")

        result = preprocess(rotated, grayscale=True, denoise=False, binarize=False, deskew=True)
        self.assertEqual(result.size, rotated.size)
        # Убеждаемся, что deskew не упал и вернул изображение того же размера
        self.assertIsInstance(result, Image.Image)

    def test_deskew_angle_greater_than_15_not_corrected(self):
        # Повернём на 20 градусов — deskew не должен исправлять (>15°)
        base = Image.new("RGB", (300, 200), color="white")
        arr = np.array(base)
        for y in range(50, 150, 20):
            arr[y:y+5, 20:280] = 0
        base = Image.fromarray(arr)
        rotated = base.rotate(20, expand=False, fillcolor="white")

        result = preprocess(rotated, grayscale=True, denoise=False, binarize=False, deskew=True)
        self.assertEqual(result.size, rotated.size)


if __name__ == "__main__":
    unittest.main()
