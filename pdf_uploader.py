"""Приём PDF-файлов в локальное хранилище проекта без сторонних библиотек."""

import argparse
from contextlib import suppress
from pathlib import Path
import shutil
import sys


DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent / "documents"


def upload_pdf(
    file_path: str | Path,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
) -> Path:
    """Скопировать PDF в хранилище и вернуть полный путь к сохранённому файлу.

    Проверяются расширение и сигнатура PDF, но не вся структура документа.
    Существующие файлы не перезаписываются. Исходный файл сохраняется.
    """
    if not str(file_path).strip():
        raise ValueError("Путь к PDF-файлу не указан.")

    source = Path(file_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Файл не найден: {source}")
    if not source.is_file():
        raise ValueError(f"Нужно указать путь к файлу, а не к папке: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError("Допускаются только файлы с расширением .pdf.")

    with source.open("rb") as source_file:
        if source_file.read(5) != b"%PDF-":
            raise ValueError("Файл пуст или не содержит стандартного заголовка PDF.")
        source_file.seek(0)

        destination = Path(storage_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        target = destination / source.name
        number = 1
        while True:
            try:
                # Режим xb создаёт новый файл и запрещает перезапись существующего.
                target_file = target.open("xb")
                break
            except FileExistsError:
                target = destination / f"{source.stem}_{number}{source.suffix}"
                number += 1

        try:
            with target_file:
                # Копируем порциями, чтобы не загружать весь документ в память.
                shutil.copyfileobj(source_file, target_file)
        except BaseException:
            # При ошибке или отмене удаляем незавершённую копию.
            with suppress(OSError):
                target.unlink()
            raise

    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сохранить PDF-файл в папку документов проекта."
    )
    parser.add_argument(
        "path", nargs="?", help="Путь к PDF; если не указан, программа запросит его."
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=DEFAULT_STORAGE_DIR,
        help="Папка для загрузки (по умолчанию documents рядом с программой).",
    )
    args = parser.parse_args()

    try:
        file_path = args.path
        if file_path is None:
            file_path = input("Введите путь к PDF-файлу: ").strip()
        saved_path = upload_pdf(file_path, args.storage_dir)
    except (OSError, ValueError) as error:
        print(f"Ошибка загрузки: {error}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nЗагрузка отменена.", file=sys.stderr)
        return 130

    print(f"PDF успешно загружен: {saved_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
