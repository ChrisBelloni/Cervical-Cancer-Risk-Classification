"""Utilitarios simples para o pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_directory(path: Path | str) -> Path:
    """Cria um diretorio caso ele ainda nao exista."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: Path | str, data: dict[str, Any]) -> None:
    """Salva um dicionario em formato JSON."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_text(path: Path | str, content: str) -> None:
    """Salva texto em UTF-8."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(content)


def log_message(message: str) -> None:
    """Imprime uma mensagem padronizada de execucao."""
    print(f"[INFO] {message}")
