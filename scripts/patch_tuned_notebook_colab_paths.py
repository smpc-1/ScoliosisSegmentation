"""Actualiza rutas de Colab en notebooks *_tuned (proyecto + dataset en Drive)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"

OLD_PROJECT = "/content/drive/MyDrive/DataRadriografias"
NEW_PROJECT = "/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation-Yeisson-work"
NEW_DATASET = (
    "/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation-Yeisson-work"
    "/data/ScoliosisDataSetYeisson"
)

REPLACEMENTS: list[tuple[str, str]] = [
    (OLD_PROJECT, NEW_PROJECT),
    ("ROOT / 'data' / 'Scoliosis_Dataset'", f'Path("{NEW_DATASET}")'),
    ('ROOT / "data" / "Scoliosis_Dataset"', f'Path("{NEW_DATASET}")'),
    ("DATASET_ROOT = ROOT / 'data' / 'Scoliosis_Dataset'", f'DATASET_ROOT = Path("{NEW_DATASET}")'),
    ('DATASET_ROOT = ROOT / "data" / "Scoliosis_Dataset"', f'DATASET_ROOT = Path("{NEW_DATASET}")'),
    # Metadatos del dataset (indice, diccionario) viven en data/ScoliosisDataSetYeisson/
    (
        "search_roots = [ROOT, ROOT / 'data', ROOT / 'reports']",
        "search_roots = [ROOT, ROOT / 'data' / 'ScoliosisDataSetYeisson', ROOT / 'data', ROOT / 'reports']",
    ),
    (
        "search_roots = [ROOT, DATASET_ROOT, ROOT / 'data', ROOT / 'reports']",
        "search_roots = [ROOT, DATASET_ROOT, ROOT / 'data' / 'ScoliosisDataSetYeisson', ROOT / 'data', ROOT / 'reports']",
    ),
    (
        "INDEX_PATH = ROOT / 'data' / 'indice_dataset.csv'",
        "INDEX_PATH = ROOT / 'data' / 'ScoliosisDataSetYeisson' / 'indice_dataset.csv'",
    ),
    (
        "DICT_PATH = ROOT / 'data' / 'diccionario_etiquetas_T1_T12_L1_L5.json'",
        "DICT_PATH = ROOT / 'data' / 'ScoliosisDataSetYeisson' / 'diccionario_etiquetas_T1_T12_L1_L5.json'",
    ),
    ("INDEX_PATH = ROOT / 'indice_dataset.csv'", "INDEX_PATH = ROOT / 'data' / 'ScoliosisDataSetYeisson' / 'indice_dataset.csv'"),
    (
        "DICT_PATH = ROOT / 'diccionario_etiquetas_T1_T12_L1_L5.json'",
        "DICT_PATH = ROOT / 'data' / 'ScoliosisDataSetYeisson' / 'diccionario_etiquetas_T1_T12_L1_L5.json'",
    ),
]


def patch_notebook(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    changes = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") not in ("code", "markdown"):
            continue
        src = cell.get("source", [])
        if not src:
            continue
        text = "".join(src)
        new_text = text
        for old, new in REPLACEMENTS:
            if old in new_text:
                new_text = new_text.replace(old, new)
                changes += 1
        if new_text != text:
            if isinstance(src, list):
                cell["source"] = new_text.splitlines(keepends=True)
                if cell["source"] and not cell["source"][-1].endswith("\n"):
                    cell["source"][-1] += "\n"
            else:
                cell["source"] = new_text
    if changes:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changes


def main() -> None:
    extra = [
        NB_DIR / "01_colab_thoracolumbar_coverage_strategy_clean.ipynb",
        NB_DIR / "02_colab_train_spine_binary_and_thoracolumbar.ipynb",
    ]
    notebooks = sorted(NB_DIR.glob("*_tuned.ipynb")) + [p for p in extra if p.exists()]
    if not notebooks:
        raise SystemExit(f"No hay notebooks para parchear en {NB_DIR}")
    total = 0
    for nb in notebooks:
        n = patch_notebook(nb)
        total += n
        print(f"{'OK' if n else 'skip':4} {nb.name} ({n} reemplazos en celdas)")
    print(f"\nListo: {len(notebooks)} notebooks, {total} celdas modificadas.")


if __name__ == "__main__":
    main()
