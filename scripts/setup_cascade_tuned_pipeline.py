"""Split cascade notebook into original + tuned and clone downstream notebooks for A/B runs."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"

# Rutas Colab para notebooks *_tuned (Drive → Other computers sync)
COLAB_PROJECT_ROOT = (
    "/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation-Yeisson-work"
)
COLAB_DATASET_ROOT = (
    f"{COLAB_PROJECT_ROOT}/data/ScoliosisDataSetYeisson"
)
COLAB_PATH_REPLACEMENTS: list[tuple[str, str]] = [
    ("/content/drive/MyDrive/DataRadriografias", COLAB_PROJECT_ROOT),
    ("ROOT / 'data' / 'Scoliosis_Dataset'", f'Path("{COLAB_DATASET_ROOT}")'),
    ("DATASET_ROOT = ROOT / 'data' / 'Scoliosis_Dataset'", f'DATASET_ROOT = Path("{COLAB_DATASET_ROOT}")'),
    (
        "search_roots = [ROOT, ROOT / 'data', ROOT / 'reports']",
        "search_roots = [ROOT, ROOT / 'data' / 'ScoliosisDataSetYeisson', ROOT / 'data', ROOT / 'reports']",
    ),
    (
        "INDEX_PATH = ROOT / 'indice_dataset.csv'",
        "INDEX_PATH = ROOT / 'data' / 'ScoliosisDataSetYeisson' / 'indice_dataset.csv'",
    ),
    (
        "DICT_PATH = ROOT / 'diccionario_etiquetas_T1_T12_L1_L5.json'",
        "DICT_PATH = ROOT / 'data' / 'ScoliosisDataSetYeisson' / 'diccionario_etiquetas_T1_T12_L1_L5.json'",
    ),
]

CASCADE_ORIG = NB_DIR / "03_colab_train_spine_cascade_binary_to_thoracolumbar_explained.ipynb"
CASCADE_TUNED = NB_DIR / "03_colab_train_spine_cascade_binary_to_thoracolumbar_explained_tuned.ipynb"

DOWNSTREAM = [
    "04_colab_infer_analyze_thoracolumbar_predictions_explained.ipynb",
    "05_colab_postprocess_anatomical_thoracolumbar_v2_explained.ipynb",
    "06_colab_train_visible_range_estimator_and_clip_thoracolumbar_explained.ipynb",
    "07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained.ipynb",
    "08_colab_final_inference_pipeline_thoracolumbar_explained.ipynb",
    "09_colab_final_project_summary_thoracolumbar_explained.ipynb",
    "10_colab_hard_case_mining_and_refined_sampling_thoracolumbar_explained.ipynb",
    "11_colab_definitive_pipeline_decision_support_thoracolumbar_explained.ipynb",
    "12_colab_refined_retraining_last_visible_thoracolumbar_explained.ipynb",
    "13_colab_hybrid_conservative_clipping_policy_thoracolumbar_explained.ipynb",
    "14_colab_final_deployment_policy_thoracolumbar_explained.ipynb",
]

# Longer keys first to avoid partial replacements.
TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    (
        "last_visible_estimator_thoracolumbar_refined_best.pt",
        "last_visible_estimator_thoracolumbar_refined_tuned_best.pt",
    ),
    (
        "thoracolumbar_partial_cascade_explained_best.pt",
        "thoracolumbar_partial_cascade_explained_tuned_best.pt",
    ),
    (
        "visible_range_estimator_thoracolumbar_best.pt",
        "visible_range_estimator_thoracolumbar_tuned_best.pt",
    ),
    (
        "last_visible_estimator_thoracolumbar_best.pt",
        "last_visible_estimator_thoracolumbar_tuned_best.pt",
    ),
    (
        "training_runs_cascade_thoracolumbar_explained",
        "training_runs_cascade_thoracolumbar_explained_tuned",
    ),
    (
        "thoracolumbar_postprocess_anatomical_v2_explained",
        "thoracolumbar_postprocess_anatomical_v2_explained_tuned",
    ),
    (
        "thoracolumbar_inference_analysis_explained",
        "thoracolumbar_inference_analysis_explained_tuned",
    ),
    (
        "last_visible_estimator_thoracolumbar_refined_explained",
        "last_visible_estimator_thoracolumbar_refined_explained_tuned",
    ),
    (
        "visible_range_estimator_thoracolumbar_explained",
        "visible_range_estimator_thoracolumbar_explained_tuned",
    ),
    (
        "last_visible_estimator_thoracolumbar_explained",
        "last_visible_estimator_thoracolumbar_explained_tuned",
    ),
    (
        "hard_case_mining_thoracolumbar_explained",
        "hard_case_mining_thoracolumbar_explained_tuned",
    ),
    (
        "definitive_pipeline_decision_support_thoracolumbar",
        "definitive_pipeline_decision_support_thoracolumbar_tuned",
    ),
    (
        "hybrid_conservative_clipping_policy_thoracolumbar",
        "hybrid_conservative_clipping_policy_thoracolumbar_tuned",
    ),
    (
        "final_inference_pipeline_thoracolumbar",
        "final_inference_pipeline_thoracolumbar_tuned",
    ),
    (
        "final_deployment_policy_thoracolumbar",
        "final_deployment_policy_thoracolumbar_tuned",
    ),
    (
        "final_project_summary_thoracolumbar",
        "final_project_summary_thoracolumbar_tuned",
    ),
    (
        "cascade_explained_best.pt",
        "cascade_explained_tuned_best.pt",
    ),
]


def apply_tuned_paths(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    for old, new in COLAB_PATH_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def tuned_notebook_name(name: str) -> str:
    if name.endswith("_explained.ipynb"):
        return name.replace("_explained.ipynb", "_explained_tuned.ipynb")
    raise ValueError(f"Unexpected notebook name: {name}")


def patch_cascade_tuned_title(nb: dict) -> None:
    src = "".join(nb["cells"][0].get("source", []))
    if "Tuned" not in src and src.startswith("# Cascada"):
        src = src.replace(
            "# Cascada Thoracolumbar Mejorada y Explicada - Colab\n",
            "# Cascada Thoracolumbar Mejorada y Explicada (Tuned) - Colab\n",
            1,
        )
        lines = src.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        nb["cells"][0]["source"] = lines


def main() -> None:
    if not CASCADE_ORIG.exists():
        raise FileNotFoundError(CASCADE_ORIG)

    # 1) Preserve current (Fase 3+4) notebook as _tuned
    shutil.copy2(CASCADE_ORIG, CASCADE_TUNED)
    tuned_nb = json.loads(CASCADE_TUNED.read_text(encoding="utf-8"))
    tuned_text = json.dumps(tuned_nb, ensure_ascii=False)
    tuned_text = apply_tuned_paths(tuned_text)
    tuned_nb = json.loads(tuned_text)
    patch_cascade_tuned_title(tuned_nb)
    CASCADE_TUNED.write_text(json.dumps(tuned_nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", CASCADE_TUNED.name)

    # 2) Restore original cascade from remote Yeisson (sin Fase 3/4)
    subprocess.run(
        ["git", "checkout", "origin/Yeisson", "--", str(CASCADE_ORIG.relative_to(REPO))],
        cwd=REPO,
        check=True,
    )
    print("Restored", CASCADE_ORIG.name, "from origin/Yeisson")

    # 3) Clone downstream notebooks
    for name in DOWNSTREAM:
        src = NB_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst_name = tuned_notebook_name(name)
        dst = NB_DIR / dst_name
        shutil.copy2(src, dst)
        nb = json.loads(dst.read_text(encoding="utf-8"))
        raw = json.dumps(nb, ensure_ascii=False)
        raw = apply_tuned_paths(raw)
        nb = json.loads(raw)
        dst.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Wrote", dst_name)


if __name__ == "__main__":
    main()
