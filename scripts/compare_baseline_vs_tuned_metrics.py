"""
Extrae métricas clave de la línea base (Yeisson) y la línea tuneada (Fase 3+4)
y genera tablas comparativas en reports/.

Uso:
  python scripts/compare_baseline_vs_tuned_metrics.py
  python scripts/compare_baseline_vs_tuned_metrics.py --root "D:/DataRadriografias"
  python scripts/compare_baseline_vs_tuned_metrics.py --baseline-only

Los CSV deben vivir bajo <root>/analysis_outputs/ o <root>/outputs/
(con la misma estructura que generan los notebooks 03–14).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "baseline_vs_tuned"
SUBSET = "partial"

PRIMARY_METRICS = [
    "test_macro_dice_fg",
    "test_macro_iou_fg",
    "test_macro_dice_thoracic",
    "test_macro_dice_lumbar",
    "best_val_macro_dice_fg",
]


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    notebook: str
    label: str
    baseline_dir: str
    tuned_dir: str
    files: tuple[str, ...]


FINAL_VARIANT_METRIC_COLS = (
    "macro_dice_fg",
    "macro_iou_fg",
    "macro_dice_lumbar",
    "mean_extra_count",
    "mean_missing_count",
)

STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "02_binary",
        "02",
        "Modelo binario (localización columna)",
        "training_runs_binary_thoracolumbar",
        "training_runs_binary_thoracolumbar",
        ("binary_spine_test_metrics.csv",),
    ),
    StageSpec(
        "03_cascade",
        "03",
        "Cascada multiclase (entrenamiento)",
        "training_runs_cascade_thoracolumbar_explained",
        "training_runs_cascade_thoracolumbar_explained_tuned",
        (
            f"thoracolumbar_{SUBSET}_experiment_summary.csv",
            f"thoracolumbar_{SUBSET}_test_metrics.csv",
        ),
    ),
    StageSpec(
        "04_inference",
        "04",
        "Inferencia / análisis test",
        "thoracolumbar_inference_analysis_explained",
        "thoracolumbar_inference_analysis_explained_tuned",
        ("inference_experiment_summary.csv", "inference_global_metrics.csv"),
    ),
    StageSpec(
        "05_postprocess_v2",
        "05",
        "Postproceso anatómico v2",
        "thoracolumbar_postprocess_anatomical_v2_explained",
        "thoracolumbar_postprocess_anatomical_v2_explained_tuned",
        ("postprocess_v2_experiment_summary.csv", "postprocess_v2_metric_comparison.csv"),
    ),
    StageSpec(
        "06_visible_range",
        "06",
        "Estimador rango visible",
        "visible_range_estimator_thoracolumbar_explained",
        "visible_range_estimator_thoracolumbar_explained_tuned",
        ("visible_range_experiment_summary.csv", "clipping_metric_comparison.csv"),
    ),
    StageSpec(
        "07_last_visible",
        "07",
        "Estimador última vértebra visible (pipeline final)",
        "last_visible_estimator_thoracolumbar_explained",
        "last_visible_estimator_thoracolumbar_explained_tuned",
        (
            "last_visible_experiment_summary.csv",
            "last_visible_clipping_metric_comparison.csv",
        ),
    ),
    StageSpec(
        "09_project_summary",
        "09",
        "Resumen final (tabla de variantes del pipeline)",
        "final_project_summary_thoracolumbar",
        "final_project_summary_thoracolumbar_tuned",
        ("final_pipeline_comparison.csv",),
    ),
    StageSpec(
        "11_pipeline_decision",
        "11",
        "Comparación global de variantes",
        "definitive_pipeline_decision_support_thoracolumbar",
        "definitive_pipeline_decision_support_thoracolumbar_tuned",
        ("global_pipeline_compare.csv",),
    ),
    StageSpec(
        "13_hybrid_policy",
        "13",
        "Política híbrida conservadora",
        "hybrid_conservative_clipping_policy_thoracolumbar",
        "hybrid_conservative_clipping_policy_thoracolumbar_tuned",
        ("hybrid_policy_compare.csv", "hybrid_policy_recommendation.csv"),
    ),
    StageSpec(
        "14_deployment",
        "14",
        "Política de despliegue final",
        "final_deployment_policy_thoracolumbar",
        "final_deployment_policy_thoracolumbar_tuned",
        ("final_policy_evaluation_table.csv", "final_policy_decision_table.csv"),
    ),
)


def artifact_roots(root: Path) -> list[Path]:
    return [root / "analysis_outputs", root / "outputs", root / "reports" / "analysis_outputs"]


def resolve_stage_dir(roots: list[Path], subdir: str) -> Path | None:
    for base in roots:
        candidate = base / subdir
        if candidate.is_dir():
            return candidate
    return None


def read_metric_value_csv(path: Path | None, metric: str) -> float | None:
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "metric" in cols and "value" in cols:
        mcol, vcol = cols["metric"], cols["value"]
        hit = df[df[mcol].astype(str) == metric]
        if not hit.empty:
            return float(hit[vcol].iloc[0])

    if metric in df.columns and len(df) == 1:
        return float(df[metric].iloc[0])

    if metric in df.columns:
        return float(df[metric].iloc[-1])

    return None


def read_post_v2_metric(path: Path | None, metric: str) -> float | None:
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    if "metric" not in df.columns or "value_post_v2" not in df.columns:
        return None
    hit = df[df["metric"].astype(str) == metric]
    if hit.empty:
        return None
    return float(hit["value_post_v2"].iloc[0])


def extract_from_clipping_compare(path: Path | None, column: str, metric: str = "macro_dice_fg") -> float | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "metric" not in df.columns or column not in df.columns:
        return None
    hit = df[df["metric"].astype(str) == metric]
    if hit.empty:
        return None
    return float(hit[column].iloc[0])


def extract_from_global_pipeline(path: Path, variant: str = "last_visible_pred_clip") -> float | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "variant" not in df.columns or "macro_dice_fg" not in df.columns:
        return None
    hit = df[df["variant"].astype(str) == variant]
    if hit.empty:
        best = df.sort_values("macro_dice_fg", ascending=False).iloc[0]
        return float(best["macro_dice_fg"])
    return float(hit["macro_dice_fg"].iloc[0])


def extract_from_policy_eval(path: Path, metric_name: str = "clip_macro_dice_fg") -> float | None:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "metric" in df.columns and "value" in df.columns:
        hit = df[df["metric"].astype(str) == metric_name]
        if not hit.empty:
            return float(hit["value"].iloc[0])
    if "policy_name" in df.columns and "mean_proxy_dice" in df.columns:
        if metric_name == "best_mean_proxy_dice":
            return float(df["mean_proxy_dice"].max())
    return None


def mean_column(path: Path | None, column: str) -> float | None:
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    if column not in df.columns:
        return None
    return float(df[column].mean())


def load_stage_metrics(stage_dir: Path | None, spec: StageSpec) -> dict[str, float]:
    out: dict[str, float] = {}
    if stage_dir is None:
        return out

    for fname in spec.files:
        path = stage_dir / fname
        if not path.is_file():
            continue

        if fname.endswith("_experiment_summary.csv") or fname.endswith("_global_metrics.csv"):
            for m in PRIMARY_METRICS + [
                "raw_macro_dice_fg",
                "oracle_macro_dice_fg",
                "last_pred_clip_macro_dice_fg",
                "pred_clip_macro_dice_fg",
                "post_v2_macro_dice_fg",
            ]:
                val = read_metric_value_csv(path, m)
                if val is not None:
                    out[m] = val

        if fname == "last_visible_clipping_metric_comparison.csv":
            for col, key in [
                ("raw", "clip_raw_macro_dice_fg"),
                ("last_pred_clip", "last_pred_clip_macro_dice_fg"),
                ("oracle_clip", "oracle_clip_macro_dice_fg"),
            ]:
                val = extract_from_clipping_compare(path, col)
                if val is not None:
                    out[key] = val

        if fname == "clipping_metric_comparison.csv":
            val = extract_from_clipping_compare(path, "pred_clip", "macro_dice_fg")
            if val is not None:
                out["pred_clip_macro_dice_fg"] = val

        if fname == "global_pipeline_compare.csv":
            val = extract_from_global_pipeline(path)
            if val is not None:
                out["pipeline_best_macro_dice_fg"] = val

        if fname == "final_policy_evaluation_table.csv":
            val = extract_from_policy_eval(path, "clip_macro_dice_fg")
            if val is None:
                val = extract_from_policy_eval(path, "best_mean_proxy_dice")
            if val is not None:
                out["deployment_proxy_macro_dice"] = val

    if f"thoracolumbar_{SUBSET}_test_metrics.csv" in spec.files:
        path = stage_dir / f"thoracolumbar_{SUBSET}_test_metrics.csv"
        if path.is_file():
            df = pd.read_csv(path)
            for m in PRIMARY_METRICS:
                if m in df.columns:
                    out[m] = float(df[m].iloc[0])

    if "binary_spine_test_metrics.csv" in spec.files:
        path = stage_dir / "binary_spine_test_metrics.csv"
        if path.is_file():
            df = pd.read_csv(path)
            for col in ("test_dice", "test_iou", "val_dice", "val_iou", "dice", "iou"):
                if col in df.columns:
                    key = col if col.startswith(("test_", "val_")) else f"test_{col}"
                    out[key] = float(df[col].iloc[0])

    if "final_pipeline_comparison.csv" in spec.files:
        path = stage_dir / "final_pipeline_comparison.csv"
        if path.is_file():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                variant = str(row["variant"])
                for col in FINAL_VARIANT_METRIC_COLS:
                    if col in df.columns and pd.notna(row[col]):
                        out[f"{variant}.{col}"] = float(row[col])

    if "last_visible_experiment_summary.csv" in spec.files:
        for m in (
            "last_test_exact_acc",
            "last_test_within1_acc",
            "last_test_mae",
            "last_test_overprediction_rate",
        ):
            val = read_metric_value_csv(stage_dir / "last_visible_experiment_summary.csv", m)
            if val is not None:
                out[m] = val

    return out


def _dir(roots: list[Path], name: str) -> Path | None:
    return resolve_stage_dir(roots, name)


def build_final_pipeline_comparison(roots: list[Path], tuned: bool) -> pd.DataFrame | None:
    """Tabla del notebook 09: 5 variantes × 5 métricas (como en la imagen del informe)."""
    suffix = "_tuned" if tuned else ""
    summary_dir = _dir(roots, f"final_project_summary_thoracolumbar{suffix}")
    # Solo usar cache del baseline (reports/). El nb 09 tuneado exporta valores hardcodeados
    # antiguos; la tabla tuneada se reconstruye siempre desde CSV de 04-07.
    cached = summary_dir / "final_pipeline_comparison.csv" if summary_dir else None
    if cached is not None and cached.is_file() and not tuned:
        return pd.read_csv(cached)

    inf = _dir(roots, f"thoracolumbar_inference_analysis_explained{suffix}")
    post = _dir(roots, f"thoracolumbar_postprocess_anatomical_v2_explained{suffix}")
    vis = _dir(roots, f"visible_range_estimator_thoracolumbar_explained{suffix}")
    last = _dir(roots, f"last_visible_estimator_thoracolumbar_explained{suffix}")
    if not any([inf, post, vis, last]):
        return None

    inf_glob = inf / "inference_global_metrics.csv" if inf else None
    post_exp = post / "postprocess_v2_experiment_summary.csv" if post else None
    post_ps = post / "postprocess_v2_per_sample_compare.csv" if post else None
    post_cmp = post / "postprocess_v2_metric_comparison.csv" if post else None
    vis_clip = vis / "clipping_metric_comparison.csv" if vis else None
    vis_ps = vis / "clipping_per_sample_compare.csv" if vis else None
    last_clip = last / "last_visible_clipping_metric_comparison.csv" if last else None
    last_ps = last / "last_visible_per_sample_compare.csv" if last else None

    def clip_metric(compare: Path | None, col: str, metric: str) -> float | None:
        return extract_from_clipping_compare(compare, col, metric)

    rows = [
        {
            "variant": "raw_multiclass_baseline",
            "macro_dice_fg": clip_metric(last_clip, "raw", "macro_dice_fg")
            or read_metric_value_csv(post_exp, "raw_macro_dice_fg"),
            "macro_iou_fg": clip_metric(last_clip, "raw", "macro_iou_fg")
            or read_metric_value_csv(post_exp, "raw_macro_iou_fg"),
            "macro_dice_lumbar": clip_metric(last_clip, "raw", "macro_dice_lumbar")
            or read_metric_value_csv(post_exp, "raw_macro_dice_lumbar"),
            "mean_extra_count": mean_column(last_ps, "raw_extra_count")
            or read_metric_value_csv(post_exp, "mean_raw_extra_count"),
            "mean_missing_count": mean_column(last_ps, "raw_missing_count")
            or read_metric_value_csv(post_exp, "mean_raw_missing_count"),
        },
        {
            "variant": "postprocess_v2",
            "macro_dice_fg": read_metric_value_csv(post_exp, "post_v2_macro_dice_fg"),
            "macro_iou_fg": read_post_v2_metric(post_cmp, "macro_iou_fg"),
            "macro_dice_lumbar": read_metric_value_csv(post_exp, "post_v2_macro_dice_lumbar"),
            "mean_extra_count": mean_column(post_ps, "post_extra_count")
            or read_metric_value_csv(post_exp, "mean_post_extra_count"),
            "mean_missing_count": mean_column(post_ps, "post_missing_count")
            or read_metric_value_csv(post_exp, "mean_post_missing_count"),
        },
        {
            "variant": "visible_range_pred_clip",
            "macro_dice_fg": clip_metric(vis_clip, "pred_clip", "macro_dice_fg")
            or read_metric_value_csv(inf_glob, "test_macro_dice_fg"),
            "macro_iou_fg": clip_metric(vis_clip, "pred_clip", "macro_iou_fg")
            or read_metric_value_csv(inf_glob, "test_macro_iou_fg"),
            "macro_dice_lumbar": clip_metric(vis_clip, "pred_clip", "macro_dice_lumbar")
            or read_metric_value_csv(inf_glob, "test_macro_dice_lumbar"),
            "mean_extra_count": mean_column(vis_ps, "pred_extra_count")
            or read_metric_value_csv(vis / "clipping_presence_summary.csv", "mean_pred_extra_count")
            if vis
            else None,
            "mean_missing_count": mean_column(vis_ps, "pred_missing_count")
            or read_metric_value_csv(vis / "clipping_presence_summary.csv", "mean_pred_missing_count")
            if vis
            else None,
        },
        {
            "variant": "last_visible_pred_clip",
            "macro_dice_fg": clip_metric(last_clip, "last_pred_clip", "macro_dice_fg")
            or read_metric_value_csv(last / "last_visible_experiment_summary.csv", "last_pred_clip_macro_dice_fg")
            if last
            else None,
            "macro_iou_fg": clip_metric(last_clip, "last_pred_clip", "macro_iou_fg"),
            "macro_dice_lumbar": clip_metric(last_clip, "last_pred_clip", "macro_dice_lumbar"),
            "mean_extra_count": mean_column(last_ps, "last_extra_count")
            or read_metric_value_csv(last / "last_visible_experiment_summary.csv", "mean_last_extra_count")
            if last
            else None,
            "mean_missing_count": mean_column(last_ps, "last_missing_count")
            or read_metric_value_csv(last / "last_visible_experiment_summary.csv", "mean_last_missing_count")
            if last
            else None,
        },
        {
            "variant": "oracle_clip_reference",
            "macro_dice_fg": clip_metric(last_clip, "oracle_clip", "macro_dice_fg"),
            "macro_iou_fg": clip_metric(last_clip, "oracle_clip", "macro_iou_fg"),
            "macro_dice_lumbar": clip_metric(last_clip, "oracle_clip", "macro_dice_lumbar"),
            "mean_extra_count": mean_column(last_ps, "oracle_extra_count"),
            "mean_missing_count": mean_column(last_ps, "oracle_missing_count"),
        },
    ]
    df = pd.DataFrame(rows)
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_dir / "final_pipeline_comparison.csv", index=False)
    return df


def compare_variant_tables(baseline_df: pd.DataFrame, tuned_df: pd.DataFrame | None) -> pd.DataFrame:
    """Une tablas final_pipeline_comparison baseline vs tuneado por variante."""
    b = baseline_df.set_index("variant")
    if tuned_df is None or tuned_df.empty:
        out = baseline_df.copy()
        out["line"] = "baseline"
        return out

    t = tuned_df.set_index("variant")
    rows: list[dict] = []
    for variant in b.index:
        row: dict = {"variant": variant}
        for col in FINAL_VARIANT_METRIC_COLS:
            bv = b.loc[variant, col] if col in b.columns else None
            tv = t.loc[variant, col] if variant in t.index and col in t.columns else None
            row[f"baseline_{col}"] = bv
            row[f"tuned_{col}"] = tv
            if bv is not None and tv is not None and pd.notna(bv) and pd.notna(tv):
                row[f"delta_{col}"] = float(tv) - float(bv)
            else:
                row[f"delta_{col}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparison_row(
    spec: StageSpec,
    metric: str,
    baseline: float | None,
    tuned: float | None,
) -> dict:
    delta = None
    pct = None
    if baseline is not None and tuned is not None:
        delta = tuned - baseline
        if baseline != 0:
            pct = (delta / baseline) * 100.0
    return {
        "stage_id": spec.stage_id,
        "notebook": spec.notebook,
        "stage_label": spec.label,
        "metric": metric,
        "baseline": baseline,
        "tuned": tuned,
        "delta_tuned_minus_baseline": delta,
        "delta_pct": pct,
        "baseline_available": baseline is not None,
        "tuned_available": tuned is not None,
    }


def write_variant_markdown(
    baseline_table: pd.DataFrame,
    variant_compare: pd.DataFrame,
    auxiliary: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "## Tabla de variantes del pipeline (notebook 09)",
        "",
        "Cinco variantes desplegables / de referencia, con las mismas columnas que el informe final.",
        "",
        "### Baseline",
        "",
        "| Variante | Macro Dice FG | Macro IoU FG | Macro Dice Lumbar | Mean Extra | Mean Missing |",
        "|----------|---------------|--------------|-------------------|------------|----------------|",
    ]
    for _, r in baseline_table.iterrows():
        lines.append(
            f"| {r['variant']} | {r['macro_dice_fg']:.4f} | {r['macro_iou_fg']:.4f} | "
            f"{r['macro_dice_lumbar']:.4f} | {r['mean_extra_count']:.4f} | {r['mean_missing_count']:.4f} |"
        )
    lines.append("")

    if not variant_compare.empty and "delta_macro_dice_fg" in variant_compare.columns:
        lines.append("### Baseline vs tuneado (Δ = tuneado − baseline)")
        lines.append("")
        lines.append("| Variante | Δ Dice FG | Δ IoU FG | Δ Dice Lumbar | Δ Extra | Δ Missing |")
        lines.append("|----------|-----------|----------|---------------|---------|-----------|")
        for _, r in variant_compare.iterrows():
            def fmt(col: str) -> str:
                v = r.get(col)
                return "" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:+.4f}"

            lines.append(
                f"| {r['variant']} | {fmt('delta_macro_dice_fg')} | {fmt('delta_macro_iou_fg')} | "
                f"{fmt('delta_macro_dice_lumbar')} | {fmt('delta_mean_extra_count')} | "
                f"{fmt('delta_mean_missing_count')} |"
            )
        lines.append("")

        last_row = variant_compare[variant_compare["variant"] == "last_visible_pred_clip"]
        if not last_row.empty:
            d_dice = last_row["delta_macro_dice_fg"].iloc[0]
            if pd.notna(d_dice):
                verdict = "mejora" if d_dice > 0 else "empeora" if d_dice < 0 else "igual"
                lines.append(
                    f"**Conclusión pipeline desplegable (`last_visible_pred_clip`):** la línea tuneada "
                    f"**{verdict}** en Macro Dice FG (Δ {d_dice:+.4f})."
                )
                lines.append("")

    if not auxiliary.empty:
        lines.append("## Métricas auxiliares (binario y last_visible)")
        lines.append("")
        lines.append("| Bloque | Métrica | Baseline | Tuneado | Δ |")
        lines.append("|--------|---------|----------|---------|---|")
        for _, r in auxiliary.iterrows():
            b = "" if pd.isna(r.get("baseline")) else f"{r['baseline']:.4f}"
            t = "" if pd.isna(r.get("tuned")) else f"{r['tuned']:.4f}"
            d = "" if pd.isna(r.get("delta")) else f"{r['delta']:+.4f}"
            lines.append(f"| {r['block']} | {r['metric']} | {b} | {t} | {d} |")
        lines.append("")

    out_path.write_text(out_path.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")


def metric_from_extracted(df: pd.DataFrame, stage_id: str, metric: str) -> float | None:
    if df.empty:
        return None
    hit = df[(df["stage_id"] == stage_id) & (df["metric"] == metric)]
    if hit.empty:
        return None
    return float(hit["value"].iloc[0])


def build_auxiliary_metrics(baseline_df: pd.DataFrame, tuned_df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("02_binary", "test_dice", "Dice test (binario)"),
        ("02_binary", "test_iou", "IoU test (binario)"),
        ("03_cascade", "test_macro_dice_fg", "Macro Dice FG (cascada multiclase)"),
        ("07_last_visible", "last_test_exact_acc", "last_exact_acc"),
        ("07_last_visible", "last_test_within1_acc", "last_within1_acc"),
    ]
    rows: list[dict] = []
    for block, key, label in specs:
        bv = metric_from_extracted(baseline_df, block, key)
        tv = metric_from_extracted(tuned_df, block, key)
        delta = None
        if bv is not None and tv is not None:
            delta = tv - bv
        rows.append({"block": block, "metric": label, "metric_key": key, "baseline": bv, "tuned": tv, "delta": delta})
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, out_path: Path, roots_checked: list[str]) -> None:
    lines = [
        "# Comparativo baseline vs tuneado (Yeisson)",
        "",
        "Métricas extraídas automáticamente de los CSV exportados por los notebooks.",
        "Ejecutar de nuevo tras completar la línea `_tuned`:",
        "",
        "```bash",
        "python scripts/compare_baseline_vs_tuned_metrics.py",
        "```",
        "",
        f"Raíces buscadas: {', '.join(roots_checked)}",
        "",
    ]

    cascade = summary[summary["stage_id"] == "03_cascade"]
    if not cascade.empty:
        lines.append("## Cascada (notebook 03)")
        lines.append("")
        lines.append("| Métrica | Baseline | Tuneado | Δ | Δ % |")
        lines.append("|---------|----------|---------|---|-----|")
        for _, r in cascade.iterrows():
            b = "" if pd.isna(r["baseline"]) else f"{r['baseline']:.4f}"
            t = "" if pd.isna(r["tuned"]) else f"{r['tuned']:.4f}"
            d = "" if pd.isna(r["delta_tuned_minus_baseline"]) else f"{r['delta_tuned_minus_baseline']:+.4f}"
            p = "" if pd.isna(r["delta_pct"]) else f"{r['delta_pct']:+.2f}%"
            lines.append(f"| {r['metric']} | {b} | {t} | {d} | {p} |")
        lines.append("")

    finals = summary[summary["stage_id"].isin(["07_last_visible", "11_pipeline_decision", "14_deployment"])]
    if not finals.empty:
        lines.append("## Métricas finales del pipeline")
        lines.append("")
        lines.append("| Etapa | Métrica | Baseline | Tuneado | Δ |")
        lines.append("|-------|---------|----------|---------|---|")
        for _, r in finals.iterrows():
            b = "" if pd.isna(r["baseline"]) else f"{r['baseline']:.4f}"
            t = "" if pd.isna(r["tuned"]) else f"{r['tuned']:.4f}"
            d = "" if pd.isna(r["delta_tuned_minus_baseline"]) else f"{r['delta_tuned_minus_baseline']:+.4f}"
            lines.append(f"| {r['stage_label']} | {r['metric']} | {b} | {t} | {d} |")
        lines.append("")

    missing_b = summary[~summary["baseline_available"]]["stage_id"].unique()
    missing_t = summary[~summary["tuned_available"]]["stage_id"].unique()
    if len(missing_b):
        lines.append(f"**Sin baseline local:** {', '.join(missing_b)}")
        lines.append("")
    if len(missing_t):
        lines.append(f"**Sin tuneado (pendiente ejecución):** {', '.join(missing_t)}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar métricas baseline vs tuneado.")
    parser.add_argument("--root", type=Path, default=REPO, help="Raíz del proyecto (contiene analysis_outputs/).")
    parser.add_argument("--baseline-only", action="store_true", help="Solo exportar métricas baseline.")
    parser.add_argument("--tuned-only", action="store_true", help="Solo exportar métricas tuneadas.")
    args = parser.parse_args()

    roots = artifact_roots(args.root.resolve())
    roots_checked = [str(r) for r in roots if r.exists()]

    REPORTS.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict] = []
    tuned_rows: list[dict] = []
    compare_rows: list[dict] = []

    for spec in STAGES:
        b_dir = resolve_stage_dir(roots, spec.baseline_dir)
        t_dir = resolve_stage_dir(roots, spec.tuned_dir)

        b_metrics = load_stage_metrics(b_dir, spec) if not args.tuned_only else {}
        t_metrics = load_stage_metrics(t_dir, spec) if not args.baseline_only else {}

        for m, v in b_metrics.items():
            baseline_rows.append(
                {
                    "stage_id": spec.stage_id,
                    "notebook": spec.notebook,
                    "stage_label": spec.label,
                    "metric": m,
                    "value": v,
                    "source_dir": str(b_dir) if b_dir else "",
                }
            )
        for m, v in t_metrics.items():
            tuned_rows.append(
                {
                    "stage_id": spec.stage_id,
                    "notebook": spec.notebook,
                    "stage_label": spec.label,
                    "metric": m,
                    "value": v,
                    "source_dir": str(t_dir) if t_dir else "",
                }
            )

        all_metrics = sorted(set(b_metrics) | set(t_metrics))
        for m in all_metrics:
            compare_rows.append(
                build_comparison_row(spec, m, b_metrics.get(m), t_metrics.get(m))
            )

    baseline_df = pd.DataFrame(baseline_rows)
    tuned_df = pd.DataFrame(tuned_rows)
    compare_df = pd.DataFrame(compare_rows)

    baseline_df.to_csv(REPORTS / "baseline_metrics_extracted.csv", index=False)
    if not args.baseline_only:
        tuned_df.to_csv(REPORTS / "tuned_metrics_extracted.csv", index=False)
    compare_df.to_csv(REPORTS / "comparison_baseline_vs_tuned.csv", index=False)

    manifest = {
        "root": str(args.root.resolve()),
        "artifact_roots_checked": roots_checked,
        "stages": [
            {
                "stage_id": s.stage_id,
                "baseline_dir": s.baseline_dir,
                "tuned_dir": s.tuned_dir,
                "baseline_resolved": str(resolve_stage_dir(roots, s.baseline_dir) or ""),
                "tuned_resolved": str(resolve_stage_dir(roots, s.tuned_dir) or ""),
            }
            for s in STAGES
        ],
    }
    (REPORTS / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_markdown(compare_df, REPORTS / "comparison_baseline_vs_tuned.md", roots_checked)

    baseline_variant_table = build_final_pipeline_comparison(roots, tuned=False)
    tuned_variant_table = None if args.baseline_only else build_final_pipeline_comparison(roots, tuned=True)

    if baseline_variant_table is not None:
        baseline_variant_table.to_csv(REPORTS / "baseline_pipeline_variants.csv", index=False)
        variant_compare = compare_variant_tables(
            baseline_variant_table,
            tuned_variant_table if tuned_variant_table is not None else pd.DataFrame(),
        )
        variant_compare.to_csv(REPORTS / "comparison_pipeline_variants.csv", index=False)
        if tuned_variant_table is not None and not tuned_variant_table.empty:
            tuned_variant_table.to_csv(REPORTS / "tuned_pipeline_variants.csv", index=False)

        aux_df = build_auxiliary_metrics(baseline_df, tuned_df)
        aux_df.to_csv(REPORTS / "comparison_auxiliary_metrics.csv", index=False)
        write_variant_markdown(
            baseline_variant_table,
            variant_compare,
            aux_df,
            REPORTS / "comparison_baseline_vs_tuned.md",
        )

    print(f"Reportes en: {REPORTS}")
    print(f"  baseline filas: {len(baseline_df)}")
    print(f"  tuned filas:    {len(tuned_df)}")
    print(f"  comparación:    {len(compare_df)}")
    if baseline_df.empty and tuned_df.empty:
        print(
            "\nNo se encontraron CSV. Copia la carpeta analysis_outputs/ desde Colab/Drive "
            f"a:\n  {args.root / 'analysis_outputs'}"
        )


if __name__ == "__main__":
    main()
