# Comparativo baseline vs tuneado (Yeisson)

## Resumen ejecutivo

**Objetivo:** evaluar si las mejoras del notebook **03 tuneado** (Fase 3: LR multiclase ×0.5 + `CosineAnnealingLR`; Fase 4: augmentación geométrica suave en ROI) mejoran el pipeline thoracolumbar frente al baseline Yeisson.

**Veredicto:** **sí, la línea tuneada mejora de forma consistente.** El binario (02) no se reentrenó; el resto del pipeline consume el modelo multiclase tuneado y refleja la mejora en cascada.

| Ámbito | Resultado clave |
|--------|-----------------|
| **Cascada multiclase (03)** | `test_macro_dice_fg` **0.3226 → 0.3399** (+5.4%). Lumbar +5.9%, IoU FG +6.7%. |
| **Pipeline desplegable (`last_visible_pred_clip`)** | Macro Dice FG **0.3324 → 0.3519** (+5.9%). Menos vértebras extra (−0.73 de media). |
| **Estimador última vértebra (07)** | `last_exact_acc` **33% → 42%** (+9 pp). Sobrepredicción **64% → 40%** (−24 pp). |
| **Binario (02)** | Sin cambio (mismo checkpoint): Dice 0.8967, IoU 0.8127. |

**Trade-off:** sube `mean_missing_count` en clipping (~0.02 → ~0.49 en `last_visible_pred_clip`); conviene revisar casos límite en `last_visible_per_sample_compare.csv` tuneado.

**Fuentes de datos:** baseline en `reports/analysis_outputs/`; tuneado en `analysis_outputs/*_tuned/`. La tabla de variantes (sección 6) se reconstruye desde CSV de notebooks **04–07**, no desde valores fijos del 09.

**Archivos de este informe:**

| Archivo | Descripción |
|---------|-------------|
| `comparison_baseline_vs_tuned.csv` | Todas las métricas por etapa |
| `comparison_pipeline_variants.csv` | Cinco variantes × cinco métricas (estilo informe) |
| `comparison_auxiliary_metrics.csv` | Binario, cascada y last_visible |
| `comparison_manifest.json` | Rutas resueltas por etapa |

---

Métricas extraídas automáticamente con:

```bash
python scripts/compare_baseline_vs_tuned_metrics.py
```

Raíces buscadas: `analysis_outputs/`, `outputs/`, `reports/analysis_outputs/`

## Cascada (notebook 03)

| Métrica | Baseline | Tuneado | Δ | Δ % |
|---------|----------|---------|---|-----|
| best_val_macro_dice_fg | 0.2314 | 0.2491 | +0.0177 | +7.65% |
| test_macro_dice_fg | 0.3226 | 0.3399 | +0.0173 | +5.36% |
| test_macro_dice_lumbar | 0.3956 | 0.4189 | +0.0233 | +5.89% |
| test_macro_dice_thoracic | 0.2922 | 0.3070 | +0.0148 | +5.06% |
| test_macro_iou_fg | 0.2001 | 0.2134 | +0.0133 | +6.67% |

## Métricas finales del pipeline

| Etapa | Métrica | Baseline | Tuneado | Δ |
|-------|---------|----------|---------|---|
| Estimador última vértebra visible (pipeline final) | clip_raw_macro_dice_fg | 0.3249 | 0.3399 | +0.0150 |
| Estimador última vértebra visible (pipeline final) | last_pred_clip_macro_dice_fg | 0.3324 | 0.3519 | +0.0196 |
| Estimador última vértebra visible (pipeline final) | last_test_exact_acc | 0.3333 | 0.4222 | +0.0889 |
| Estimador última vértebra visible (pipeline final) | last_test_mae | 2.2667 | 2.0000 | -0.2667 |
| Estimador última vértebra visible (pipeline final) | last_test_overprediction_rate | 0.6444 | 0.4000 | -0.2444 |
| Estimador última vértebra visible (pipeline final) | last_test_within1_acc | 0.5556 | 0.5778 | +0.0222 |
| Estimador última vértebra visible (pipeline final) | oracle_clip_macro_dice_fg | 0.3577 | 0.3711 | +0.0134 |
| Estimador última vértebra visible (pipeline final) | oracle_macro_dice_fg | 0.3577 | 0.3711 | +0.0134 |
| Estimador última vértebra visible (pipeline final) | raw_macro_dice_fg | 0.3249 | 0.3399 | +0.0150 |
| Comparación global de variantes | pipeline_best_macro_dice_fg |  | 0.3519 |  |
| Política de despliegue final | deployment_proxy_macro_dice |  | 0.9254 |  |

**Sin baseline local:** 11_pipeline_decision, 14_deployment

## Tabla de variantes del pipeline (notebook 09)

Cinco variantes desplegables / de referencia, con las mismas columnas que el informe final.

### Baseline

| Variante | Macro Dice FG | Macro IoU FG | Macro Dice Lumbar | Mean Extra | Mean Missing |
|----------|---------------|--------------|-------------------|------------|----------------|
| raw_multiclass_baseline | 0.3249 | 0.2018 | 0.3978 | 3.0222 | 0.0222 |
| postprocess_v2 | 0.3250 | 0.2022 | 0.4009 | 2.5778 | 1.0444 |
| visible_range_pred_clip | 0.3205 | 0.1988 | 0.3875 | 3.0222 | 0.0222 |
| last_visible_pred_clip | 0.3324 | 0.2079 | 0.4205 | 2.2444 | 0.0222 |
| oracle_clip_reference | 0.3577 | 0.2291 | 0.4955 | 0.0000 | 0.0000 |

### Baseline vs tuneado (Δ = tuneado − baseline)

| Variante | Δ Dice FG | Δ IoU FG | Δ Dice Lumbar | Δ Extra | Δ Missing |
|----------|-----------|----------|---------------|---------|-----------|
| raw_multiclass_baseline | +0.0150 | +0.0117 | +0.0211 | +0.0222 |  |
| postprocess_v2 | +0.0141 | +0.0114 | +0.0257 | -0.4444 | +0.4889 |
| visible_range_pred_clip | +0.0217 | +0.0166 | +0.0394 | -0.1333 | +0.0000 |
| last_visible_pred_clip | +0.0196 | +0.0158 | +0.0383 | -0.7333 | +0.4667 |
| oracle_clip_reference | +0.0134 | +0.0108 | +0.0135 | +0.0000 | +0.0000 |

**Conclusión pipeline desplegable (`last_visible_pred_clip`):** la línea tuneada **mejora** en Macro Dice FG (Δ +0.0196).

## Métricas auxiliares (binario y last_visible)

| Bloque | Métrica | Baseline | Tuneado | Δ |
|--------|---------|----------|---------|---|
| 02_binary | Dice test (binario) | 0.8967 | 0.8967 | +0.0000 |
| 02_binary | IoU test (binario) | 0.8127 | 0.8127 | +0.0000 |
| 03_cascade | Macro Dice FG (cascada multiclase) | 0.3226 | 0.3399 | +0.0173 |
| 07_last_visible | last_exact_acc | 0.3333 | 0.4222 | +0.0889 |
| 07_last_visible | last_within1_acc | 0.5556 | 0.5778 | +0.0222 |
