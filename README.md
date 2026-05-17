# Pipeline thoracolumbar tuneado (Colab)

Rama **`yeisson-tuned-pipeline`**: experimento reproducible de segmentación toracolumbar (`T1..T12`, `L1..L5`) con mejoras **Fase 3** y **Fase 4** en el entrenamiento multiclase (notebook 03 tuneado), más las etapas posteriores del pipeline Yeisson adaptadas a la línea `*_tuned`.

Este documento es la guía única de esta rama. No incluye notebooks base 03–14 (sin `_tuned`); esos viven en la rama `Yeisson` solo como referencia histórica.

- Cristian Camilo Nino Rincon
- Sandra Milena Pantoja Cárdenas
- Integrante pendiente 3
- Integrante pendiente 4

## Para quién es esta guía

Si acabas de clonar la rama y **no conoces el proyecto**, sigue este orden:

1. Lee [Arquitectura](#arquitectura-de-la-solución).
2. Completa [Requisitos previos](#requisitos-previos).
3. Elige [modo de ejecución](#modos-de-ejecución).
4. Ejecuta los [notebooks en orden](#orden-de-ejecución-de-notebooks).
5. Genera [métricas y conclusiones](#métricas-y-conclusiones-scripts-python).

---

## Arquitectura de la solución

Pipeline en cascada: cada etapa consume artefactos de las anteriores.

```text
Radiografía + máscara (dataset local, no versionado)
        │
        ▼
[01] Manifest, cobertura y split conceptual
        │
        ▼
[02] Modelo binario → localiza columna
        │
        ▼
[03 tuned] ROI espinal + modelo multiclase (Fase 3 + Fase 4)
        │
        ├──► [04 tuned] Inferencia y análisis de errores (test)
        ├──► [05 tuned] Postproceso anatómico conservador (v2)
        ├──► [06 tuned] Estimador de rango visible (primera/última vértebra)
        └──► [07 tuned] Estimador de última vértebra visible + clipping
                    │
                    ▼
            [08 tuned] Pipeline de inferencia sobre imágenes nuevas
                    │
                    ▼
            [09 tuned] Resumen técnico y tablas del proyecto
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
[10 tuned] Hard-case mining   [11 tuned] Comparación global de variantes
        │                       │
        ▼                       ▼
[12 tuned] Reentrenamiento     [13 tuned] Política híbrida de clipping
         refinado last_visible          │
                                        ▼
                              [14 tuned] Política de despliegue final
```

**Variante operativa recomendada** (tras las etapas 07–14): `binary → multiclase tuneada → last_visible_pred_clip` (detalle en `reports/baseline_vs_tuned/comparison_baseline_vs_tuned.md`).

### Qué aporta el notebook 03 tuneado

| Fase | Cambio |
|------|--------|
| **Fase 3** | Learning rate de la cabeza multiclase ×0.5, scheduler `CosineAnnealingLR`, `cudnn.deterministic=True` |
| **Fase 4** | Augmentación geométrica suave en ROI (`apply_fase4_roi_geom_augment_uint8`) |

El notebook **02** no se reentrena en este experimento: se reutiliza el checkpoint binario incluido en el repositorio salvo que ejecutes una corrida completa desde cero.

---

## Estructura del repositorio (esta rama)

```text
ScoliosisSegmentation-Yeisson-work/
├── data/                              # NO versionado (.gitignore)
│   └── ScoliosisDataSetYeisson/       # Imágenes, máscaras, índice, diccionario
├── models/
│   ├── binary_spine_thoracolumbar_best.pt
│   ├── thoracolumbar_partial_cascade_explained_tuned_best.pt
│   ├── visible_range_estimator_thoracolumbar_tuned_best.pt
│   ├── last_visible_estimator_thoracolumbar_tuned_best.pt
│   └── last_visible_estimator_thoracolumbar_refined_tuned_best.pt
├── notebooks/
│   ├── 01_colab_thoracolumbar_coverage_strategy_clean.ipynb
│   ├── 02_colab_train_spine_binary_and_thoracolumbar.ipynb
│   └── 03 … 14 *_explained_tuned.ipynb
├── analysis_outputs/                  # Salidas del experimento tuneado (*_tuned)
├── reports/
│   ├── analysis_outputs/              # Métricas baseline (solo para comparar)
│   └── baseline_vs_tuned/             # Informe comparativo (CSV + .md)
├── scripts/                           # Utilidades Python
├── outputs/                           # Entradas/salidas de inferencia final (opcional)
└── README.md                          # Este manual
```

---

## Requisitos previos

### Hardware y entorno

- **Google Colab** con GPU (entrenamiento 02, 03, 06, 07, 12).
- Python **3.10+** en máquina local (solo para scripts de métricas): `pip install pandas`.

### Dataset (obligatorio, local)

Los datos **no** están en Git. Copia el dataset en:

```text
data/ScoliosisDataSetYeisson/
├── indice_dataset.csv
├── diccionario_etiquetas_T1_T12_L1_L5.json
├── images/          # (o la estructura que use tu índice)
└── masks/           # (rutas según indice_dataset.csv)
```

### Sincronización con Colab (Drive)

Los notebooks asumen por defecto:

```python
PROJECT_ROOT = Path("/content/drive/Othercomputers/Mi portátil/ScoliosisSegmentation-Yeisson-work")
DATASET_ROOT = PROJECT_ROOT / "data" / "ScoliosisDataSetYeisson"
```

Pasos:

1. Clona o sincroniza este repositorio en Google Drive (ruta coherente con `PROJECT_ROOT`).
2. Coloca `data/ScoliosisDataSetYeisson/` dentro de esa carpeta.
3. En Colab: **Runtime → Change runtime type → GPU**.
4. Abre cada notebook desde Drive y ejecuta **todas las celdas en orden** (no saltar celdas de instalación o rutas).

Si cambias la ruta del proyecto en Drive, actualiza las celdas `PROJECT_ROOT` / `DATASET_ROOT` o ejecuta:

```bash
python scripts/patch_tuned_notebook_colab_paths.py
```

(desde la raíz del repo, en tu PC; luego vuelve a subir los notebooks modificados si aplica).

---

## Modos de ejecución

### Modo A — Revisar resultados ya incluidos en la rama

La rama puede traer `models/*_tuned*.pt` y `analysis_outputs/*_tuned/` de una corrida previa.

1. Verifica el informe: `reports/baseline_vs_tuned/comparison_baseline_vs_tuned.md`.
2. Opcional: regenera tablas con el script (sección [Métricas y conclusiones](#métricas-y-conclusiones-scripts-python)).

Útil para Yeisson o revisores que quieren **analizar sin reentrenar**.

### Modo B — Ejecución limpia (reproducir de cero)

1. Cumple [Requisitos previos](#requisitos-previos).
2. (Opcional) Respalda y borra salidas tuneadas previas:
   - `analysis_outputs/*_tuned/`
   - `models/*_tuned*.pt`
   - `models/last_visible_estimator_thoracolumbar_refined_tuned_best.pt`
3. Si quieres regenerar también partición y binario: borra además
   - `analysis_outputs/training_manifest_thoracolumbar.csv` y archivos del 01
   - `analysis_outputs/training_runs_binary_thoracolumbar/`
   - `models/binary_spine_thoracolumbar_best.pt`
4. Ejecuta los notebooks en el [orden indicado](#orden-de-ejecución-de-notebooks).
5. Al terminar, ejecuta `compare_baseline_vs_tuned_metrics.py`.

Tiempo estimado: varias horas con GPU (03 y 12 son los más costosos).

---

## Orden de ejecución de notebooks

Ejecutar **en este orden**. No avances si el notebook anterior falló o no exportó sus CSV/checkpoints.

| Paso | Notebook | Depende de | Salidas principales |
|------|----------|------------|---------------------|
| 1 | `01_colab_thoracolumbar_coverage_strategy_clean.ipynb` | Dataset local | `analysis_outputs/training_manifest_thoracolumbar.csv`, matrices de cobertura |
| 2 | `02_colab_train_spine_binary_and_thoracolumbar.ipynb` | 01 (manifest) | `models/binary_spine_thoracolumbar_best.pt`, `analysis_outputs/training_runs_binary_thoracolumbar/` |
| 3 | `03_colab_train_spine_cascade_binary_to_thoracolumbar_explained_tuned.ipynb` | 02 (binario), manifest | `models/thoracolumbar_partial_cascade_explained_tuned_best.pt`, `analysis_outputs/training_runs_cascade_thoracolumbar_explained_tuned/` |
| 4 | `04_colab_infer_analyze_thoracolumbar_predictions_explained_tuned.ipynb` | 03 (multiclase tuneada) | `analysis_outputs/thoracolumbar_inference_analysis_explained_tuned/` |
| 5 | `05_colab_postprocess_anatomical_thoracolumbar_v2_explained_tuned.ipynb` | 04 | `analysis_outputs/thoracolumbar_postprocess_anatomical_v2_explained_tuned/` |
| 6 | `06_colab_train_visible_range_estimator_and_clip_thoracolumbar_explained_tuned.ipynb` | 03, 04 | `models/visible_range_estimator_thoracolumbar_tuned_best.pt`, `analysis_outputs/visible_range_estimator_thoracolumbar_explained_tuned/` |
| 7 | `07_colab_train_last_visible_estimator_and_clip_thoracolumbar_explained_tuned.ipynb` | 03, 04 | `models/last_visible_estimator_thoracolumbar_tuned_best.pt`, `analysis_outputs/last_visible_estimator_thoracolumbar_explained_tuned/` |
| 8 | `08_colab_final_inference_pipeline_thoracolumbar_explained_tuned.ipynb` | 02, 03, 07 | `analysis_outputs/final_inference_pipeline_thoracolumbar_tuned/`, opcionalmente `outputs/` |
| 9 | `09_colab_final_project_summary_thoracolumbar_explained_tuned.ipynb` | 04–07 (CSV exportados) | `analysis_outputs/final_project_summary_thoracolumbar_tuned/` |
| 10 | `10_colab_hard_case_mining_and_refined_sampling_thoracolumbar_explained_tuned.ipynb` | 04, 05, 07 tuneados | `analysis_outputs/hard_case_mining_thoracolumbar_explained_tuned/` |
| 11 | `11_colab_definitive_pipeline_decision_support_thoracolumbar_explained_tuned.ipynb` | 04–07, 09 | `analysis_outputs/definitive_pipeline_decision_support_thoracolumbar_tuned/` |
| 12 | `12_colab_refined_retraining_last_visible_thoracolumbar_explained_tuned.ipynb` | 10 (pesos de muestreo), 07 | `models/last_visible_estimator_thoracolumbar_refined_tuned_best.pt`, `analysis_outputs/last_visible_estimator_thoracolumbar_refined_explained_tuned/` |
| 13 | `13_colab_hybrid_conservative_clipping_policy_thoracolumbar_explained_tuned.ipynb` | 11, 12 | `analysis_outputs/hybrid_conservative_clipping_policy_thoracolumbar_tuned/` |
| 14 | `14_colab_final_deployment_policy_thoracolumbar_explained_tuned.ipynb` | 11–13 | `analysis_outputs/final_deployment_policy_thoracolumbar_tuned/` |

**Notas:**

- Las carpetas bajo `analysis_outputs/` del experimento tuneado llevan sufijo `_tuned`.
- El 01 y el 02 escriben en `analysis_outputs/` **sin** sufijo `_tuned` (preparación compartida).
- El 08 puede ejecutarse antes del 09–14 si solo necesitas inferencia; el resumen y las políticas finales requieren las etapas intermedias.

---

## Métricas y conclusiones (scripts Python)

### 1. `compare_baseline_vs_tuned_metrics.py` (obligatorio al cerrar el experimento)

Consolida métricas de la línea **baseline** (`reports/analysis_outputs/`) frente a la línea **tuneada** (`analysis_outputs/*_tuned/`) y escribe el informe en `reports/baseline_vs_tuned/`.

```bash
cd /ruta/a/ScoliosisSegmentation-Yeisson-work
pip install pandas
python scripts/compare_baseline_vs_tuned_metrics.py
```

Opciones útiles:

```bash
# Si el repo está en otra ruta
python scripts/compare_baseline_vs_tuned_metrics.py --root "D:/MiCarpeta/ScoliosisSegmentation-Yeisson-work"

# Solo extraer métricas tuneadas (sin comparar)
python scripts/compare_baseline_vs_tuned_metrics.py --tuned-only
```

**Archivos generados:**

| Archivo | Contenido |
|---------|-----------|
| `comparison_baseline_vs_tuned.md` | Resumen ejecutivo y tablas (léelo primero) |
| `comparison_baseline_vs_tuned.csv` | Todas las métricas por etapa |
| `comparison_pipeline_variants.csv` | Variantes del pipeline (estilo notebook 09) |
| `comparison_auxiliary_metrics.csv` | Binario, cascada, last_visible |
| `comparison_manifest.json` | Rutas resueltas por etapa (auditoría) |

**Conclusión esperada** (corrida de referencia): mejora consistente del tuneado en cascada multiclase y en `last_visible_pred_clip`; detalle numérico en `comparison_baseline_vs_tuned.md`.

### 2. `patch_tuned_notebook_colab_paths.py` (solo si cambias rutas Drive)

Actualiza `PROJECT_ROOT`, `DATASET_ROOT` y `search_roots` en los notebooks `*_tuned` (y opcionalmente 01/02). Edita las constantes al inicio del script antes de ejecutarlo.

### 3. `setup_cascade_tuned_pipeline.py` (mantenedores, no para reproducir)

Regenera notebooks `*_tuned` a partir de versiones base en otra rama. **No lo ejecutes** si solo quieres repetir el experimento: usa los notebooks ya versionados en esta rama.

---

## Comprobación rápida tras cada etapa

| Tras notebook | Comprueba que exista |
|---------------|----------------------|
| 01 | `analysis_outputs/training_manifest_thoracolumbar.csv` |
| 02 | `models/binary_spine_thoracolumbar_best.pt` |
| 03 tuned | `models/thoracolumbar_partial_cascade_explained_tuned_best.pt` |
| 04 tuned | `.../inference_experiment_summary.csv` en carpeta `*_tuned` |
| 07 tuned | `models/last_visible_estimator_thoracolumbar_tuned_best.pt` |
| 12 tuned | `models/last_visible_estimator_thoracolumbar_refined_tuned_best.pt` |
| 14 tuned | `final_policy_decision_table.csv` en `final_deployment_policy_thoracolumbar_tuned/` |

Si falta un archivo, revisa la celda de exportación del notebook y errores de ruta (`FileNotFoundError` en `DATASET_ROOT`).

---

## Errores frecuentes y solución

| Síntoma | Causa probable | Qué hacer |
|---------|----------------|-----------|
| `FileNotFoundError` en `PROJECT_ROOT` | Drive no montado o ruta distinta | Monta Drive; corrige `PROJECT_ROOT` o usa `patch_tuned_notebook_colab_paths.py` |
| `FileNotFoundError` en imagen/máscara | Dataset incompleto o rutas del índice | Verifica `data/ScoliosisDataSetYeisson/` e `indice_dataset.csv` |
| Notebook 10 falla al leer CSV | 04, 05 o 07 no ejecutados | Completa esas etapas tuneadas primero |
| Notebook 03 OOM | ROI/resolución altas en GPU pequeña | Runtime con más RAM/GPU; reduce batch en celda de configuración si el notebook lo permite |
| `compare_...py` sin filas tuneadas | CSV no copiados desde Colab a `analysis_outputs/` | Sincroniza Drive → PC; confirma carpetas `*_tuned` bajo `analysis_outputs/` |
| `compare_...py` sin baseline | Normal si solo reejecutaste tuneado | Baseline de referencia está en `reports/analysis_outputs/` (incluido en la rama) |
| Desconexión Colab (errno 107, etc.) | Sesión larga o caché en Drive | Reejecuta desde última celda estable; en 12 usa caché local de máscaras si el notebook lo define |
| Métricas del 09 incoherentes | Celdas con tablas hardcodeadas antiguas | Usa `09_*_tuned.ipynb` de esta rama; valida con `comparison_pipeline_variants.csv` del script |

---

## Flujo resumido (checklist)

- [ ] Clonar rama `yeisson-tuned-pipeline`
- [ ] Colocar dataset en `data/ScoliosisDataSetYeisson/`
- [ ] Configurar `PROJECT_ROOT` / GPU en Colab
- [ ] Ejecutar notebooks **01 → 02 → 03 tuned → … → 14 tuned**
- [ ] Verificar checkpoints en `models/` y CSV en `analysis_outputs/*_tuned/`
- [ ] Ejecutar `python scripts/compare_baseline_vs_tuned_metrics.py`
- [ ] Leer conclusiones en `reports/baseline_vs_tuned/comparison_baseline_vs_tuned.md`

---

## Referencia del equipo

Proyecto de segmentación toracolumbar — Grupo 18. Rama preparada para revisión del experimento tuneado (mejoras Fase 3–4 + pipeline completo `*_tuned`).

Para la línea baseline original (notebooks sin `_tuned`), usar la rama **`Yeisson`** del mismo repositorio.
