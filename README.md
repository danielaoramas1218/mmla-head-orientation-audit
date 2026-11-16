# Auditing head-orientation attention in a co-located MMLA dataset

Code accompanying the article *Auditing Head-Orientation Attention in Co-located Multimodal
Learning Analytics: Integration, Synchronisation and Validity in an Agile-Team Dataset*.

The scripts audit the attention measure released with the multimodal dataset of agile
student teams by Miranda et al. (*Data in Brief* 61, 111828, 2025,
doi:10.1016/j.dib.2025.111828). They read the released files and never modify them.

## Data

The data are **not** redistributed here. Download the released dataset (parquet files and
audio per session) and, optionally, the rendered videos (Mendeley Data,
doi:10.17632/rbpvjdpkby.1). Point every script to **one** copy of the dataset folder
(`ROOT` below); scripts search it recursively, so duplicated copies are counted twice.

Sessions analysed: 16 groups × conditions A/B = 32 sessions (groups 03, 06 and 17, recorded
under conditions C/D, are excluded). G21_B (`12-07-B`) has an unrevised transcript and is
excluded from attention analyses by the scripts.

## Requirements

Python ≥ 3.10 and `pip install -r requirements.txt`.

## How to reproduce the article (run in this order, from one working folder)

| Step | Script | Output | Article |
|---|---|---|---|
| 1 | `python scripts/f0_chequeos.py ROOT` | `f0_resultados.csv`, `f0_matrices.txt` | Integration checks, Table I, Fig. 1–2 |
| 2 | `python scripts/llave_azar.py ROOT` | `llave_azar.csv`, `llave_azar_resumen.txt` | Speaker–identity key vs. chance |
| 3 | `python scripts/mirada_recalc.py ROOT f0_resultados.csv` | `mirada/`, `mirada_resumen.csv` | Recomputed head-orientation targets |
| 4 | `python scripts/validez_oyentes_puros.py ROOT f0_resultados.csv` | `validez_oyentes_puros.csv` | Construct check (listeners only, main) |
| 5 | `python scripts/validez_oyente_hablante.py ROOT f0_resultados.csv` | `validez_por_hablante.csv` | Construct check (denominator incl. active speaker) |
| 6 | `python scripts/sensibilidad_theta.py ROOT f0_resultados.csv` | `sens_none*_down*/` | Threshold sensitivity (recomputes targets) |
| 7 | `python scripts/sensibilidad_oyentes.py ROOT f0_resultados.csv` | `tabla_sensibilidad.tex` | Table II |
| 8 | `python scripts/validez_down.py ROOT f0_resultados.csv` | `validez_down_*.csv/.txt` | Behaviour of the *down* class |
| 9 | `python scripts/muestra_anotacion.py ...` | annotation sheet + hidden key | Manual validation sample |
| 10 | `figures/figuras_paper.ipynb` (Colab or Jupyter) | `figuras/*.pdf` | Figures |

Main parameters (fixed in `scripts/mirada_recalc.py`): `PITCH_ABAJO` (θ_down = 15°) and
`ANG_NADIE` (θ_none = 30°). Effective frame rate = video frames / audio duration; the
transcript–video offset is estimated per session in step 1.

Script names and comments are in Spanish; the variables are documented in each file header.

## Note on the manual validation

`muestra_anotacion.py` draws the annotation sample with a fixed seed. The hidden key
(automatic labels) is not included here so that annotators remain blind; it will be added
once annotation is complete.

## Licence

MIT (see `LICENSE`). Please cite the article and the original dataset.
