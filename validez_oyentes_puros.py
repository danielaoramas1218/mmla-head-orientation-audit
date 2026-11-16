"""
Revisión Q1, punto M2: el denominador p_off de la razón incluía los frames de quien
está hablando en ese momento (no solo oyentes). Este script recalcula la razón
con dos definiciones y las compara:
  original : p_off sobre los 3 participantes != k mientras habla otra persona j (incluye a j)
  oyentes  : p_off solo sobre los oyentes, es decir, se excluyen los frames de j
p_on no cambia (cuando k habla, los otros tres son oyentes).

Uso (en la carpeta f0, con mirada/ de la especificación principal 30°/15°):
    python validez_oyentes_puros.py "C:\\...\\RECIFE" f0_resultados.csv
Salida: validez_oyentes_puros.csv y resumen en pantalla
"""
import sys, os, glob
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from validez_oyente_hablante import lag_sesion, hablante_unico, EXCLUIR


def razon_geom(H, col):
    d = H[H[col] > 0].dropna(subset=[col]).copy()
    d["lr"] = np.log(d[col])
    m = smf.mixedlm("lr ~ 1", d, groups=d["sesion"]).fit(reml=True)
    lo, hi = m.conf_int().loc["Intercept"]
    ses = d.groupby("sesion").lr.mean()
    return (f"{np.exp(m.params['Intercept']):.2f} [{np.exp(lo):.2f}–{np.exp(hi):.2f}]; "
            f">1 en {int((d[col] > 1).sum())}/{len(d)} hablante-sesión y {int((ses > 0).sum())}/{len(ses)} sesiones")


def main(root, f0csv):
    f0 = pd.read_csv(f0csv)
    f0["base"] = f0.sesion.apply(lambda s: os.path.basename(str(s).replace("\\", "/")).replace(".transcription.parquet", ""))
    f0 = f0.drop_duplicates("base").set_index("base")
    trans = {}
    for p in glob.glob(os.path.join(root, "**", "*.transcription.parquet"), recursive=True):
        trans.setdefault(os.path.basename(p).replace(".transcription.parquet", ""), p)
    filas = []
    for f in sorted(glob.glob(os.path.join("mirada", "*.mirada.csv"))):
        base = os.path.basename(f).replace(".mirada.csv", "")
        if base in EXCLUIR or base not in f0.index or base not in trans:
            continue
        lag = lag_sesion(f0.loc[base])
        if np.isnan(lag):
            continue
        d = pd.read_csv(f, usecols=["t_video_s", "track_id", "clase", "target"])
        t = pd.read_parquet(trans[base]).dropna(subset=["speaker", "start", "end"])
        t["sp"] = pd.to_numeric(t.speaker.astype(str).str.replace("SPEAKER_", "", regex=False), errors="coerce")
        t = t.dropna(subset=["sp"]); t["sp"] = t.sp.astype(int)
        d["spk"] = hablante_unico(t, d.t_video_s.values + lag)
        for k in sorted(t.sp.unique()):
            L = d[d.track_id != k]
            on = L[L.spk == k]
            off_orig = L[(L.spk != k) & (L.spk != 0)]
            off_oy = off_orig[off_orig.track_id != off_orig.spk]       # sin el hablante activo
            if len(on) < 30 or len(off_oy) < 30:
                continue
            row = {"sesion": base, "hablante": k, "frames_habla": len(on),
                   "frames_otro_orig": len(off_orig), "frames_otro_oyentes": len(off_oy)}
            for nom, col_on, a, b in [("target", on.target == k, off_orig.target == k, off_oy.target == k),
                                      ("clase", on.clase.astype(str) == str(k),
                                       off_orig.clase.astype(str) == str(k), off_oy.clase.astype(str) == str(k))]:
                p1 = col_on.mean()
                row[f"p_otro_orig_{nom}"] = a.mean(); row[f"p_otro_oyentes_{nom}"] = b.mean()
                row[f"razon_orig_{nom}"] = p1 / a.mean() if a.mean() > 0 else np.nan
                row[f"razon_oyentes_{nom}"] = p1 / b.mean() if b.mean() > 0 else np.nan
            filas.append(row)
    H = pd.DataFrame(filas)
    H.to_csv("validez_oyentes_puros.csv", index=False, encoding="utf-8-sig")
    print(f"Sesiones: {H.sesion.nunique()}  hablante-sesión: {len(H)}\n")
    for nom, etiqueta in [("target", "geometría publicada"), ("clase", "clases recalculadas")]:
        print(f"[{etiqueta}]")
        print(f"   original (denominador incluye al hablante activo): {razon_geom(H, f'razon_orig_{nom}')}")
        print(f"   solo oyentes:                                       {razon_geom(H, f'razon_oyentes_{nom}')}")
        print(f"   p_off medio: original {H[f'p_otro_orig_{nom}'].mean():.3f} vs oyentes {H[f'p_otro_oyentes_{nom}'].mean():.3f}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
