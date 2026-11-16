"""
Chequeo de validez de constructo de la mirada recalculada (no usa códigos de discurso):
¿los oyentes orientan la cabeza hacia quien habla?

Para cada sesión y cada hablante k compara, en los frames de los oyentes (personas != k):
  p_habla = proporción de frames que apuntan a k mientras k habla SOLO
  p_otro  = proporción de frames que apuntan a k mientras habla SOLO otra persona
  razón   = p_habla / p_otro   (>1 = se mira más a k cuando habla)
Se calcula con dos objetivos:
  - "target":  participante más cercano a la dirección de la cabeza (geometría de ObservedUser)
  - "clase":   mirada recalculada (cuenta solo si la clase es k; "abajo" y "nadie" no cuentan)

Usa el desfase de cada sesión de f0_resultados.csv (se recalcula t aquí, así que
sirve aunque mirada/ se haya generado con la versión anterior del script).
Excluye G21_B (12-07-B) y sesiones sin desfase confiable.

Uso:
    python validez_oyente_hablante.py "C:\\...\\DataSet Fondecyt1211905\\DataSet Fondecyt1211905" f0_resultados.csv
Salida:
    validez_por_hablante.csv   una fila por sesión x hablante
    validez_por_sesion.csv     una fila por sesión
    (resumen en pantalla)
Requisitos: pandas, pyarrow, numpy, scipy
"""
import sys, os, glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXCLUIR = {"12-07-B"}


def lag_sesion(r):
    ok = r.get("llave_OK") in (True, "True")
    parcial = pd.notna(r.get("diag_r_lag_optimo")) and r["diag_r_lag_optimo"] >= 0.2 and 0 <= r["lag_optimo_s"] <= 2
    return float(r["lag_optimo_s"]) if (ok or parcial) else np.nan


def hablante_unico(t, tt):
    spk = np.zeros(len(tt), int); n = np.zeros(len(tt), int)
    for s, e, k in zip(t.start.values, t.end.values, t.sp.values):
        m = (tt >= s) & (tt <= e); spk[m] = k; n[m] += 1
    return np.where(n == 1, spk, 0)


def main(root, f0csv):
    f0 = pd.read_csv(f0csv)
    f0["base"] = f0.sesion.apply(lambda s: os.path.basename(str(s)).replace(".transcription.parquet", ""))
    f0 = f0.drop_duplicates("base").set_index("base")
    trans = {os.path.basename(p).replace(".transcription.parquet", ""): p
             for p in glob.glob(os.path.join(root, "**", "*.transcription.parquet"), recursive=True)}
    filas, ses = [], []
    for f in sorted(glob.glob(os.path.join("mirada", "*.mirada.csv"))):
        base = os.path.basename(f).replace(".mirada.csv", "")
        if base in EXCLUIR or base not in f0.index or base not in trans:
            continue
        lag = lag_sesion(f0.loc[base])
        if np.isnan(lag):
            print(f"  sin desfase confiable, se omite: {base}"); continue
        d = pd.read_csv(f, usecols=["t_video_s", "track_id", "clase", "target"])
        t = pd.read_parquet(trans[base]).dropna(subset=["speaker", "start", "end"])
        t["sp"] = pd.to_numeric(t.speaker.astype(str).str.replace("SPEAKER_", "", regex=False), errors="coerce")
        t = t.dropna(subset=["sp"]); t["sp"] = t.sp.astype(int)
        d["spk"] = hablante_unico(t, d.t_video_s.values + lag)
        cond = base[-1]
        oy = d[(d.spk > 0) & (d.track_id != d.spk)]
        ses.append({"sesion": base, "condicion": cond, "lag_s": lag,
                    "oyente_a_hablante_target_%": round((oy.target == oy.spk).mean() * 100, 1),
                    "oyente_a_hablante_clase_%": round((oy.clase == oy.spk.astype(str)).mean() * 100, 1),
                    "oyente_abajo_%": round((oy.clase == "abajo").mean() * 100, 1),
                    "frames_oyente": len(oy)})
        for k in sorted(t.sp.unique()):
            L = d[d.track_id != k]
            on = L[L.spk == k]; off = L[(L.spk != k) & (L.spk != 0)]
            if len(on) < 30 or len(off) < 30:
                continue
            row = {"sesion": base, "condicion": cond, "hablante": k, "frames_habla": len(on), "frames_otro": len(off)}
            for nom, hit_on, hit_off in [("target", on.target == k, off.target == k),
                                         ("clase", on.clase == str(k), off.clase == str(k))]:
                p1, p0 = hit_on.mean(), hit_off.mean()
                row[f"p_habla_{nom}"] = round(p1, 4); row[f"p_otro_{nom}"] = round(p0, 4)
                row[f"razon_{nom}"] = round(p1 / p0, 3) if p0 > 0 else np.nan
            filas.append(row)
    H = pd.DataFrame(filas); S = pd.DataFrame(ses)
    H.to_csv("validez_por_hablante.csv", index=False, encoding="utf-8-sig")
    S.to_csv("validez_por_sesion.csv", index=False, encoding="utf-8-sig")

    print(f"\nSesiones: {S.sesion.nunique()}   hablante-sesión: {len(H)}")
    for nom in ["target", "clase"]:
        r = H[f"razon_{nom}"].dropna()
        w = wilcoxon(H[f"p_habla_{nom}"], H[f"p_otro_{nom}"])
        print(f"\n[{nom}] razón p_habla/p_otro: mediana {r.median():.2f} (IQR {r.quantile(.25):.2f}–{r.quantile(.75):.2f}); "
              f"razón > 1 en {int((r > 1).sum())}/{len(r)} hablante-sesión; Wilcoxon p = {w.pvalue:.2g}")
    print(f"\nOyente -> hablante (por sesión): target mediana {S['oyente_a_hablante_target_%'].median()}%, "
          f"clase mediana {S['oyente_a_hablante_clase_%'].median()}%  (azar ~33% con 3 compañeros)")
    print("\nPor condición (mediana por sesión):")
    print(S.groupby("condicion")[["oyente_a_hablante_target_%", "oyente_a_hablante_clase_%", "oyente_abajo_%"]].median().to_string())
    S["g"] = S.sesion.str[:-2]
    P = S.pivot(index="g", columns="condicion", values="oyente_a_hablante_clase_%").dropna()
    if len(P) >= 5 and {"A", "B"} <= set(P.columns):
        w = wilcoxon(P["A"], P["B"])
        print(f"\nA vs B pareado (oyente->hablante, clase): n={len(P)} grupos, B>A en {int((P.B > P.A).sum())}, Wilcoxon p = {w.pvalue:.2g}")
        print("  (DESCRIPTIVO: orden e historia de usuario aún no confirmados con los autores del dataset)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
