"""
Test contra el azar de la llave hablante <-> identidad (revisión Q1, punto A2).

Para cada sesión:
  1) Permutaciones: entre las 24 asignaciones posibles hablante -> identidad, ¿qué lugar
     ocupa la identidad (k -> k) en correlación media? p_perm = rango / 24.
  2) Desplazamiento circular: se desplaza el habla respecto del video en s segundos
     (s de 60 s hasta T-60 s, cada 10 s) y se RE-OPTIMIZA el desfase en la misma grilla
     de [-8, 8] s. Así el nulo incluye la misma búsqueda de desfase que el análisis real
     (resuelve la circularidad). p_shift = proporción de desplazamientos con r_max >= r_obs.
     También se registra con qué frecuencia el nulo "aprueba" la llave por azar.

Uso (en la carpeta donde está f0_chequeos.py):
    python llave_azar.py "C:\\...\\RECIFE"
Salida: llave_azar.csv, llave_azar_resumen.txt
"""
import sys, os, glob, itertools
import numpy as np
import pandas as pd
from f0_chequeos import find_sibling, audio_duration, mouth_activity, LAGS

PASO_NULO_S = 10.0
MARGEN_S = 60.0


def cargar_transcripcion(tp):
    t = pd.read_parquet(tp).dropna(subset=["speaker", "start", "end"])
    sp = pd.to_numeric(t.speaker.astype(str).str.replace("SPEAKER_", "", regex=False), errors="coerce")
    t = t[sp.notna()].copy(); t["sp"] = sp[sp.notna()].astype(int)
    revisada = t.sp.min() != 0
    if not revisada:
        t["sp"] += 1
    return t, revisada


def habla_base(t, nf, fps, tracks):
    """Habla binaria en frames de video, sin desfase (S0[k] = habla en t=k/fps)."""
    tt = np.arange(nf) / fps
    S = np.zeros((nf, len(tracks)))
    pos = {k: i for i, k in enumerate(tracks)}
    for s, e, k in zip(t.start.values, t.end.values, t.sp.values):
        if k in pos:
            S[(tt >= s) & (tt <= e), pos[k]] = 1
    return S


def matriz_corr(A, S):
    """Correlaciones de Pearson columna a columna con NaN en A (identidades) -> M[hablante, identidad]."""
    n = A.shape[1]
    M = np.full((n, n), np.nan)
    for j in range(n):
        m = ~np.isnan(A[:, j])
        if m.sum() < 30:
            continue
        a = A[m, j]; a = (a - a.mean()) / (a.std() or np.nan)
        Sm = S[m]
        sd = Sm.std(axis=0); sd[sd == 0] = np.nan
        M[:, j] = ((Sm - Sm.mean(axis=0)) * a[:, None]).mean(axis=0) / sd
    return M


def mejor_lag(A, S0, fps, extra=0):
    """Re-optimiza el desfase: S en tiempo de video = S0 desplazado (lag + extra)."""
    best = (-np.inf, None, None)
    for lag in LAGS:
        r = int(round((lag + extra) * fps))
        M = matriz_corr(A, np.roll(S0, -r, axis=0))
        d = np.nanmean(np.diag(M))
        if d > best[0]:
            best = (d, lag, M)
    return best


def pasa_llave(M):
    return all(np.nanargmax(M[i]) == i for i in range(M.shape[0]) if not np.all(np.isnan(M[i])))


def sesion(tp):
    base = tp[: -len(".transcription.parquet")]
    t, revisada = cargar_transcripcion(tp)
    ut, ft = find_sibling(base, ["userTracker.parquet"]), find_sibling(base, ["faceTracker.parquet"])
    if not (ut and ft):
        return None
    u = pd.read_parquet(ut, columns=["frame", "track_id"])
    nf = int(u.frame.max()); tracks = sorted(int(x) for x in u.track_id.unique())
    dur, _ = audio_duration(base)
    fps = nf / dur if dur else nf / float(t.end.max())
    act, _ = mouth_activity(ft, nf)
    tr = [k for k in tracks if k in act.columns]
    A = act[tr].values
    S0 = habla_base(t, nf, fps, tr)
    r_obs, lag, M = mejor_lag(A, S0, fps)
    n = len(tr)
    perms = list(itertools.permutations(range(n)))
    vals = [np.nanmean([M[i, p[i]] for i in range(n)]) for p in perms]
    rango = int(np.sum(np.array(vals) >= vals[perms.index(tuple(range(n)))]))
    nulos, pasa = [], []
    for s in np.arange(MARGEN_S, nf / fps - MARGEN_S, PASO_NULO_S):
        rn, _, Mn = mejor_lag(A, S0, fps, extra=s)
        nulos.append(rn); pasa.append(pasa_llave(Mn))
    nulos = np.array(nulos)
    return {"sesion": os.path.basename(base), "transcripcion_revisada": revisada,
            "lag_s": lag, "r_diag_obs": round(r_obs, 3), "llave_pasa": pasa_llave(M),
            "rango_identidad_de_24": rango, "p_perm": round(rango / len(perms), 3),
            "n_desplazamientos": len(nulos), "r_nulo_mediana": round(np.median(nulos), 3),
            "r_nulo_p95": round(np.percentile(nulos, 95), 3),
            "p_shift": round((np.sum(nulos >= r_obs) + 1) / (len(nulos) + 1), 4),
            "nulo_aprueba_llave_%": round(100 * np.mean(pasa), 1)}


def main(root):
    filas = []
    for tp in sorted(glob.glob(os.path.join(root, "**", "*.transcription.parquet"), recursive=True)):
        if os.path.basename(tp).replace(".transcription.parquet", "")[-1] not in "AB":
            continue
        print("  ", os.path.basename(tp), flush=True)
        r = sesion(tp)
        if r:
            filas.append(r); print("     ", r, flush=True)
    D = pd.DataFrame(filas)
    D.to_csv("llave_azar.csv", index=False, encoding="utf-8-sig")
    V = D[D.llave_pasa]
    txt = (f"Sesiones: {len(D)}; llave aprobada: {len(V)}\n"
           f"Entre las aprobadas: identidad 1.ª de 24 permutaciones en {int((V.rango_identidad_de_24 == 1).sum())}/{len(V)}\n"
           f"r diagonal observado: mediana {V.r_diag_obs.median():.3f} | nulo (desplazado, desfase re-optimizado): "
           f"mediana {V.r_nulo_mediana.median():.3f}, p95 mediana {V.r_nulo_p95.median():.3f}\n"
           f"p_shift < 0.05 en {int((V.p_shift < 0.05).sum())}/{len(V)} sesiones aprobadas; p_shift máx {V.p_shift.max():.4f}\n"
           f"El nulo aprueba la llave por azar en una mediana de {D['nulo_aprueba_llave_%'].median():.1f}% de los desplazamientos "
           f"(rango {D['nulo_aprueba_llave_%'].min():.1f}–{D['nulo_aprueba_llave_%'].max():.1f}%); "
           f"azar teórico de que las 4 filas tengan el máximo en la diagonal ≈ 1/24 = 4.2%\n")
    open("llave_azar_resumen.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
