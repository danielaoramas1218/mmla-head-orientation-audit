"""
Revisión Q1, punto M2: ¿la clase "down" se comporta como mirar la mesa/materiales,
o como ruido (oclusión de la cara por la mano, postura)?  Análisis exploratorio, sin videos.

Predicciones si "down" = atención a los materiales:
  P1  sincronía: dos personas miran abajo a la vez más de lo esperado por azar
      (lift observado vs. nulo por desplazamiento circular). La oclusión por la mano es
      individual y no debería sincronizarse.
  P2  la sincronía se mantiene al quitar los frames con la mano sobre la cara.
  P3  hablar vs. escuchar: la proporción "down" difiere entre roles (descriptivo).
  P4  mano en la cara: proporción de frames "down" con una mano dentro del recuadro facial,
      frente a frames que no son "down" (cuánto de la clase es oclusión).
  P5  condición A vs. B (Planning Poker usa cartas): descriptivo, pareado por grupo.

Uso (en la carpeta f0, con mirada/ de la especificación principal 30°/15°):
    python validez_down.py "C:\\...\\RECIFE" f0_resultados.csv
Salidas: validez_down_personas.csv, validez_down_sesiones.csv, validez_down_resumen.txt,
         validez_down_muestra_mano.csv (frames marcados "mano en la cara", para revisar en los videos renderizados)
"""
import sys, os, glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from validez_oyente_hablante import lag_sesion, hablante_unico, EXCLUIR

N_NULO = 200
MIN_SHIFT_S = 30.0
rng = np.random.default_rng(20261116)


def mano_en_cara(hands_path, fd_path):
    fd = pd.read_parquet(fd_path, columns=["frame", "track_id", "min_x", "max_x", "min_y", "max_y"])
    fd = fd.drop_duplicates(["frame", "track_id"])
    w = 0.0; h = 0.0
    fd["x0"], fd["x1"], fd["y0"], fd["y1"] = fd.min_x - w, fd.max_x + w, fd.min_y - h, fd.max_y + h
    hd = pd.read_parquet(hands_path)
    xs = [c for c in hd.columns if c.endswith("_x")]; ys = [c.replace("_x", "_y") for c in xs]
    m = hd.merge(fd[["frame", "track_id", "x0", "x1", "y0", "y1"]], on=["frame", "track_id"])
    X, Y = m[xs].values, m[ys].values
    dentro = ((X >= m.x0.values[:, None]) & (X <= m.x1.values[:, None]) &
              (Y >= m.y0.values[:, None]) & (Y <= m.y1.values[:, None])).sum(axis=1) >= 5
    m["mano"] = dentro
    return m.groupby(["frame", "track_id"]).mano.any().rename("mano").reset_index()


def lift_pares(D, fps):
    """D: DataFrame frames x personas con 1/0 (down) y NaN. Devuelve lift observado y nulo medio."""
    cols = list(D.columns); obs, nul = [], []
    min_shift = int(MIN_SHIFT_S * fps); n = len(D)
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            x, y = D[cols[a]].values, D[cols[b]].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 500 or np.nanmean(x) == 0 or np.nanmean(y) == 0:
                continue
            def lift(xx, yy):
                k = ~(np.isnan(xx) | np.isnan(yy))
                px, py, pxy = xx[k].mean(), yy[k].mean(), (xx[k] * yy[k]).mean()
                return pxy / (px * py) if px > 0 and py > 0 else np.nan
            obs.append(lift(x, y))
            sh = rng.integers(min_shift, n - min_shift, N_NULO)
            nul.append(np.nanmean([lift(x, np.roll(y, s)) for s in sh]))
    return np.nanmean(obs) if obs else np.nan, np.nanmean(nul) if nul else np.nan, len(obs)


def main(root, f0csv):
    f0 = pd.read_csv(f0csv)
    f0["base"] = f0.sesion.apply(lambda s: os.path.basename(str(s).replace("\\", "/")).replace(".transcription.parquet", ""))
    f0 = f0.drop_duplicates("base").set_index("base")
    def indice(sufijo):
        d = {}
        for p in glob.glob(os.path.join(root, "**", f"*{sufijo}"), recursive=True):
            d.setdefault(os.path.basename(p).replace(sufijo, "").rstrip("._"), p)
        return d
    trans, hands, fds = indice(".transcription.parquet"), indice("handsTracker.parquet"), indice("faceDirection.parquet")
    P, S, M = [], [], []
    for f in sorted(glob.glob(os.path.join("mirada", "*.mirada.csv"))):
        base = os.path.basename(f).replace(".mirada.csv", "")
        if base in EXCLUIR or base not in f0.index or base not in trans:
            continue
        lag = lag_sesion(f0.loc[base])
        if np.isnan(lag):
            continue
        fps = float(f0.loc[base]["fps_real(frames/audio)"])
        print("  ", base, flush=True)
        d = pd.read_csv(f, usecols=["frame", "t_video_s", "track_id", "clase"])
        d["down"] = (d.clase.astype(str) == "abajo").astype(float)
        if base in hands and base in fds:
            d = d.merge(mano_en_cara(hands[base], fds[base]), on=["frame", "track_id"], how="left")
            d["mano"] = d.mano.fillna(False).astype(bool)
        else:
            d["mano"] = False
        t = pd.read_parquet(trans[base]).dropna(subset=["speaker", "start", "end"])
        t["sp"] = pd.to_numeric(t.speaker.astype(str).str.replace("SPEAKER_", "", regex=False), errors="coerce")
        t = t.dropna(subset=["sp"]); t["sp"] = t.sp.astype(int)
        fl = d[d.mano]
        if len(fl):
            M.append(fl.sample(min(3, len(fl)), random_state=1)[["frame", "t_video_s", "track_id", "clase"]].assign(sesion=base))
        d["spk"] = hablante_unico(t, d.t_video_s.values + lag)
        for k, g in d.groupby("track_id"):
            hab, esc = g[g.spk == k], g[(g.spk > 0) & (g.spk != k)]
            dn, nd = g[g.down == 1], g[g.down == 0]
            P.append({"sesion": base, "condicion": base[-1], "persona": int(k),
                      "down_total": g.down.mean(),
                      "down_hablando": hab.down.mean() if len(hab) >= 30 else np.nan,
                      "down_escuchando": esc.down.mean() if len(esc) >= 30 else np.nan,
                      "mano_si_down": dn.mano.mean() if len(dn) >= 30 else np.nan,
                      "mano_si_no_down": nd.mano.mean() if len(nd) >= 30 else np.nan})
        D = d.pivot_table(index="frame", columns="track_id", values="down")
        D = D.reindex(range(int(d.frame.min()), int(d.frame.max()) + 1))
        Dm = d.assign(down=np.where(d.mano, np.nan, d.down)).pivot_table(index="frame", columns="track_id", values="down")
        Dm = Dm.reindex(D.index)
        lo, ln, npares = lift_pares(D, fps)
        lom, lnm, _ = lift_pares(Dm, fps)
        S.append({"sesion": base, "condicion": base[-1], "down_sesion": d.down.mean(), "pares": npares,
                  "lift_obs": lo, "lift_nulo": ln, "lift_obs_sin_mano": lom, "lift_nulo_sin_mano": lnm})
    P, S = pd.DataFrame(P), pd.DataFrame(S)
    if M:
        pd.concat(M).to_csv("validez_down_muestra_mano.csv", index=False, encoding="utf-8-sig")
    P.to_csv("validez_down_personas.csv", index=False, encoding="utf-8-sig")
    S.to_csv("validez_down_sesiones.csv", index=False, encoding="utf-8-sig")

    out = [f"Sesiones: {len(S)}  persona-sesión: {len(P)}\n"]
    r = (S.lift_obs / S.lift_nulo).dropna(); rm = (S.lift_obs_sin_mano / S.lift_nulo_sin_mano).dropna()
    w = wilcoxon(S.lift_obs - S.lift_nulo); wm = wilcoxon((S.lift_obs_sin_mano - S.lift_nulo_sin_mano).dropna())
    out.append(f"P1 sincronía 'down' entre pares: lift observado mediana {S.lift_obs.median():.2f} vs nulo {S.lift_nulo.median():.2f}; "
               f"obs > nulo en {int((r > 1).sum())}/{len(r)} sesiones; Wilcoxon p = {w.pvalue:.2g}\n")
    out.append(f"P2 sin frames con mano en la cara: lift {S.lift_obs_sin_mano.median():.2f} vs nulo {S.lift_nulo_sin_mano.median():.2f}; "
               f"obs > nulo en {int((rm > 1).sum())}/{len(rm)} sesiones; Wilcoxon p = {wm.pvalue:.2g}\n")
    q = P.dropna(subset=["down_hablando", "down_escuchando"])
    wq = wilcoxon(q.down_escuchando, q.down_hablando)
    out.append(f"P3 'down' escuchando {q.down_escuchando.median()*100:.1f}% vs hablando {q.down_hablando.median()*100:.1f}% "
               f"(medianas, n = {len(q)} persona-sesión; escuchando > hablando en {int((q.down_escuchando > q.down_hablando).sum())}); "
               f"Wilcoxon p = {wq.pvalue:.2g}\n")
    m = P.dropna(subset=["mano_si_down", "mano_si_no_down"])
    out.append(f"P4 mano sobre la cara: {m.mano_si_down.median()*100:.1f}% de los frames 'down' vs "
               f"{m.mano_si_no_down.median()*100:.1f}% del resto (medianas, n = {len(m)} persona-sesión)\n")
    S["g"] = S.sesion.str[:-2]
    C = S.pivot(index="g", columns="condicion", values="down_sesion").dropna()
    if {"A", "B"} <= set(C.columns) and len(C) >= 5:
        wc = wilcoxon(C.A, C.B)
        out.append(f"P5 'down' por sesión: A {C.A.median()*100:.1f}% vs B {C.B.median()*100:.1f}% (n = {len(C)} grupos; "
                   f"B > A en {int((C.B > C.A).sum())}); Wilcoxon p = {wc.pvalue:.2g}  [DESCRIPTIVO]\n")
    txt = "".join(out)
    open("validez_down_resumen.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
