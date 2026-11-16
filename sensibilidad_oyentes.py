"""
Tabla II con la definición corregida (solo oyentes en el denominador).
Reutiliza las carpetas sens_none*_down* que ya generó sensibilidad_theta.py
(si alguna falta, la recalcula). Imprime y guarda la tabla en LaTeX lista para pegar.

Uso (en la carpeta f0):
    python sensibilidad_oyentes.py "C:\\...\\RECIFE" f0_resultados.csv
Salidas: sensibilidad_oyentes.csv, tabla_sensibilidad.tex
"""
import sys, os, contextlib, io
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mirada_recalc as mr
import validez_oyentes_puros as vop

COMBOS = [(30.0, 15.0), (20.0, 15.0), (45.0, 15.0), (30.0, 10.0), (30.0, 20.0)]


def gm_ic(H, col):
    d = H[H[col] > 0].dropna(subset=[col]).copy(); d["lr"] = np.log(d[col])
    m = smf.mixedlm("lr ~ 1", d, groups=d["sesion"]).fit(reml=True)
    lo, hi = m.conf_int().loc["Intercept"]
    ses = d.groupby("sesion").lr.mean()
    return np.exp(m.params["Intercept"]), np.exp(lo), np.exp(hi), int((d[col] > 1).sum()), len(d), int((ses > 0).sum()), len(ses)


def main(root, f0csv):
    root, f0csv = os.path.abspath(root), os.path.abspath(f0csv)
    base = os.getcwd(); filas = []
    for th, td in COMBOS:
        carpeta = os.path.join(base, f"sens_none{int(th)}_down{int(td)}")
        os.makedirs(carpeta, exist_ok=True); os.chdir(carpeta)
        print(f"=== θ_none = {th:.0f}°, θ_down = {td:.0f}° ===", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            if not os.path.exists("mirada_resumen.csv"):
                mr.ANG_NADIE = th; mr.PITCH_ABAJO = td
                mr.main(root, f0csv)
            vop.main(root, f0csv)
        R = pd.read_csv("mirada_resumen.csv"); H = pd.read_csv("validez_oyentes_puros.csv")
        g, lo, hi, n1, n, s1, s = gm_ic(H, "razon_oyentes_clase")
        filas.append({"theta_none": th, "theta_down": td, "abajo": R["abajo_%"].median(), "nadie": R["nadie_%"].median(),
                      "gm": g, "lo": lo, "hi": hi, "hs": f"{n1}/{n}", "ses": f"{s1}/{s}"})
        print(filas[-1], flush=True); os.chdir(base)
    T = pd.DataFrame(filas); T.to_csv("sensibilidad_oyentes.csv", index=False, encoding="utf-8-sig")

    def fila(r):
        return (f"${r.theta_none:.0f}^\\circ$ & ${r.theta_down:.0f}^\\circ$ & {r.abajo:.1f}\\,\\% & {r.nadie:.1f}\\,\\% & "
                f"{r.gm:.2f} [{r.lo:.2f}--{r.hi:.2f}] & {r.hs} & {r.ses} \\\\")
    lineas = [r"\begin{table}[t]", r"\centering",
              r"\caption{Sensitivity of the recomputed measure to the \emph{no-one} ($\theta_{\text{none}}$) and \emph{down} ($\theta_{\text{down}}$) thresholds (31 sessions, 124 speaker--session combinations). The first row is the main specification.}",
              r"\label{tab:theta}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{@{}rrrrrcc@{}}", r"\toprule",
              r"$\theta_{\text{none}}$ & $\theta_{\text{down}}$ & \emph{Down}$^{a}$ & \emph{No-one}$^{a}$ & Ratio [95\,\% CI]$^{b}$ & Spk.--sess. $>1$ & Sess. $>1$ \\",
              r"\midrule", fila(T.iloc[0]), r"\midrule"] + [fila(r) for _, r in T.iloc[1:].iterrows()] + [
              r"\midrule",
              r"\multicolumn{7}{@{}l}{\footnotesize $^{a}$Median share of frames across 32 sessions. $^{b}$Geometric mean ratio, listeners only, from a linear}\\",
              r"\multicolumn{7}{@{}l}{\footnotesize mixed model of the log ratio with a random intercept per session.}\\",
              r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    tex = "\n".join(lineas)
    open("tabla_sensibilidad.tex", "w", encoding="utf-8").write(tex)
    print("\n----- copia desde aquí -----\n" + tex)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
