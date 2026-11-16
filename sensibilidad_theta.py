"""
Sensibilidad de los umbrales θ_none ("nadie") y θ_down ("abajo") para el paper 1.
Repite mirada_recalc + validez_oyente_hablante con θ_none = 20/30/45° (θ_down = 15°)
y θ_down = 10/15/20° (θ_none = 30°),
cada uno en su propia subcarpeta (no toca tu carpeta mirada/ actual).
θ = 30 debe reproducir los valores del paper (razón geométrica 2.17 [1.95–2.41]).

Uso (en la carpeta f0, donde están mirada_recalc.py y validez_oyente_hablante.py):
    python sensibilidad_theta.py "C:\\...\\RECIFE" f0_resultados.csv
Salida: sensibilidad_theta.txt y sensibilidad_theta.csv
"""
import sys, os, contextlib, io
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mirada_recalc as mr
import validez_oyente_hablante as vz

# (θ_none, θ_down); la primera fila con 30/15 reproduce el paper
COMBOS = [(30.0, 15.0), (20.0, 15.0), (45.0, 15.0), (30.0, 10.0), (30.0, 20.0)]


def razon_geometrica(H, col):
    d = H.dropna(subset=[col]).copy()
    d = d[d[col] > 0]
    d["lr"] = np.log(d[col])
    m = smf.mixedlm("lr ~ 1", d, groups=d["sesion"]).fit(reml=True)
    b = m.params["Intercept"]; lo, hi = m.conf_int().loc["Intercept"]
    return np.exp(b), np.exp(lo), np.exp(hi), int((d[col] > 1).sum()), len(d)


def main(root, f0csv):
    root, f0csv = os.path.abspath(root), os.path.abspath(f0csv)
    base_dir = os.getcwd()
    filas = []
    for th, td in COMBOS:
        carpeta = os.path.join(base_dir, f"sens_none{int(th)}_down{int(td)}")
        os.makedirs(carpeta, exist_ok=True)
        os.chdir(carpeta)
        mr.ANG_NADIE = th; mr.PITCH_ABAJO = td
        print(f"\n=== θ_none = {th:.0f}°, θ_down = {td:.0f}° (puede tardar varios minutos) ===", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            mr.main(root, f0csv)
            vz.main(root, f0csv)
        R = pd.read_csv("mirada_resumen.csv")
        H = pd.read_csv("validez_por_hablante.csv")
        gm, lo, hi, n1, n = razon_geometrica(H, "razon_clase")
        ses = H.groupby("sesion").apply(lambda g: np.exp(np.log(g.razon_clase[g.razon_clase > 0]).mean()))
        fila = {"theta_none": th, "theta_down": td, "abajo_%_min": R["abajo_%"].min(), "abajo_%_max": R["abajo_%"].max(),
                "nadie_%_mediana": R["nadie_%"].median(), "nadie_%_min": R["nadie_%"].min(), "nadie_%_max": R["nadie_%"].max(),
                "a_persona_%_mediana": R["a_persona_%"].median(), "abajo_%_mediana": R["abajo_%"].median(),
                "razon_geom": round(gm, 2), "IC_inf": round(lo, 2), "IC_sup": round(hi, 2),
                "hablante_sesion_>1": f"{n1}/{n}", "sesiones_>1": f"{int((ses > 1).sum())}/{len(ses)}"}
        filas.append(fila)
        print(fila, flush=True)
        os.chdir(base_dir)
    T = pd.DataFrame(filas)
    T = T.drop_duplicates(["theta_none", "theta_down"])
    T.to_csv("sensibilidad_theta.csv", index=False, encoding="utf-8-sig")
    txt = T.to_string(index=False)
    open("sensibilidad_theta.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
