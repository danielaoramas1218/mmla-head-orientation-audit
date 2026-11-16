"""
Genera la muestra para la validación manual de la mirada (F1).

Lee las salidas de mirada_recalc.py (carpeta mirada/) y produce:
  planilla_anotacion.csv : lo que ven los anotadores (SIN las etiquetas automáticas)
  clave_oculta.csv       : etiquetas automáticas y estrato de muestreo (NO entregar a anotadores)

Diseño: por cada sesión GO y cada participante, 3 ventanas de 1 s:
  - 1 al azar entre todos sus frames con cara           (estrato "azar")
  - 1 entre sus frames clasificados "abajo"            (estrato "abajo"; si no hay, al azar)
  - 1 entre sus frames clasificados "nadie"            (estrato "nadie"; si no hay, al azar)
El estrato queda en la clave para ponderar al estimar prevalencias.
Las ventanas de una misma persona quedan separadas por al menos 10 s.

IMPORTANTE: "ventana_video" usa t = (frame-1) / fps_real (n_frames/duración del audio).
Si el reproductor muestra una duración del .mp4 distinta a la del audio, navegar por
la columna "frame" en vez del tiempo.

Uso:  python muestra_anotacion.py
      (en la carpeta donde están mirada/ y f0_resultados.csv)
"""
import os, glob
import numpy as np
import pandas as pd

SEED = 20261029
EXCLUIR = {"12-07-B"}          # G21_B: transcripción sin revisar (NO-GO)
SEP_MIN_S = 10.0
rng = np.random.default_rng(SEED)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join("mirada", "*.mirada.csv"))):
        base = os.path.basename(f).replace(".mirada.csv", "")
        if base in EXCLUIR:
            continue
        d = pd.read_csv(f)
        cond = base[-1]
        for tid, g in d.groupby("track_id"):
            g = g.sort_values("frame")
            elegidos = []
            for estrato in ["abajo", "nadie", "azar"]:
                cand = g if estrato == "azar" else g[g.clase == estrato]
                if cand.empty:
                    cand, estrato_real = g, "azar(reemplazo)"
                else:
                    estrato_real = estrato
                # separar de las ya elegidas y evitar bordes (1 s) de la sesión
                for t0 in elegidos:
                    cand = cand[(cand.t_video_s - t0).abs() >= SEP_MIN_S]
                cand = cand[(cand.t_video_s > 1) & (cand.t_video_s < g.t_video_s.max() - 1)]
                if cand.empty:
                    continue
                r = cand.iloc[rng.integers(len(cand))]
                elegidos.append(r.t_video_s)
                rows.append({"sesion": base, "grupo": base, "condicion": cond, "persona": int(tid),
                             "t_centro_s": round(r.t_video_s, 2), "frame": int(r.frame),
                             "estrato": estrato_real, "clase_recalc": r.clase, "target_bruto": int(r.target),
                             "ang_al_objetivo": round(r.ang_al_objetivo, 1), "dp": round(r.dp, 1)})
    k = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    k.insert(0, "id", [f"V{i+1:03d}" for i in range(len(k))])
    k["ventana_video"] = k.t_centro_s.apply(lambda t: f"{mmss(t - 0.5)} – {mmss(t + 0.5)}")
    planilla = k[["id", "sesion", "persona", "ventana_video", "t_centro_s", "frame"]].copy()
    planilla["video"] = planilla.sesion + ".mp4"
    planilla["objetivo (1/2/3/4/mesa/nadie/no_visible)"] = ""
    planilla["seguridad (1=baja,2=media,3=alta)"] = ""
    planilla["anotador"] = ""
    planilla["comentario"] = ""
    planilla.to_csv("planilla_anotacion.csv", index=False, encoding="utf-8-sig")
    k.to_csv("clave_oculta.csv", index=False, encoding="utf-8-sig")
    print(f"Ventanas: {len(k)}  (sesiones: {k.sesion.nunique()}, personas-sesión: {k.groupby(['sesion','persona']).ngroups})")
    print(k.estrato.value_counts().to_string())
    print(k.groupby("condicion").size().to_string())
    print("\nplanilla_anotacion.csv -> anotadores | clave_oculta.csv -> NO compartir")


if __name__ == "__main__":
    main()
