"""
Recalcula la mirada (objetivo de la orientación de cabeza) desde faceDirection,
corrigiendo los problemas de ObservedUser documentados en la auditoría:

  - sin duplicados ni arrastre de valores: frame sin cara -> objetivo vacío;
  - tasa de frames real (n_frames / duración del audio) y desfase por sesión
    (lag de f0_resultados.csv), para alinear con la transcripción;
  - clase "abajo": pitch más de PITCH_ABAJO grados por debajo de la mediana
    de la persona en la sesión (el pitch que ObservedUser ignora);
  - clase "nadie": la dirección queda a más de ANG_NADIE grados del
    compañero más cercano (en ángulo sobre el círculo de la cámara 360°);
  - asignación a sí mismo -> "nadie".

La geometría horizontal (yaw -> punto del cilindro -> participante más cercano)
es la misma de s07, así que donde ObservedUser es válido debe coincidir.

Uso:
    python mirada_recalc.py "C:\\...\\RECIFE" f0_resultados.csv
Salida:
    mirada/<sesion>.mirada.csv   una fila por (frame, track_id)
    mirada_resumen.csv           una fila por sesión
Requisitos: pip install pandas pyarrow numpy

Los umbrales se fijan ANTES de mirar resultados contra los códigos de discurso.
"""
import sys, os, glob
import numpy as np
import pandas as pd

# --- parámetros (prerregistrar) -------------------------------------------
PITCH_SIGNO_ABAJO = -1     # -1: pitch negativo = cabeza hacia abajo (CONFIRMAR CON VIDEO)
PITCH_ABAJO = 15.0         # grados bajo la mediana personal para clase "abajo"
ANG_NADIE = 30.0           # grados máximos al compañero más cercano
AMPLIFY = 1.2              # factor de s07 sobre yaw (se mantiene para comparabilidad)
W, H = 1920, 1080          # resolución del panorama (dos franjas de 180°)
# ---------------------------------------------------------------------------


def theta_from_pixel(px, py):
    """Ángulo (rad) de la posición en el cilindro, igual que pixel_to_xy + xy_to_cylindrical de s07."""
    x = np.where(py > H // 2, px + W, px).astype(float)
    return 2 * np.pi * x / (2 * W)


def gaze_point_theta(theta, yaw_deg):
    """Ángulo del punto donde el rayo de mirada vuelve a cortar el círculo unitario (geometría de s07)."""
    P = np.stack([np.cos(theta), np.sin(theta)], 1)
    phi = np.radians(yaw_deg * AMPLIFY)
    alpha = theta + np.pi - phi                      # rotación de -P alrededor de -z por phi
    u = np.stack([np.cos(alpha), np.sin(alpha)], 1)
    b = 2 * (P * u).sum(1)
    t = np.where(b < 0, -b, 0.0)                     # raíz mayor de s07: t1 = (-b + |b|)/2
    Q = P + t[:, None] * u
    return np.arctan2(Q[:, 1], Q[:, 0])


def angdiff(a, b):
    return np.degrees(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def process(fd_path, nf, fps, lag, ob_path=None):
    fd = pd.read_parquet(fd_path, columns=["frame", "track_id", "x1", "y1", "min_x", "min_y", "nx", "ny", "pitch", "yaw"])
    fd = fd.drop_duplicates(["frame", "track_id"])
    fd["theta"] = theta_from_pixel((fd.nx + fd.min_x + fd.x1).values, (fd.ny + fd.min_y + fd.y1).values)
    # dp > 0 = cabeza más abajo que su mediana personal (en grados)
    fd["dp"] = PITCH_SIGNO_ABAJO * (fd.pitch - fd.groupby("track_id").pitch.transform("median"))
    # posiciones de cada persona por frame (relleno corto de posición, no de dirección)
    pos = fd.pivot_table(index="frame", columns="track_id", values="theta").reindex(range(1, nf + 1))
    pos = pos.ffill(limit=15).bfill(limit=15)
    pos = pos.fillna(pos.median())                   # posición típica si falta mucho tiempo
    tracks = list(pos.columns)
    fd["gtheta"] = gaze_point_theta(fd.theta.values, fd.yaw.values)
    P = pos.loc[fd.frame.values].values              # (n, k) posiciones en ese frame
    D = angdiff(fd.gtheta.values[:, None], P)
    j = np.nanargmin(D, axis=1)
    fd["target"] = np.array(tracks)[j]
    fd["ang_al_objetivo"] = D[np.arange(len(fd)), j]
    fd["clase"] = fd.target.astype(int).astype(str)
    fd.loc[fd.target == fd.track_id, "clase"] = "nadie"
    fd.loc[fd.ang_al_objetivo > ANG_NADIE, "clase"] = "nadie"
    fd.loc[fd.dp > PITCH_ABAJO, "clase"] = "abajo"
    fd["t_video_s"] = (fd.frame - 1) / fps
    fd["t_transcr_s"] = fd.t_video_s + lag             # tiempo en el reloj de la transcripción
    out = fd[["frame", "t_video_s", "t_transcr_s", "track_id", "clase", "target",
              "ang_al_objetivo", "dp", "pitch", "yaw"]].sort_values(["frame", "track_id"])
    resumen = {}
    if ob_path and os.path.exists(ob_path):
        ob = pd.read_parquet(ob_path)
        col = "observed_user_by_tack_id" if "observed_user_by_tack_id" in ob else "observed_user_by_track_id"
        ob = ob.drop_duplicates(["frame", "track_id"], keep=False)  # descarta los pares con arrastre
        m = out.merge(ob[["frame", "track_id", col]], on=["frame", "track_id"])
        # índice 0-3 de s07 sobre los track_id ordenados presentes en el caché
        trk_sorted = sorted(ob.track_id.unique())
        m["ob_track"] = [trk_sorted[i] if 0 <= i < len(trk_sorted) else -1 for i in m[col].astype(int)]
        resumen["acuerdo_target_vs_ObservedUser_%"] = round((m.target == m.ob_track).mean() * 100, 1)
        resumen["ObservedUser_asigna_persona_cuando_recalc_abajo_%"] = round(
            (m[m.clase == "abajo"].ob_track > 0).mean() * 100, 1) if (m.clase == "abajo").any() else None
    return out, resumen


def main(root, f0csv):
    f0 = pd.read_csv(f0csv)
    f0["base"] = f0.sesion.apply(lambda s: os.path.basename(str(s)).replace(".transcription.parquet", ""))
    os.makedirs("mirada", exist_ok=True)
    rows = []
    for fp in sorted(glob.glob(os.path.join(root, "**", "*faceDirection.parquet"), recursive=True)):
        base = os.path.basename(fp).replace(".faceDirection.parquet", "").replace("_faceDirection.parquet", "")
        r = f0[f0.base == base]
        if r.empty or pd.isna(r.iloc[0].get("fps_real(frames/audio)")):
            print(f"  (sin datos F0) {base}"); continue
        r = r.iloc[0]
        cond = base[-1]
        if cond not in "AB":
            continue
        fps, nf = float(r["fps_real(frames/audio)"]), int(r["n_frames"])
        # lag de la sesión; se acepta también con llave parcial si la sincronía es buena (G02_B, G10_B)
        ok = r.get("llave_OK") in (True, "True") or (pd.notna(r.get("diag_r_lag_optimo")) and r["diag_r_lag_optimo"] >= 0.2
                                                     and 0 <= r["lag_optimo_s"] <= 2)
        lag = float(r["lag_optimo_s"]) if ok else np.nan
        ob = os.path.join(os.path.dirname(fp), base + ".ObservedUser.parquet")
        print(f"{base}: fps={fps:.3f} lag={lag:+.2f}s", flush=True)
        out, res = process(fp, nf, fps, 0.0 if np.isnan(lag) else lag, ob)
        out.to_csv(os.path.join("mirada", base + ".mirada.csv"), index=False)
        prop = out.clase.value_counts(normalize=True)
        row = {"sesion": base, "condicion": cond, "llave_OK": r.get("llave_OK"), "lag_s": lag,
               "frames_persona_validos": len(out), "frames_esperados": nf * 4,
               "cobertura_%": round(len(out) / (nf * 4) * 100, 1),
               "abajo_%": round(prop.get("abajo", 0) * 100, 1),
               "nadie_%": round(prop.get("nadie", 0) * 100, 1),
               "a_persona_%": round(prop.drop(["abajo", "nadie"], errors="ignore").sum() * 100, 1)}
        row.update(res)
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv("mirada_resumen.csv", index=False, encoding="utf-8-sig")
    print("\nListo: carpeta mirada/ y mirada_resumen.csv")
    print(s.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
