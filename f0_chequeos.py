"""
F0 — Chequeos de integración multimodal por sesión (dataset Fondecyt 1211905).

Recorre una carpeta (recursivamente), encuentra cada sesión por su
`*.transcription.parquet` y calcula, por sesión:

  1. Participantes y cobertura de cara (userTracker / faceTracker / faceDirection).
  2. Llave hablante (transcripción) <-> track_id (video): correlación entre
     movimiento de boca y habla diarizada; ¿la diagonal es el máximo?
  3. Tasa de frames real (frames / duración del audio) y, si hay .mp4 y
     OpenCV instalado, la fps que declara el contenedor.
  4. Desfase transcripción-video (lag que maximiza la diagonal).
  5. Duplicados (frame, track_id) en ObservedUser.
  6. Proporción de frames con la cabeza >15° y >20° por debajo de la
     mediana de cada persona (pitch).

Uso (Windows, desde la carpeta donde está este script):
    python f0_chequeos.py "C:\\ruta\\a\\RECIFE"
Salida: f0_resultados.csv (una fila por sesión) y f0_matrices.txt (matrices
4x4 hablante x track). No modifica ningún archivo de datos.

Requisitos: pip install pandas pyarrow numpy
Opcional:   pip install opencv-python   (para leer la fps del .mp4)
"""
import sys, os, glob, wave, contextlib, traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LAGS = np.arange(-8, 8.01, 0.25)


def find_sibling(base, kinds):
    for k in kinds:
        for sep in (".", "_"):
            p = f"{base}{sep}{k}"
            if os.path.exists(p):
                return p
    return None


def audio_duration(base):
    p = find_sibling(base, ["wav"]) or (base + ".wav" if os.path.exists(base + ".wav") else None)
    if not p:
        return None, None
    try:
        with contextlib.closing(wave.open(p, "rb")) as w:
            return w.getnframes() / float(w.getframerate()), p
    except Exception:
        try:
            import soundfile as sf
            return sf.info(p).duration, p
        except Exception:
            return None, p


def mp4_fps(base):
    p = base + ".mp4"
    if not os.path.exists(p):
        return None, None
    try:
        import cv2
        cap = cv2.VideoCapture(p)
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return fps, n
    except Exception:
        return None, None


def mouth_activity(face_path, nf):
    cols = ["frame", "track_id", "13_y", "14_y", "10_y", "152_y"]
    ft = pq.read_table(face_path, columns=cols).to_pandas()
    ft["ap"] = (ft["14_y"] - ft["13_y"]).abs() / (ft["152_y"] - ft["10_y"]).abs().replace(0, np.nan)
    A = ft.pivot_table(index="frame", columns="track_id", values="ap").reindex(range(1, nf + 1))
    cov = A.notna().mean()
    A = A.interpolate(limit=5)
    act = A.rolling(7, center=True, min_periods=4).std()
    return act, cov


def speak_matrix(t, nf, fps, lag, tracks):
    tt = np.arange(nf) / fps + lag
    S = np.zeros((nf, len(tracks)))
    pos = {k: i for i, k in enumerate(tracks)}
    for s, e, k in zip(t.start.values, t.end.values, t.sp.values):
        if k in pos:
            S[(tt >= s) & (tt <= e), pos[k]] = 1
    return S


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def check_session(tp):
    base = tp[: -len(".transcription.parquet")]
    name = os.path.relpath(base)
    row = {"sesion": name, "condicion": base[-1] if base[-2] == "-" else ""}
    t = pd.read_parquet(tp)
    t = t.dropna(subset=["speaker", "start", "end"])
    spk = t.speaker.astype(str).str.replace("SPEAKER_", "", regex=False)
    spk = pd.to_numeric(spk, errors="coerce")  # 'None', 'nan', '' -> NaN
    row = {**row, "segmentos_sin_speaker": int(spk.isna().sum())}
    t = t[spk.notna()].copy()
    t["sp"] = spk[spk.notna()].astype(int)
    if t["sp"].min() == 0:  # transcripción sin revisar (SPEAKER_00..03)
        t["sp"] += 1
        row["transcripcion"] = "SIN_REVISAR(SPEAKER_xx)"
    else:
        row["transcripcion"] = "revisada"
    row["n_segmentos"] = len(t)
    row["hablantes"] = t.sp.nunique()

    ut = find_sibling(base, ["userTracker.parquet"])
    ft = find_sibling(base, ["faceTracker.parquet"])
    fd = find_sibling(base, ["faceDirection.parquet"])
    ob = find_sibling(base, ["ObservedUser.parquet"])
    row.update({"tiene_userTracker": bool(ut), "tiene_faceTracker": bool(ft),
                "tiene_faceDirection": bool(fd), "tiene_ObservedUser": bool(ob)})
    if not ut:
        return row, None
    u = pd.read_parquet(ut, columns=["frame", "track_id"])
    nf = int(u.frame.max())
    tracks = sorted(int(x) for x in u.track_id.unique())
    row["n_frames"] = nf
    row["track_ids"] = "-".join(map(str, tracks))
    row["frames_con_4_personas_%"] = round((u.groupby("frame").track_id.nunique() == 4).mean() * 100, 1)

    dur, _ = audio_duration(base)
    row["dur_audio_s"] = round(dur, 2) if dur else None
    row["dur_transcr_fin_s"] = round(float(t.end.max()), 2)
    fps_c, nf_c = mp4_fps(base)
    row["fps_mp4_cv2"] = round(fps_c, 3) if fps_c else None
    if dur:
        fps = nf / dur
        row["fps_real(frames/audio)"] = round(fps, 3)
    else:
        fps = nf / float(t.end.max())  # aproximación si no hay .wav
        row["fps_real(frames/audio)"] = None
        row["fps_aprox(frames/fin_transcr)"] = round(fps, 3)

    mtxt = None
    if ft:
        act, cov = mouth_activity(ft, nf)
        row["cobertura_cara_min_%"] = round(cov.min() * 100, 1)
        tr = [k for k in tracks if k in act.columns]
        actv = act[tr].values
        def diag(f, lag, sl=slice(None)):
            S = speak_matrix(t, nf, f, lag, tr)
            return np.nanmean([corr(actv[sl, i], S[sl, i]) for i in range(len(tr))])
        res = [(lag, diag(fps, lag)) for lag in LAGS]
        lag_best, r_best = max(res, key=lambda x: -1 if np.isnan(x[1]) else x[1])
        row["lag_optimo_s"] = lag_best
        row["diag_r_lag0"] = round(diag(fps, 0), 3)
        row["diag_r_lag_optimo"] = round(r_best, 3)
        for k in range(3):
            sl = slice(k * nf // 3, (k + 1) * nf // 3)
            rk = [(lag, diag(fps, lag, sl)) for lag in np.arange(-4, 4.01, 0.25)]
            row[f"lag_tercio{k+1}"] = max(rk, key=lambda x: -1 if np.isnan(x[1]) else x[1])[0]
        if fps_c and abs(fps_c - fps) > 0.05:
            row["diag_r_con_fps_mp4"] = round(diag(fps_c, 0), 3)
        S = speak_matrix(t, nf, fps, lag_best, tr)
        M = pd.DataFrame([[corr(actv[:, j], S[:, i]) for j in range(len(tr))] for i in range(len(tr))],
                         index=[f"spk{k}" for k in tr], columns=[f"trk{k}" for k in tr])
        asign = {int(i[3:]): (int(M.loc[i].idxmax()[3:]) if M.loc[i].notna().any() else -1) for i in M.index}
        row["asignacion_speaker->track"] = " ".join(f"{a}->{b}" for a, b in asign.items())
        row["llave_OK"] = all(a == b for a, b in asign.items()) and len(set(asign.values())) == len(asign)
        mtxt = f"== {name}  (lag {lag_best:+.2f}s)\n{M.round(3).to_string()}\n"

    if ob:
        o = pd.read_parquet(ob)
        row["ObservedUser_filas"] = len(o)
        row["ObservedUser_duplicados"] = int(o.duplicated(["frame", "track_id"]).sum())
        row["ObservedUser_tracks"] = o.track_id.nunique()
    if fd:
        d = pd.read_parquet(fd, columns=["frame", "track_id", "pitch", "yaw"])
        row["faceDirection_filas"] = len(d)
        dp = d.pitch - d.groupby("track_id").pitch.transform("median")
        for th in (15, 20):
            row[f"pitch_bajo_{th}_%"] = round((dp < -th).mean() * 100, 1)
            row[f"pitch_sobre_{th}_%"] = round((dp > th).mean() * 100, 1)
    return row, mtxt


def main(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.transcription.parquet"), recursive=True))
    print(f"Sesiones encontradas: {len(files)}")
    rows, mats = [], []
    for i, tp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.relpath(tp, root)}", flush=True)
        try:
            r, m = check_session(tp)
        except Exception as e:
            r, m = {"sesion": os.path.relpath(tp), "ERROR": f"{type(e).__name__}: {e}"}, None
            traceback.print_exc()
        rows.append(r)
        if m:
            mats.append(m)
    df = pd.DataFrame(rows)
    df.to_csv("f0_resultados.csv", index=False, encoding="utf-8-sig")
    with open("f0_matrices.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(mats))
    print("\nListo: f0_resultados.csv y f0_matrices.txt")
    cols = [c for c in ["sesion", "llave_OK", "fps_real(frames/audio)", "fps_mp4_cv2", "lag_optimo_s",
                        "diag_r_lag_optimo", "ObservedUser_duplicados", "pitch_bajo_15_%"] if c in df]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
