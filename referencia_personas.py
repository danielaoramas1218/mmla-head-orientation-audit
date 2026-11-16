"""
Genera una imagen de referencia por sesión con el número (1-4) de cada persona,
dibujando los recuadros de userTracker sobre un frame del video.

Uso:
    python referencia_personas.py "C:\\...\\RECIFE" "C:\\...\\carpeta_con_los_mp4"
Salida: carpeta referencias/ con <sesion>_personas.png (3 frames por sesión: inicio, medio, final)

Requisitos: pip install opencv-python pandas pyarrow
"""
import sys, os, glob
import cv2
import pandas as pd

COLORES = {1: (0, 0, 255), 2: (0, 200, 0), 3: (255, 200, 0), 4: (0, 200, 255)}


def buscar_video(carpeta, base):
    # acepta "<base>.mp4" y nombres como "Grupo01_A_<base>.results.mp4"
    for ext in (".mp4", ".MP4", ".mov", ".avi"):
        c = glob.glob(os.path.join(carpeta, "**", f"*{base}*{ext}"), recursive=True)
        c = [p for p in c if base in os.path.basename(p)]
        if c:
            return sorted(c, key=len)[0]
    return None


def main(root, videos):
    os.makedirs("referencias", exist_ok=True)
    trackers = glob.glob(os.path.join(root, "**", "*.userTracker.parquet"), recursive=True)
    faltan = []
    for ut in sorted(trackers):
        base = os.path.basename(ut).replace(".userTracker.parquet", "")
        if base[-1] not in "AB":
            continue
        vid = buscar_video(videos, base)
        if not vid:
            faltan.append(base); continue
        u = pd.read_parquet(ut, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
        cap = cv2.VideoCapture(vid)
        nfv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        paneles = []
        for frac in (0.1, 0.5, 0.9):
            f = int(u.frame.max() * frac)
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ok, img = cap.read()
            if not ok:
                continue
            for _, r in u[u.frame == f].iterrows():
                tid = int(r.track_id); col = COLORES.get(tid, (255, 255, 255))
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), col, 4)
                cv2.putText(img, f"PERSONA {tid}", (int(r.x1) + 8, int(r.y1) + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4)
            cv2.putText(img, f"{base}  frame {f}", (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            paneles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        cap.release()
        if paneles:
            cv2.imwrite(os.path.join("referencias", f"{base}_personas.png"), cv2.vconcat(paneles))
            aviso = "" if abs(nfv - u.frame.max()) <= 2 else f"  ATENCIÓN: video {nfv} frames vs tracker {u.frame.max()}"
            print(f"OK {base}{aviso}")
    if faltan:
        print("\nSin video encontrado para:", ", ".join(faltan))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
