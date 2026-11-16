# %% [markdown]
# # Figuras del paper de validación de la mirada (MMLA)
#
# Genera 5 figuras listas para IEEEtran (PDF vectorial + PNG 300 dpi):
#
# | Figura | Qué muestra | Datos |
# |---|---|---|
# | 1 | Llave hablante↔participante: matrices de correlación boca–habla (una sesión que calza y una que falla) | `f0_matrices.txt` |
# | 2 | Validez: los oyentes miran a quien habla (razón por sesión y por hablante, 31 sesiones) | `validez_por_hablante.csv` |
# | 3 | Línea de tiempo multimodal de 60 s: quién habla y hacia dónde mira cada persona | `<sesion>.mirada.csv`, `<sesion>.transcription.parquet` |
# | 4 | Dirección de la cabeza en el círculo de la cámara: picos hacia los compañeros (cf. Stiefelhagen & Zhu, 2002) | `<sesion>.faceDirection.parquet` |
# | 5 | Chequeos de integración por sesión: fps real, desfase y duplicados | `f0_resultados.csv` |
#
# **Cómo usarlo en Colab:** sube los archivos a una carpeta de Google Drive (por ejemplo `MyDrive/figuras_paper/`), ajusta `CARPETA` y `SESION` en la celda de configuración, y ejecuta todo (Entorno de ejecución → Ejecutar todas).

# %%
# --- Configuración -----------------------------------------------------------
import os
EN_COLAB = "google.colab" in str(get_ipython()) if "get_ipython" in globals() else False
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    CARPETA = "/content/drive/MyDrive/figuras_paper"     # <- ajusta si usaste otro nombre
else:
    CARPETA = os.environ.get("CARPETA_FIGS", ".")

SESION = "20230821162858-B"     # sesión para las figuras 3 y 4 (G01_B)
T0_TIMELINE = 120.0             # inicio (s, tiempo de video) de la ventana de la figura 3
DUR_TIMELINE = 60.0             # duración de la ventana (s)
SALIDA = os.path.join(CARPETA, "figuras")
os.makedirs(SALIDA, exist_ok=True)

# %%
import re, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# --- Estilo (paleta categórica validada para daltonismo; texto en tinta neutra) ---
P = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}   # participantes 1–4, orden fijo
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURF = "#e6e5e0", "#ffffff"
ABAJO, NADIE = "#4a4946", "#d4d3cd"                                 # clases no-persona: neutras
DIVERG = LinearSegmentedColormap.from_list("div", ["#1f5fae", "#f2f1ec", "#c43c3b"])
COL1, COL2 = 3.5, 7.16                                              # anchos IEEE (pulgadas)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

def guardar(fig, nombre):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(SALIDA, f"{nombre}.{ext}"), dpi=300)
    print("guardada:", nombre)

def buscar(patron):
    r = sorted(glob.glob(os.path.join(CARPETA, "**", patron), recursive=True))
    if not r:
        raise FileNotFoundError(f"No encontré {patron} en {CARPETA}")
    return r[0]

GRUPO = {"20230821162858": "G01", "20230822102617": "G02", "20230823132752": "G04", "20230824171736": "G07",
         "20230825132118": "G08", "20230825172911": "G09", "20230828153245": "G10", "20230830151818": "G11",
         "20230831135353": "G12", "20230926162012": "G14", "20230926173026": "G15", "20231004102507": "G16",
         "20231123140617": "G18", "20230803020857": "G19", "20230803012952": "G20", "12-07": "G21"}
def etiqueta(sesion):
    b = re.split(r"[\\/]", str(sesion).strip())[-1].replace(".transcription.parquet", "")
    return f"{GRUPO.get(b[:-2], b[:-2])}_{b[-1]}"

# %% [markdown]
# ## Figura 1 — Llave hablante ↔ participante

# %%
def leer_matrices(path):
    txt = open(path, encoding="utf-8").read()
    out = {}
    for bloque in txt.split("== ")[1:]:
        cab, *filas = bloque.strip().splitlines()
        ses = etiqueta(cab.split("(")[0].strip())
        lag = float(re.search(r"lag ([+-][\d.]+)s", cab).group(1))
        vals = [list(map(float, f.split()[1:])) for f in filas[1:5]]
        out[ses] = (np.array(vals), lag)
    return out

def figura1(mats, sesiones=("G09_A", "G21_B")):
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 2.05), gridspec_kw={"wspace": 0.28})
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    for ax, s in zip(axs, sesiones):
        M, lag = mats[s]
        im = ax.imshow(M, cmap=DIVERG, norm=norm)
        for i in range(4):
            for j in range(4):
                fuerte = abs(M[i, j]) > 0.3
                ax.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                        fontsize=6.3, color="white" if fuerte else INK, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4), [f"P{k}" for k in range(1, 5)]); ax.set_yticks(range(4), [f"S{k}" for k in range(1, 5)])
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ok = all(M[i].argmax() == i for i in range(4))
        ax.set_title(f"{s.replace('_', ' ')}: {'key valid' if ok else 'not recovered'}", loc="left",
                     color=INK if ok else "#c43c3b", fontsize=7.5)
    axs[0].set_ylabel("transcript speaker")
    fig.supxlabel("visual identity (mouth movement)", fontsize=7.5, color=INK2, y=0.02)
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.04, ticks=[-0.5, 0, 0.5], shrink=0.8)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, length=0); cb.set_label("Pearson r", fontsize=7)
    guardar(fig, "fig1_llave_hablante_identidad")
    return fig

# %% [markdown]
# ## Figura 2 — Los oyentes miran a quien habla

# %%
def figura2(H, ic_target=(1.83, 2.13), gm_target=1.97, ic_clase=(1.95, 2.41), gm_clase=2.17):
    H = H.copy(); H["ses"] = H.sesion.apply(etiqueta)
    orden = H.groupby("ses").razon_target.apply(lambda x: np.exp(np.log(x).mean())).sort_values().index.tolist()
    y = {s: i for i, s in enumerate(orden)}
    fig, axs = plt.subplots(1, 2, figsize=(COL1, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(1)
    for ax, (col, gm, ic, titulo) in zip(axs, [("razon_target", gm_target, ic_target, "Released geometry"),
                                                ("razon_clase", gm_clase, ic_clase, "Recomputed classes")]):
        ax.axvspan(*ic, color=P[1], alpha=0.10, lw=0)
        ax.axvline(gm, color=P[1], lw=1.2)
        ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        for _, r in H.iterrows():
            ax.scatter(r[col], y[r.ses] + rng.uniform(-0.22, 0.22), s=7, color=P[int(r.hablante)],
                       alpha=0.55, lw=0, zorder=2)
        g = H.groupby("ses")[col].apply(lambda x: np.exp(np.log(x).mean()))
        ax.scatter(g.values, [y[s] for s in g.index], s=11, marker="D", color=INK, edgecolor=SURF, lw=0.5, zorder=3)
        ax.set_xscale("log"); ax.set_xlim(0.5, 10)
        ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"]); ax.minorticks_off()
        ax.grid(axis="x", which="major"); ax.set_axisbelow(True)
        ax.set_title(titulo, loc="left")
        ax.text(gm * 1.06, len(orden) - 0.2, f"{gm:.2f}×", color=P[1], fontsize=7, va="bottom")
    axs[0].set_yticks(range(len(orden)), [s.replace("_", " ") for s in orden], fontsize=6.3)
    axs[0].set_ylim(-0.8, len(orden) + 0.6)
    axs[1].tick_params(axis="y", length=0)
    fig.supxlabel("ratio: listeners oriented to speaker k while k speaks vs. while another speaks (log scale)",
                  fontsize=7, color=INK2, y=0.035)
    hand = [plt.Line2D([], [], marker="o", ls="", color=P[k], ms=3.5, label=f"P{k}") for k in range(1, 5)]
    hand += [plt.Line2D([], [], marker="D", ls="", color=INK, ms=4, label="session mean"),
             Patch(color=P[1], alpha=0.25, label="model 95% CI")]
    fig.legend(handles=hand, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.035),
               handletextpad=0.3, columnspacing=0.9)
    guardar(fig, "fig2_validez_oyente_hablante")
    return fig

# %% [markdown]
# ## Figura 3 — Línea de tiempo multimodal (60 s)

# %%
def hablante_unico(t, tt):
    spk = np.zeros(len(tt), int); n = np.zeros(len(tt), int)
    for s, e, k in zip(t.start.values, t.end.values, t.sp.values):
        m = (tt >= s) & (tt <= e); spk[m] = k; n[m] += 1
    return np.where(n == 1, spk, 0)

def figura3(mir, trans, lag, t0=T0_TIMELINE, dur=DUR_TIMELINE, titulo=""):
    d = mir[(mir.t_video_s >= t0) & (mir.t_video_s < t0 + dur)].copy()
    t = trans.dropna(subset=["speaker"]).copy()
    t["sp"] = pd.to_numeric(t.speaker.astype(str).str.replace("SPEAKER_", ""), errors="coerce").astype(int)
    fps = 1 / np.median(np.diff(np.sort(mir.t_video_s.unique())))
    fig, ax = plt.subplots(figsize=(COL2, 2.25))
    filas = {}
    yy = 0
    for k in range(4, 0, -1):            # P1 arriba
        filas[("habla", k)] = yy + 0.62; filas[("mira", k)] = yy; yy += 1.55
    # habla (tiempo de transcripción -> tiempo de video: t_video = t_transcr - lag)
    for _, r in t.iterrows():
        a, b = r.start - lag, r.end - lag
        if b < t0 or a > t0 + dur: continue
        ax.broken_barh([(max(a, t0), min(b, t0 + dur) - max(a, t0))], (filas[("habla", r.sp)], 0.42),
                       color=P[r.sp], lw=0)
    # mira: un segmento por frame, color = objetivo
    w = 1 / fps
    for k in range(1, 5):
        dk = d[d.track_id == k]
        cols = [P[int(c)] if c not in ("abajo", "nadie") else (ABAJO if c == "abajo" else NADIE) for c in dk.clase]
        ax.bar(dk.t_video_s, 0.42, width=w * 1.02, bottom=filas[("mira", k)], color=cols, lw=0, align="edge")
    ax.set_xlim(t0, t0 + dur); ax.set_ylim(-0.25, yy - 0.3)
    ticks, labs = [], []
    for k in range(1, 5):
        ticks += [filas[("habla", k)] + 0.21, filas[("mira", k)] + 0.21]
        labs += [f"P{k} speaks", f"P{k} looks at"]
    ax.set_yticks(ticks, labs, fontsize=6.6); ax.tick_params(axis="y", length=0)
    for k in range(1, 5):
        ax.get_yticklabels()[2 * (k - 1)].set_color(INK); ax.get_yticklabels()[2 * (k - 1) + 1].set_color(INK2)
    for k in range(1, 4):
        ax.axhline(filas[("mira", k)] - 0.47, color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("video time (s)")
    hand = [Patch(color=P[k], label=f"P{k}") for k in range(1, 5)] + \
           [Patch(color=ABAJO, label="down (table)"), Patch(color=NADIE, label="no one")]
    ax.legend(handles=hand, ncol=6, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.1, columnspacing=1.0)
    guardar(fig, "fig3_linea_de_tiempo")
    return fig

# %% [markdown]
# ## Figura 4 — Dirección de la cabeza en el círculo de la cámara

# %%
W_IMG, H_IMG, AMPLIFY = 1920, 1080, 1.2
def theta_from_pixel(px, py):
    x = np.where(py > H_IMG // 2, px + W_IMG, px).astype(float)
    return 2 * np.pi * x / (2 * W_IMG)

def gaze_point_theta(theta, yaw_deg):
    Pp = np.stack([np.cos(theta), np.sin(theta)], 1)
    phi = np.radians(yaw_deg * AMPLIFY)
    alpha = theta + np.pi - phi
    u = np.stack([np.cos(alpha), np.sin(alpha)], 1)
    b = 2 * (Pp * u).sum(1)
    t = np.where(b < 0, -b, 0.0)
    Q = Pp + t[:, None] * u
    return np.arctan2(Q[:, 1], Q[:, 0])

def figura4(fd, titulo=""):
    fd = fd.drop_duplicates(["frame", "track_id"]).copy()
    fd["theta"] = theta_from_pixel((fd.nx + fd.min_x + fd.x1).values, (fd.ny + fd.min_y + fd.y1).values)
    fd["g"] = gaze_point_theta(fd.theta.values, fd.yaw.values)
    pos = fd.groupby("track_id").theta.median()
    fig, axs = plt.subplots(1, 4, figsize=(COL2, 1.75), subplot_kw={"projection": "polar"},
                            gridspec_kw={"wspace": 0.15})
    bins = np.linspace(-np.pi, np.pi, 73)
    for ax, k in zip(axs, range(1, 5)):
        g = fd[fd.track_id == k].g.values
        h, _ = np.histogram(((g + np.pi) % (2 * np.pi)) - np.pi, bins=bins)
        h = h / h.sum()
        ax.bar(bins[:-1], h, width=np.diff(bins), align="edge", color=P[k], alpha=0.85, lw=0)
        rmax = h.max() * 1.25
        for j, th in pos.items():
            if j == k:
                ax.plot([th], [rmax], marker="o", ms=6.5, mfc=SURF, mec=P[k], mew=1.6, clip_on=False, zorder=4)
            else:
                ax.plot([th, th], [0, rmax], color=P[j], lw=0.8, ls=(0, (2, 2)), zorder=1)
                ax.plot([th], [rmax], marker="o", ms=5, color=P[j], mec=SURF, mew=0.8, clip_on=False, zorder=4)
        ax.set_ylim(0, rmax); ax.set_yticks([]); ax.set_xticks([])
        ax.spines["polar"].set_color(GRID); ax.grid(color=GRID, lw=0.5)
        ax.set_title(f"P{k}", color=INK, pad=6)
    hand = [plt.Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, ms=5, label="own seat"),
            plt.Line2D([], [], color=INK2, lw=0.8, ls=(0, (2, 2)), marker="o", ms=4, label="teammate seats"),
            Patch(color=MUTED, alpha=0.8, label="where the head points (share of frames)")]
    fig.legend(handles=hand, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    guardar(fig, "fig4_direccion_cabeza_polar")
    return fig

# %% [markdown]
# ## Figura 5 — Chequeos de integración por sesión

# %%
def figura5(f0):
    f = f0[f0.sesion.astype(str).str.contains(r"[-_][AB]$", regex=True)].copy()
    f["ses"] = f.sesion.apply(etiqueta)
    f["dup_pct"] = f.ObservedUser_duplicados / f.ObservedUser_filas * 100
    f = f.sort_values("ses", ascending=False).reset_index(drop=True)
    ok = (f.llave_OK.astype(str) == "True").values
    y = np.arange(len(f))
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 4.0), sharey=True, gridspec_kw={"wspace": 0.12})
    # (a) fps
    ax = axs[0]
    ax.axvline(15, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(14.985, len(f) - 0.3, "nominal\n15 fps", ha="right", va="top", fontsize=6.3, color=INK2)
    ax.hlines(y, 14.4, f["fps_real(frames/audio)"], color=GRID, lw=1.1)
    ax.scatter(f["fps_real(frames/audio)"], y, s=13, color=P[1], zorder=3)
    ax.set_xlim(14.4, 15.05); ax.set_xticks([14.4, 14.6, 14.8, 15.0])
    ax.set_xlabel("frames / audio duration (fps)")
    ax.set_title("(a) Effective frame rate", loc="left")
    # (b) desfase
    ax = axs[1]
    ax.axvline(0, color=MUTED, lw=0.8)
    med = f.lag_optimo_s[ok].median()
    ax.axvline(med, color=P[1], lw=1, alpha=0.5)
    ax.scatter(f.lag_optimo_s[ok], y[ok], s=13, color=P[1], zorder=3, label="key valid")
    ax.scatter(f.lag_optimo_s[~ok], y[~ok], s=16, marker="x", color="#c43c3b", zorder=3, label="key not valid")
    ax.text(med - 0.15, -1.6, f"median +{med:.2f} s", fontsize=6.3, color=P[1], va="center", ha="right")
    ax.set_xlim(-7.2, 2.2); ax.set_xlabel("transcript minus video (s)")
    ax.set_title("(b) Audio\u2013video offset", loc="left")
    ax.legend(frameon=False, loc="center left", handletextpad=0.2, fontsize=6.5)
    # (c) duplicados
    ax = axs[2]
    ax.hlines(y, 0, f.dup_pct, color=P[2], lw=2.4)
    ax.set_xlabel("duplicated rows (%)")
    ax.set_title("(c) Duplicates in ObservedUser", loc="left")
    axs[0].set_yticks(y, [s.replace("_", " ") for s in f.ses], fontsize=6.3)
    axs[0].set_ylim(-2.3, len(f) - 0.4)
    for a_ in axs:
        a_.tick_params(axis="y", length=0); a_.grid(axis="x"); a_.set_axisbelow(True)
    for a_ in axs[1:]:
        a_.spines["left"].set_visible(False)
    guardar(fig, "fig5_chequeos_integracion")
    return fig

# %% [markdown]
# ## Ejecutar

# %%
if __name__ == "__main__" or EN_COLAB:
    mats = leer_matrices(buscar("f0_matrices.txt"))
    figura1(mats)
    figura2(pd.read_csv(buscar("validez_por_hablante.csv")))
    f0 = pd.read_csv(buscar("f0_resultados.csv"))
    lag = float(f0[f0.sesion.astype(str).str.endswith(SESION)].lag_optimo_s.iloc[0])
    figura3(pd.read_csv(buscar(f"{SESION}.mirada.csv")), pd.read_parquet(buscar(f"{SESION}.transcription.parquet")),
            lag, titulo=f"{etiqueta(SESION).replace('_', ' ')}: speech and head-direction targets")
    figura4(pd.read_parquet(buscar(f"{SESION}.faceDirection.parquet")),
            titulo=f"{etiqueta(SESION).replace('_', ' ')}: where each participant's head points on the camera circle")
    figura5(f0)
    plt.show()
