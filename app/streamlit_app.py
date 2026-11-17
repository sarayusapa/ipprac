"""Lung nodule detection (image processing pipeline): interactive demo.

    streamlit run app/streamlit_app.py

Needs the scan index built once by:  python app/prepare_demo.py
Results are never stored: each scan is processed when its button is pressed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_core as dc  # noqa: E402
from lungnod.io import load_scan  # noqa: E402

st.set_page_config(page_title="Lung Nodule Detection", layout="wide")

FONT = '"Paper Mono", ui-monospace, SFMono-Regular, Menlo, monospace'
TITLE_FONT = '"Times New Roman", Times, serif'
st.markdown(
    f"""<style>
@font-face {{ font-family: "Paper Mono"; font-weight: 400; src: url("app/static/PaperMono-Regular.woff2") format("woff2"); }}
@font-face {{ font-family: "Paper Mono"; font-weight: 700; src: url("app/static/PaperMono-Bold.woff2") format("woff2"); }}
html, body, p, li, span, label, div, td, th, input, button, textarea, select,
h1, h2, h3, h4, h5, h6, .stMarkdown, .stCaption {{ font-family: {FONT} !important; }}
h1, h1 span {{ font-family: {TITLE_FONT} !important; font-weight: 600 !important;
  font-size: 2.1rem !important; letter-spacing: 0 !important; }}
h2, h3, h4 {{ font-weight: 700 !important; letter-spacing: 0 !important; }}
h3 {{ font-size: 1.45rem !important; }}
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{ font-size: 1.2rem !important; line-height: 1.6 !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ font-size: 1.05rem !important; }}
[data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{ font-size: 1.1rem !important; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ font-size: 1.1rem !important; }}
[data-testid="stSelectbox"] div, [data-testid="stSpinner"] p {{ font-size: 1.05rem !important; }}
button[data-baseweb="tab"] p, button[data-baseweb="tab"] div {{
  font-family: {FONT} !important; font-size: 1.35rem !important; font-weight: 700 !important; }}
button[data-baseweb="tab"] {{ padding: 0.4rem 1.2rem !important; }}
[data-testid="stBaseButton-primary"] {{ background-color: #2e7d32 !important;
  border-color: #2e7d32 !important; color: #ffffff !important; }}
[data-testid="stBaseButton-primary"]:hover {{ background-color: #1b5e20 !important;
  border-color: #1b5e20 !important; color: #ffffff !important; }}
[data-testid="stBaseButton-primary"]:active {{ background-color: #14481a !important; }}
[data-testid="stBaseButton-primary"] p {{ color: #ffffff !important; font-weight: 700 !important; }}
[data-testid="stBaseButton-secondary"] p {{ font-size: 1rem !important; white-space: nowrap !important; }}
.side-info, .side-info p, .side-info li, .side-info b {{ font-size: 0.82rem !important; line-height: 1.45 !important; }}
.side-info ul {{ margin: 0.15rem 0 0.6rem 1.0rem !important; padding: 0 !important; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: "Material Symbols Rounded" !important; }}
</style>""",
    unsafe_allow_html=True,
)

GREEN, RED, GREY, GT, NEUTRAL = "#15803d", "#b91c1c", "#6b7280", "#ca8a04", "#1f4e79"
VCOL = {"TP": GREEN, "FP": RED, "ignored": GREY}
VTXT = {"TP": "nodule", "FP": "false alarm", "ignored": "excluded finding"}
N_SUCCESS, N_FAILURE, SEED = 40, 10, 7
WINDOW = (-1000, 400)


# ------------------------------------------------------------------ helpers
@st.cache_data
def scan_list():
    """40 scans where every ground-truth nodule outranks every false alarm and 10 where the
    method ranks the nodules poorly or misses them, shuffled together (fixed seed)."""
    idx = dc.scan_index()
    ok = idx[idx.fps_above_weakest == 0]
    bad = idx[(idx.n_nodules > 0) & ((idx.fps_above_best >= 10) | (idx.fps_above_best == -1))]
    best = ok.sort_values("showcase_rank").iloc[:1]
    ok = ok.drop(best.index).sample(N_SUCCESS - 1, random_state=SEED)
    bad = bad.sample(N_FAILURE, random_state=SEED)
    mix = pd.concat([ok, bad]).sample(frac=1, random_state=SEED)
    return pd.concat([best, mix]).reset_index(drop=True)


@st.cache_resource(max_entries=1)
def get_scan(path: str):
    return load_scan(path)


def to_gray(a, lo=WINDOW[0], hi=WINDOW[1]):
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def base_fig(img, height=620, colorscale=None, zrange=None):
    if colorscale is None:
        fig = go.Figure(go.Image(z=np.stack([img] * 3, -1), hoverinfo="skip"))
    else:
        fig = go.Figure(go.Heatmap(z=img, colorscale=colorscale, zmin=zrange[0], zmax=zrange[1],
                                   showscale=True, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), dragmode="pan",
                      plot_bgcolor="black", font=dict(family=FONT))
    fig.update_xaxes(visible=False, scaleanchor="y")
    fig.update_yaxes(visible=False)
    return fig


def circle(fig, x, y, r_px, color, width=2, dash=None):
    fig.add_shape(type="circle", x0=x - r_px, x1=x + r_px, y0=y - r_px, y1=y + r_px,
                  line=dict(color=color, width=width, dash=dash))


def mark_on_slice(blobs, z):
    """The track's blob on slice z (centre, radius in mm), or None."""
    b = min(blobs, key=lambda b: abs(b.z - z))
    return (b.x, b.y, b.radius_mm) if b.z == z else None


# ------------------------------------------------------------------ sidebar
idx = scan_list()
st.sidebar.markdown("### Lung Nodule Detection")
uids = idx.seriesuid.tolist()
uid = st.sidebar.selectbox("Scan", uids)
info = idx.set_index("seriesuid").loc[uid]
path = info.path
scan_info = st.sidebar.container()

# a new scan always starts unprocessed: drop any result from the previous one
if st.session_state.get("uid") != uid:
    st.session_state.uid = uid
    st.session_state.pop("result", None)
    st.session_state.pop("steps", None)
    st.session_state.mark = 0
    st.session_state.pop("z", None)
res = st.session_state.get("result")

scan = get_scan(path)
dz, sy, sx = (float(v) for v in scan.spacing)
ann, exc = dc.annotations(uid)
ann_vox = scan.world_to_voxel(ann[["coordX", "coordY", "coordZ"]].to_numpy()) if len(ann) else np.zeros((0, 3))
if "z" not in st.session_state:
    st.session_state.z = int(round(ann_vox[0, 0])) if len(ann) else scan.hu.shape[0] // 2

if res is None:
    if st.sidebar.button("Run pipeline on scan", type="primary", width="stretch"):
        bar = st.sidebar.progress(0.0, text="Starting")
        out = dc.run_pipeline(scan, lambda f, m: bar.progress(min(f, 1.0), text=m))
        st.session_state.result = out
        t = out["table"]
        st.session_state.z = int(t[t.probability > 0].vox_z.iloc[0]) if (t.probability > 0).any() else st.session_state.z
        st.rerun()
    top_k = 0
else:
    k_default = int(min(max(len(ann) + 3, 5), 20))
    top_k = st.sidebar.slider("Number of marks shown", 1, 30, k_default)
    show_lung = st.sidebar.toggle("Show lung mask outline", False)
    st.sidebar.markdown(
        "<div class='side-info'><b>Run time on this scan</b><ul>"
        + "".join(f"<li>{k}: {v:.1f} s</li>" for k, v in res["timings"].items())
        + "</ul></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ results table
if res is not None:
    tracks = res["table"]
    tracks = tracks[tracks.probability > 0].reset_index(drop=True)
    tracks = pd.concat([tracks, dc.label_marks(tracks, ann, exc)], axis=1)
    tracks["rank"] = np.arange(1, len(tracks) + 1)
    top = tracks.head(top_k)
    nod_rank = []
    for j in range(len(ann)):
        r = tracks["rank"][tracks.nodule_idx == j]
        nod_rank.append(int(r.min()) if len(r) else None)
    found = sum(1 for r in nod_rank if r and r <= top_k)
    n_fp = int((top.verdict == "FP").sum())

items = [f"Slices: {scan.hu.shape[0]}", f"Slice spacing: {dz:.2f} mm", f"Annotated nodules: {len(ann)}"]
if res is not None:
    items.append(f"Top {top_k} marks: {found} nodule{'s' if found != 1 else ''}, "
                 f"{n_fp} false alarm{'s' if n_fp != 1 else ''}")
scan_info.markdown(
    "<div class='side-info'><b>Scan information</b><ul>"
    + "".join(f"<li>{t}</li>" for t in items) + "</ul></div>", unsafe_allow_html=True)

st.title("Lung Nodule Detection")


def ground_truth_list():
    if not len(ann):
        return
    st.markdown("### Dataset ground truth")
    for j, r in ann.iterrows():
        txt = f"Nodule {j + 1}: {r.diameter_mm:.1f} mm, slice {int(round(ann_vox[j, 0]))}"
        if res is not None:
            txt += f", rank {nod_rank[j]}" if nod_rank[j] else ", not detected"
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown(txt)
        if c2.button("Go to slice", key=f"go{j}"):
            st.session_state.z = int(round(ann_vox[j, 0]))
            st.rerun()


def slice_viewer(with_marks: bool):
    z = st.slider("Slice", 0, scan.hu.shape[0] - 1, key="z")
    fig = base_fig(to_gray(scan.hu[z]))
    if with_marks and show_lung:
        lung = res["lung"][z]
        ys, xs = np.nonzero(lung ^ np.roll(lung, 1, 0) | lung ^ np.roll(lung, 1, 1))
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", marker=dict(size=1.5, color="#38bdf8"),
                                   hoverinfo="skip", showlegend=False))
    for j, r in ann.iterrows():
        az, ay, ax_ = ann_vox[j]
        h = r.diameter_mm / 2
        if abs(az - z) * dz <= h:
            rr = np.sqrt(h**2 - ((az - z) * dz) ** 2) / sx + 4
            circle(fig, ax_, ay, rr, "#facc15", 2, "dot")
    if with_marks:
        for i, row in top.iterrows():
            m = mark_on_slice(res["tracks"][row.track_id], z)
            if m is None:
                continue
            x, y, r_mm = m
            col = {"TP": "#22c55e", "FP": "#ef4444", "ignored": "#d1d5db"}[row.verdict]
            circle(fig, x, y, r_mm / sx + 1, col, 3 if i == st.session_state.mark else 2)
            fig.add_annotation(x=x + r_mm / sx + 4, y=y - r_mm / sx,
                               text=f"{row['rank']}: {row.probability:.2f}", showarrow=False,
                               font=dict(color=col, size=16, family=FONT), xanchor="left",
                               bgcolor="rgba(0,0,0,0.6)")
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    if with_marks:
        st.caption("Circle label = rank: score. Dotted yellow = ground truth.")
    else:
        st.caption("Dotted yellow = ground truth.")


# ------------------------------------------------------------------ before processing
if res is None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        slice_viewer(False)
    with right:
        ground_truth_list()
    st.stop()

# ------------------------------------------------------------------ after processing
tab_det, tab_score, tab_pipe, tab_res = st.tabs(["Detection", "Scoring system", "Pipeline", "Result"])

with tab_det:
    left, right = st.columns([3, 2], gap="large")
    with right:
        view = pd.DataFrame({
            "Rank": top["rank"], "Score": top.probability.round(3),
            "Verdict": top.verdict.map(VTXT), "Diameter (mm)": top.max_diameter_mm.round(1),
            "Slices": top.n_blobs, "Centre slice": top.vox_z.astype(int)})
        styled = view.style.apply(
            lambda col: [f"color: {VCOL[v]}; font-weight: 600" for v in top.verdict], subset=["Verdict"])
        ev = st.dataframe(styled, hide_index=True, on_select="rerun", selection_mode="single-row",
                          width="stretch", key=f"tbl_{uid}_{top_k}")
        sel = ev.selection.rows
        if sel and st.session_state.get("last_sel") != (uid, sel[0]):
            st.session_state.last_sel = (uid, sel[0])
            st.session_state.mark = sel[0]
            st.session_state.z = int(top.vox_z.iloc[sel[0]])
            st.rerun()
        st.markdown("### Dataset notation")
        st.markdown(
            f"<span style='color:{GREEN}'>**nodule**</span>: inside a ground-truth nodule.  \n"
            f"<span style='color:{RED}'>**false alarm**</span>: not on any annotated finding.  \n"
            f"<span style='color:{GREY}'>**excluded finding**</span>: on something the radiologists "
            "marked but did not accept as a nodule (smaller than 3 mm, marked by fewer than 3 of "
            "the 4 radiologists, or not a nodule, such as a scar). LUNA16 does not count these as "
            "hits or as false alarms.", unsafe_allow_html=True)
        ground_truth_list()
    with left:
        slice_viewer(True)

# ------------------------------------------------------------------ tab 2: scoring
with tab_score:
    st.markdown("Six shape rules, each scoring 0 to 1 (1 = nodule-like). "
                "**Final score = product of the six.** Candidates are ranked by final score.")
    if not len(top):
        st.info("No marks.")
    else:
        mi = st.selectbox("Mark", range(len(top)), index=min(st.session_state.mark, len(top) - 1),
                          format_func=lambda i: f"Rank {top['rank'].iloc[i]}, score "
                                                f"{top.probability.iloc[i]:.3f} ({VTXT[top.verdict.iloc[i]]})")
        row = top.iloc[mi]
        st.markdown(f"### Rank {row['rank']}: score {row.probability:.3f} "
                    f"(<span style='color:{VCOL[row.verdict]}'>{VTXT[row.verdict]}</span>)",
                    unsafe_allow_html=True)

        zs = list(range(int(row.z_first) - 1, int(row.z_last) + 2))
        if len(zs) > 11:
            zs = list(np.linspace(zs[0], zs[-1], 11).round().astype(int))
        half = int(25 / sx)
        cy, cx = int(row.vox_y), int(row.vox_x)
        cols = st.columns(len(zs))
        for cz, zz in zip(cols, zs):
            if 0 <= zz < scan.hu.shape[0]:
                crop = scan.hu[zz, max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]
                tracked = row.z_first <= zz <= row.z_last
                cz.image(to_gray(crop), caption=f"slice {zz}" + ("" if tracked else " (outside)"),
                         width="stretch")

        a, b = st.columns([1, 1], gap="large")
        with a:
            st.markdown("**The six rule scores for this mark**")
            s = [row[f"s_{r}"] for r in dc.RULES]
            bar = go.Figure(go.Bar(x=s, y=[dc.RULE_TEXT[r][0] for r in dc.RULES], orientation="h",
                                   marker_color=NEUTRAL, text=[f"{v:.2f}" for v in s],
                                   textposition="outside"))
            bar.update_layout(height=280, xaxis_range=[0, 1.15], margin=dict(l=0, r=0, t=10, b=0),
                              font=dict(family=FONT, size=17), yaxis=dict(autorange="reversed"))
            st.plotly_chart(bar, width="stretch")
            prod = " × ".join(f"{v:.2f}" for v in s)
            st.markdown(f"Score = {prod} = **{row.probability:.3f}**")
        with b:
            feats = {
                "aspect": f"{row.lifetime_mm:.1f} mm tall, {row.max_diameter_mm:.1f} mm wide: aspect {row.aspect:.2f}",
                "drift": f"centre moves {row.drift_ratio:.2f} radii from first to last slice",
                "sphere": (f"sphere fits better than tube by {row.sphere_gain:.3f}" if row.sphere_fitted
                           else "fewer than 4 slices, so no fit; neutral score 0.5"),
                "size": f"diameter {row.max_diameter_mm:.1f} mm",
                "min_slices": f"seen on {int(row.n_blobs)} slices",
                "dot": f"average roundness q = {row.hess_ratio_mean:.2f}",
            }
            for r in dc.RULES:
                st.markdown(f"**{dc.RULE_TEXT[r][0]}: {row[f's_{r}']:.2f}**. Measured: {feats[r]}.  \n"
                            f"<span style='color:#555;font-size:0.92em'>{dc.RULE_TEXT[r][1]}</span>",
                            unsafe_allow_html=True)

        bl = res["tracks"][row.track_id]
        st.markdown("**Radius across slices and roundness per slice**")
        p = dc.profile(bl, scan.spacing)
        c1, c2 = st.columns(2, gap="large")
        f = go.Figure(go.Scatter(x=p["z"], y=p["r"], mode="markers+lines", name="measured radius",
                                 marker=dict(size=9, color="black"), line=dict(color="black")))
        if "fit" in p:
            f.add_trace(go.Scatter(x=p["zz"], y=p["sphere"], name="best sphere", line=dict(color=NEUTRAL)))
            f.add_trace(go.Scatter(x=p["zz"], y=p["tube"], name="best tube",
                                   line=dict(color="#888", dash="dash")))
        f.update_layout(height=300, xaxis_title="z (mm)", yaxis_title="radius (mm)",
                        margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"),
                        font=dict(family=FONT, size=17))
        c1.plotly_chart(f, width="stretch")
        qf = go.Figure(go.Bar(x=p["z"], y=p["q"], marker_color=NEUTRAL))
        qf.update_layout(height=300, yaxis_range=[0, 1], xaxis_title="z (mm)",
                         yaxis_title="roundness q (1 = disc, 0 = line)",
                         margin=dict(l=0, r=0, t=10, b=0), font=dict(family=FONT, size=17))
        c2.plotly_chart(qf, width="stretch")

# ------------------------------------------------------------------ tab 3: pipeline
with tab_pipe:
    z = st.session_state.z
    st.markdown(f"**Slice {z}**")
    steps = st.session_state.setdefault("steps", {})
    if z not in steps:
        with st.spinner("Computing ..."):
            steps = {z: dc.slice_steps(scan.hu[z], res["lung"][z], scan.spacing, z)}
            st.session_state.steps = steps
    stp = steps[z]
    lung = res["lung"][z]
    row1, row2 = st.columns(3), st.columns(3)
    with row1[0]:
        st.markdown("**1. CT slice** (shown from −1000 to 400 HU)")
        st.plotly_chart(base_fig(to_gray(scan.hu[z]), 380), width="stretch", key="s1")
    with row1[1]:
        st.markdown("**2. Preprocessed** (HU clipped, 3×3 median filter)")
        st.plotly_chart(base_fig(to_gray(stp["filtered"]), 380), width="stretch", key="s2")
    with row1[2]:
        st.markdown("**3. Lung mask** (air below −400 HU, cleaned, 10 mm closing)")
        g = to_gray(scan.hu[z]).astype(float)
        rgb = np.stack([g * 0.6, g * 0.6 + lung * 90, g * 0.6 + lung * 120], -1).clip(0, 255).astype(np.uint8)
        f = go.Figure(go.Image(z=rgb, hoverinfo="skip"))
        f.update_layout(height=380, margin=dict(l=0, r=0, t=0, b=0))
        f.update_xaxes(visible=False, scaleanchor="y")
        f.update_yaxes(visible=False)
        st.plotly_chart(f, width="stretch", key="s3")
    with row2[0]:
        st.markdown("**4. Blob filter response** (LoG, best of 10 sizes from 1.5 to 15 mm)")
        st.plotly_chart(base_fig(stp["log"], 380, "Gray", (0, 0.5)), width="stretch", key="s4")
    with row2[1]:
        st.markdown("**5. Roundness q** (Hessian ratio, 1 = disc, 0 = line)")
        st.plotly_chart(base_fig(stp["q"], 380, "Viridis", (0, 1)), width="stretch", key="s5")
    with row2[2]:
        st.markdown(f"**6. Blobs found on this slice** ({len(stp['blobs'])})")
        f = base_fig(to_gray(scan.hu[z]), 380)
        for bb in stp["blobs"]:
            circle(f, bb.x, bb.y, bb.radius_mm / sx + 1, "#38bdf8", 1.5)
        st.plotly_chart(f, width="stretch", key="s6")
    st.markdown(
        "**7. Tracking.** Blobs on neighbouring slices are linked into 3D objects (Hungarian matching; "
        "a blob may link to one within max(3 mm, half its radius); one missing slice allowed).  \n"
        "**8. Shape features and 9. scoring.** See the Scoring system tab.")

# ------------------------------------------------------------------ tab 4: result
with tab_res:
    st.markdown("### Our method on all 888 LUNA16 scans (1,186 nodules)")
    m = dc.ROOT / "outputs" / "p0_all_v6" / "metrics.json"
    if m.exists():
        mj = json.loads(m.read_text())
        sub = mj.get("submission", {})
        sens = sub.get("sens_at", {})
        rows = [("CPM (average sensitivity at 1/8 to 8 false alarms per scan)", f"{sub.get('cpm', float('nan')):.3f}")]
        rows += [(f"Sensitivity at {k} false alarms per scan", f"{v:.3f}") for k, v in sens.items()]
        rows.append(("Nodules with at least one candidate", f"{mj.get('candidate_recall_all_tracks', float('nan')):.3f}"))
        st.table(pd.DataFrame(rows, columns=["Measure", "Value"]))
    st.markdown("""
**Comparison on the same 89 LUNA16 scans**

| Method | Learns from labels | CPM |
|---|---|---|
| Our method (image processing pipeline) | no | 0.297 |
| CT-FM 3D detector, 1 epoch | yes | 0.31 to 0.35 |
| MONAI RetinaNet, fully trained | yes | 0.819 |

""")
    froc = dc.ROOT / "docs" / "figures" / "froc_all.png"
    if froc.exists():
        st.image(str(froc), caption="FROC curve on all 888 scans", width=650)
