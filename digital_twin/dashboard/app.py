"""
Smart Factory Digital Twin — Live Dashboard
============================================
Run with:   streamlit run digital_twin/dashboard/app.py

Architecture:
  - @st.cache_resource initialises the generator, predictor and detector ONCE
  - Every rerun (triggered by time.sleep + st.rerun) generates a fresh snapshot
  - Session state holds per-machine rolling history (60 ticks ≈ 2 min @ 2 s interval)
"""

import sys
import time
import os
from pathlib import Path
from typing import Dict

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from ingestion.mock_plc      import MockPLCGenerator, MACHINE_PROFILES
from ml.failure_predictor    import FailurePredictor
from ml.anomaly_detector     import AnomalyDetector
from optimization.bottleneck import BottleneckAnalyzer
from optimization.scheduler  import ProductionScheduler

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Factory Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS — dark industrial theme
# --------------------------------------------------------------------------
st.markdown("""
<style>
  body, .main { background-color: #0d1117; color: #e0e0e0; }

  div[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] label { color: #8b949e; font-size: 0.78rem; }

  .machine-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    margin-bottom: 6px;
  }
  .running  { border-left: 4px solid #2ecc71; }
  .fault    { border-left: 4px solid #e74c3c; }
  .idle     { border-left: 4px solid #3498db; }
  .degrading{ border-left: 4px solid #f39c12; }

  .alert-critical { background-color: #3d1a1a; border: 1px solid #e74c3c;
                    border-radius: 6px; padding: 6px 10px; margin: 3px 0; }
  .alert-warning  { background-color: #3d2a00; border: 1px solid #f39c12;
                    border-radius: 6px; padding: 6px 10px; margin: 3px 0; }
  .alert-info     { background-color: #1a2a3d; border: 1px solid #3498db;
                    border-radius: 6px; padding: 6px 10px; margin: 3px 0; }

  h1 { color: #f0f6fc !important; }
  h2, h3 { color: #c9d1d9 !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Singleton resources (initialised once per Streamlit session)
# --------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    """Initialise PLC generator, ML models and optimisation engine once."""
    plc       = MockPLCGenerator()
    predictor = FailurePredictor()
    detector  = AnomalyDetector()
    analyzer  = BottleneckAnalyzer(list(MACHINE_PROFILES.keys()))
    scheduler = ProductionScheduler()
    return plc, predictor, detector, analyzer, scheduler

plc, predictor, detector, analyzer, scheduler = load_resources()

# --------------------------------------------------------------------------
# Session-state initialisation
# --------------------------------------------------------------------------

MACHINES   = list(MACHINE_PROFILES.keys())
HISTORY_LEN = 60   # readings kept per machine

if "history" not in st.session_state:
    st.session_state.history  = {m: [] for m in MACHINES}
    st.session_state.alerts   = []
    st.session_state.tick     = 0
    st.session_state.start_ts = time.time()

# --------------------------------------------------------------------------
# Generate a new data snapshot
# --------------------------------------------------------------------------

readings = plc.generate_all()
st.session_state.tick += 1

# Append to rolling history
for mid, reading in readings.items():
    buf = st.session_state.history[mid]
    buf.append(reading.to_dict())
    if len(buf) > HISTORY_LEN:
        buf.pop(0)

# Run ML inference on fresh readings
raw_dicts       = {mid: r.to_dict() for mid, r in readings.items()}
failure_results = predictor.predict_all(raw_dicts)
anomaly_results = detector.detect_all(raw_dicts)

# Build stats list for bottleneck analysis
stats_list = []
for mid, reading in readings.items():
    stats_list.append({
        "machine_id":    mid,
        "cycle_time":    reading.cycle_time,
        "uptime_pct":    70 + 20 * (1 - plc.get_degradation_states().get(mid, 0)),
        "queue_length":  0,
        "throughput_pph": 3600 / max(1, reading.cycle_time) if reading.cycle_time > 0 else 0,
    })
bottleneck = analyzer.find_bottleneck(stats_list)

# Aggregate alerts
for mid, fr in failure_results.items():
    if fr.risk_level in ("HIGH", "CRITICAL"):
        msg = f"⚠ {mid}: Failure risk {fr.risk_level} ({fr.failure_pct:.1f}%)"
        recent_msgs = [a["msg"] for a in st.session_state.alerts[-15:]]
        if msg not in recent_msgs:
            st.session_state.alerts.append({
                "time":     time.strftime("%H:%M:%S"),
                "level":    fr.risk_level,
                "msg":      msg,
            })

for mid, ar in anomaly_results.items():
    if ar.is_anomaly:
        msg = f"🔍 {mid}: Anomaly detected (score={ar.anomaly_score:.2f})"
        recent_msgs = [a["msg"] for a in st.session_state.alerts[-15:]]
        if msg not in recent_msgs:
            st.session_state.alerts.append({
                "time":  time.strftime("%H:%M:%S"),
                "level": "WARNING",
                "msg":   msg,
            })

# Keep alert buffer bounded
if len(st.session_state.alerts) > 100:
    st.session_state.alerts = st.session_state.alerts[-100:]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def machine_css_class(reading, failure_prob: float) -> str:
    if reading.fault_code > 0:
        return "fault"
    if failure_prob > 0.6:
        return "degrading"
    if reading.speed == 0:
        return "idle"
    return "running"

def status_emoji(reading, failure_prob: float) -> str:
    if reading.fault_code > 0:
        return "🔴"
    if failure_prob > 0.6:
        return "🟠"
    if reading.speed == 0:
        return "🔵"
    return "🟢"

def plotly_layout(title: str, height: int = 260) -> Dict:  # noqa: F821
    return dict(
        title=dict(text=title, font=dict(size=13, color="#c9d1d9")),
        height=height,
        margin=dict(t=40, b=30, l=45, r=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", size=11),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    )

# --------------------------------------------------------------------------
# ─── HEADER ───────────────────────────────────────────────────────────────
# --------------------------------------------------------------------------

col_hdr, col_oee = st.columns([4, 1])

with col_hdr:
    elapsed  = int(time.time() - st.session_state.start_ts)
    h, rem   = divmod(elapsed, 3600)
    m, s     = divmod(rem, 60)
    st.title("🏭 Smart Factory Digital Twin")
    st.caption(
        f"**Live** | Tick #{st.session_state.tick} | "
        f"Elapsed {h:02d}:{m:02d}:{s:02d} | "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

with col_oee:
    running_count = sum(1 for r in readings.values() if r.fault_code == 0 and r.speed > 0)
    # Synthetic OEE drifts realistically
    deg_avg  = np.mean(list(plc.get_degradation_states().values()))
    oee_val  = max(55.0, 91.0 - deg_avg * 30 + np.random.normal(0, 0.3))

    fig_oee = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(oee_val, 1),
        number=dict(suffix="%", font=dict(size=28, color="#f0f6fc")),
        title=dict(text="Factory OEE", font=dict(size=13, color="#8b949e")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#8b949e"),
            bar=dict(color="#2ecc71" if oee_val > 80 else "#f39c12" if oee_val > 65 else "#e74c3c"),
            bgcolor="#161b22",
            steps=[
                dict(range=[0,  65], color="#3d1a1a"),
                dict(range=[65, 80], color="#3d2a00"),
                dict(range=[80, 100], color="#132a17"),
            ],
            threshold=dict(line=dict(color="#f0f6fc", width=2), value=85),
        ),
    ))
    fig_oee.update_layout(height=200, margin=dict(t=30, b=0, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f0f6fc"))
    st.plotly_chart(fig_oee, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# ─── KPI ROW ──────────────────────────────────────────────────────────────
# --------------------------------------------------------------------------

faulted_count = sum(1 for r in readings.values() if r.fault_code > 0)
total_parts   = sum(r.parts_produced for r in readings.values())
avg_temp      = round(np.mean([r.temperature for r in readings.values()]), 1)
anomaly_count = sum(1 for ar in anomaly_results.values() if ar.is_anomaly)
high_risk     = sum(1 for fr in failure_results.values() if fr.risk_level in ("HIGH", "CRITICAL"))
throughput    = round(sum(3600 / max(1, r.cycle_time) for r in readings.values() if r.cycle_time > 0), 1)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🟢 Running",       running_count,     f"/ {len(readings)} machines")
k2.metric("🔴 Faulted",       faulted_count,     delta=None)
k3.metric("📦 Parts Produced", f"{total_parts:,}", "cumulative")
k4.metric("⚠ High Risk",      high_risk,         "machines")
k5.metric("🔍 Anomalies",     anomaly_count,     "active")
k6.metric("⚡ Throughput",    f"{throughput}",   "pph total")

st.divider()

# --------------------------------------------------------------------------
# ─── MACHINE STATUS GRID ──────────────────────────────────────────────────
# --------------------------------------------------------------------------

st.subheader("Machine Status")
cols = st.columns(len(MACHINES))

for i, mid in enumerate(MACHINES):
    reading = readings[mid]
    fp      = failure_results[mid].failure_probability
    ar      = anomaly_results[mid]
    css     = machine_css_class(reading, fp)
    emoji   = status_emoji(reading, fp)

    with cols[i]:
        profile_label = MACHINE_PROFILES[mid].machine_type
        status_text   = "FAULT" if reading.fault_code > 0 else ("IDLE" if reading.speed == 0 else "RUNNING")

        st.markdown(f"""
<div class="machine-card {css}">
  <b>{emoji} {mid}</b><br>
  <small style="color:#8b949e">{profile_label}</small><br><br>
  <b>{status_text}</b><br>
  🌡 {reading.temperature:.1f}°C<br>
  📳 {reading.vibration:.2f} mm/s<br>
  ⚡ {reading.current:.1f} A<br>
  {'🔴 FAULT #' + str(reading.fault_code) if reading.fault_code else ''}
  {'⚠ ANOMALY' if ar.is_anomaly else ''}
  {'⚠ BOTTLENECK' if bottleneck and bottleneck.get('bottleneck_id') == mid else ''}
</div>
""", unsafe_allow_html=True)

        risk_color = "#e74c3c" if fp > 0.7 else "#f39c12" if fp > 0.4 else "#2ecc71"
        st.markdown(
            f"<div style='height:5px;background:{risk_color};border-radius:3px;"
            f"width:{int(fp*100)}%'></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Fail risk {fp*100:.1f}%")

st.divider()

# --------------------------------------------------------------------------
# ─── DETAIL: SENSOR CHARTS ────────────────────────────────────────────────
# --------------------------------------------------------------------------

col_sel, col_info = st.columns([1, 3])
with col_sel:
    selected = st.selectbox("Machine detail view", MACHINES)

history_df = pd.DataFrame(st.session_state.history.get(selected, []))

if not history_df.empty:
    history_df["ts"] = pd.to_datetime(history_df["timestamp"], unit="s")

    c1, c2 = st.columns(2)

    with c1:
        # Temperature
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=history_df["ts"], y=history_df["temperature"],
            mode="lines", line=dict(color="#e74c3c", width=2),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.08)", name="Temp",
        ))
        fig_t.add_hline(y=75, line_dash="dot", line_color="#f39c12",
                         annotation_text="Warn 75°C", annotation_font_color="#f39c12")
        fig_t.update_layout(**plotly_layout(f"🌡 Temperature — {selected}"))
        st.plotly_chart(fig_t, use_container_width=True)

        # Current
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=history_df["ts"], y=history_df["current"],
            mode="lines", line=dict(color="#f39c12", width=2), name="Current",
        ))
        fig_c.update_layout(**plotly_layout(f"⚡ Motor Current — {selected}"),
                             yaxis_title="Amperes")
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        # Vibration
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=history_df["ts"], y=history_df["vibration"],
            mode="lines", line=dict(color="#3498db", width=2),
            fill="tozeroy", fillcolor="rgba(52,152,219,0.08)", name="Vibration",
        ))
        fig_v.add_hline(y=8.0, line_dash="dot", line_color="#f39c12",
                         annotation_text="Warn 8.0 mm/s", annotation_font_color="#f39c12")
        fig_v.update_layout(**plotly_layout(f"📳 Vibration — {selected}"),
                             yaxis_title="mm/s RMS")
        st.plotly_chart(fig_v, use_container_width=True)

        # Cycle time bar chart
        tail = history_df.tail(25)
        bar_colors = [
            "#e74c3c" if ct > MACHINE_PROFILES[selected].nominal_cycle_time * 1.3
            else "#f39c12" if ct > MACHINE_PROFILES[selected].nominal_cycle_time * 1.1
            else "#2ecc71"
            for ct in tail["cycle_time"]
        ]
        fig_cy = go.Figure()
        fig_cy.add_trace(go.Bar(
            x=tail["ts"], y=tail["cycle_time"],
            marker_color=bar_colors, name="Cycle Time",
        ))
        fig_cy.add_hline(y=MACHINE_PROFILES[selected].nominal_cycle_time,
                          line_dash="dash", line_color="#8b949e",
                          annotation_text="Nominal", annotation_font_color="#8b949e")
        fig_cy.update_layout(**plotly_layout(f"⏱ Cycle Time — {selected}"),
                              yaxis_title="Seconds")
        st.plotly_chart(fig_cy, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# ─── PREDICTIVE MAINTENANCE — FAILURE GAUGES ──────────────────────────────
# --------------------------------------------------------------------------

st.subheader("🔮 Predictive Maintenance — Failure Risk")

gauge_cols = st.columns(len(MACHINES))
for i, mid in enumerate(MACHINES):
    fr     = failure_results[mid]
    prob   = fr.failure_probability
    color  = "#e74c3c" if prob > 0.7 else "#f39c12" if prob > 0.4 else "#2ecc71"

    with gauge_cols[i]:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number=dict(suffix="%", font=dict(size=20, color="#f0f6fc")),
            title=dict(text=mid, font=dict(size=11, color="#8b949e")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#8b949e", tickfont=dict(size=8)),
                bar=dict(color=color),
                bgcolor="#161b22",
                steps=[
                    dict(range=[0,  40], color="rgba(46,204,113,0.15)"),
                    dict(range=[40, 70], color="rgba(243,156,18,0.15)"),
                    dict(range=[70, 100], color="rgba(231,76,60,0.15)"),
                ],
                threshold=dict(line=dict(color="#f0f6fc", width=2), value=70),
            ),
        ))
        fig_g.update_layout(
            height=190,
            margin=dict(t=45, b=5, l=8, r=8),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f6fc"),
        )
        st.plotly_chart(fig_g, use_container_width=True)
        risk_lbl = fr.risk_level
        label_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "red"}.get(risk_lbl, "gray")
        st.markdown(f"<center><small style='color:{label_color}'><b>{risk_lbl}</b></small></center>",
                    unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------
# ─── ANOMALY RADAR CHART ──────────────────────────────────────────────────
# --------------------------------------------------------------------------

col_radar, col_bn = st.columns([1, 1])

with col_radar:
    st.subheader("🕸 Anomaly Score by Machine")
    radar_scores = [anomaly_results[m].anomaly_score * 100 for m in MACHINES]
    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_scores + [radar_scores[0]],
        theta=MACHINES + [MACHINES[0]],
        fill="toself",
        fillcolor="rgba(52,152,219,0.15)",
        line=dict(color="#3498db", width=2),
        name="Anomaly Score",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#21262d",
                            tickfont=dict(color="#8b949e")),
            angularaxis=dict(tickfont=dict(color="#c9d1d9")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=30, b=20, l=30, r=30),
        font=dict(color="#c9d1d9"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_bn:
    st.subheader("🔍 Bottleneck Analysis")
    if bottleneck:
        bid   = bottleneck.get("bottleneck_id", "Unknown")
        score = bottleneck.get("score", 0)
        lost  = bottleneck.get("lost_throughput_pph", 0)
        rec   = bottleneck.get("recommendation", "")
        ru    = bottleneck.get("runner_up")

        st.error(f"**Bottleneck detected: {bid}**")
        b1, b2 = st.columns(2)
        b1.metric("Severity Score", f"{score:.2f}", "/ 1.0")
        b2.metric("Lost Throughput", f"{lost:.1f} pph")
        st.markdown(f"**Recommendation:** {rec}")
        if ru:
            st.caption(f"Runner-up: {ru}")

        # Bar chart of bottleneck scores
        all_scores = bottleneck.get("all_scores", {})
        if all_scores:
            fig_bn = go.Figure(go.Bar(
                x=list(all_scores.keys()),
                y=list(all_scores.values()),
                marker_color=[
                    "#e74c3c" if mid == bid else "#3498db"
                    for mid in all_scores.keys()
                ],
            ))
            fig_bn.update_layout(
                **plotly_layout("Bottleneck Score by Machine", height=200),
                yaxis_title="Score",
            )
            st.plotly_chart(fig_bn, use_container_width=True)
    else:
        st.success("✅ No bottleneck detected — production is balanced")

st.divider()

# --------------------------------------------------------------------------
# ─── DEGRADATION TIMELINE ────────────────────────────────────────────────
# --------------------------------------------------------------------------

st.subheader("📉 Machine Degradation State")
deg_states = plc.get_degradation_states()
fig_deg = go.Figure()
for mid, deg in deg_states.items():
    color = "#e74c3c" if deg > 0.7 else "#f39c12" if deg > 0.4 else "#2ecc71"
    fig_deg.add_trace(go.Bar(
        name=mid, x=[mid], y=[round(deg * 100, 1)],
        marker_color=color, text=[f"{deg*100:.1f}%"], textposition="outside",
    ))
fig_deg.update_layout(
    **plotly_layout("Degradation Level (%)", height=260),
    yaxis=dict(range=[0, 110], title="Degradation %", gridcolor="#21262d"),
    showlegend=False,
    barmode="group",
)
st.plotly_chart(fig_deg, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# ─── ALERTS & EVENTS ──────────────────────────────────────────────────────
# --------------------------------------------------------------------------

col_alerts, col_schedule = st.columns([3, 2])

with col_alerts:
    st.subheader("🚨 Alert Log")
    alerts = list(reversed(st.session_state.alerts[-15:]))
    if alerts:
        for a in alerts:
            css_cls = "alert-critical" if a["level"] == "CRITICAL" else \
                      "alert-warning"  if a["level"] in ("HIGH", "WARNING") else "alert-info"
            st.markdown(
                f'<div class="{css_cls}">'
                f'<code>{a["time"]}</code> <b>[{a["level"]}]</b> {a["msg"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ All systems nominal — no active alerts")

with col_schedule:
    st.subheader("📋 Production Schedule (EDD)")
    sample_jobs = scheduler.generate_sample_jobs(6)
    sched = scheduler.schedule_edd(sample_jobs, "CNC-001")
    sched_df = pd.DataFrame([
        {
            "Job":       sj.job.job_id,
            "Type":      sj.job.part_type,
            "Qty":       sj.job.quantity,
            "OnTime":    "✅" if not sj.is_late else "❌",
            "Tardiness": f"{sj.tardiness:.0f}s",
        }
        for sj in sched.jobs[:6]
    ])
    st.dataframe(sched_df, hide_index=True, use_container_width=True)
    s1, s2 = st.columns(2)
    s1.metric("Makespan", f"{sched.makespan/3600:.2f} hr")
    s2.metric("On-Time %", f"{sched.on_time_pct:.0f}%")

st.divider()

# --------------------------------------------------------------------------
# ─── SIDEBAR — CONTROL PANEL ──────────────────────────────────────────────
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.divider()

    st.subheader("Simulation Controls")
    refresh_rate = st.slider("Refresh interval (s)", 1, 10, 2)

    st.divider()
    st.subheader("Fault Injection")
    fault_target = st.selectbox("Target machine", MACHINES)

    col_inj, col_clr = st.columns(2)
    if col_inj.button("🔴 Inject", use_container_width=True):
        plc.inject_fault(fault_target)
        st.session_state.alerts.append({
            "time": time.strftime("%H:%M:%S"),
            "level": "CRITICAL",
            "msg": f"Manual fault injected: {fault_target}",
        })
        st.warning(f"Fault injected → {fault_target}")

    if col_clr.button("🟢 Clear", use_container_width=True):
        plc.clear_fault(fault_target)
        st.session_state.alerts.append({
            "time": time.strftime("%H:%M:%S"),
            "level": "INFO",
            "msg": f"Fault cleared: {fault_target}",
        })
        st.success(f"Fault cleared → {fault_target}")

    st.divider()
    st.subheader("Degradation States")
    for mid, deg in plc.get_degradation_states().items():
        deg_color = "🔴" if deg > 0.7 else "🟡" if deg > 0.4 else "🟢"
        st.progress(float(deg), text=f"{deg_color} {mid}: {deg*100:.1f}%")

    st.divider()
    st.subheader("System Info")
    st.info(
        f"Tick: **{st.session_state.tick}**\n\n"
        f"Machines: **{len(MACHINES)}**\n\n"
        f"Anomalies: **{anomaly_count}**\n\n"
        f"High Risk: **{high_risk}**"
    )

    if st.button("🗑 Clear Alerts"):
        st.session_state.alerts = []
        st.success("Alerts cleared")

    if st.button("♻️ Reset History"):
        st.session_state.history = {m: [] for m in MACHINES}
        st.session_state.alerts  = []
        st.session_state.tick    = 0
        st.rerun()

# --------------------------------------------------------------------------
# ─── AUTO-REFRESH ────────────────────────────────────────────────────────
# --------------------------------------------------------------------------

time.sleep(refresh_rate)
st.rerun()
