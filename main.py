from nicegui import ui
import numpy as np
import json

# ── Numerical derivative (central difference) ──────────────────────────────
def numerical_derivative(f_vals: list[float], x_vals: list[float]) -> list[float]:
    """Returns f'(x) using central differences, forward/backward at edges."""
    n = len(f_vals)
    deriv = []
    for i in range(n):
        if i == 0:
            d = (f_vals[1] - f_vals[0]) / (x_vals[1] - x_vals[0])
        elif i == n - 1:
            d = (f_vals[-1] - f_vals[-2]) / (x_vals[-1] - x_vals[-2])
        else:
            d = (f_vals[i + 1] - f_vals[i - 1]) / (x_vals[i + 1] - x_vals[i - 1])
        deriv.append(round(d, 6))
    return deriv

# ── Safe function evaluator ─────────────────────────────────────────────────
ALLOWED = {k: getattr(np, k) for k in dir(np) if not k.startswith("_")}
ALLOWED.update({"abs": abs, "round": round})

def safe_eval(expr: str, x):
    try:
        return float(eval(expr, {"__builtins__": {}}, {**ALLOWED, "x": x}))
    except Exception:
        return None

# ── Page layout ─────────────────────────────────────────────────────────────
ui.add_head_html("""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #0d0f14;
    --surface: #141720;
    --card:    #1a1e2b;
    --border:  #2a2f40;
    --accent:  #5cffb0;
    --accent2: #ff6b6b;
    --accent3: #6bbaff;
    --text:    #e8ecf5;
    --muted:   #6b7280;
    --radius:  12px;
  }

  body, .nicegui-content {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
  }

  /* Header */
  .dv-header {
    background: linear-gradient(135deg, #0d0f14 0%, #141720 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px 24px;
    display: flex;
    align-items: flex-end;
    gap: 16px;
  }
  .dv-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
    line-height: 1;
  }
  .dv-title span { color: var(--accent); }
  .dv-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-left: auto;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Main layout */
  .dv-layout {
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 0;
    min-height: calc(100vh - 85px);
  }

  /* Sidebar */
  .dv-sidebar {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 28px 24px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }
  .dv-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }
  .dv-input-group { display: flex; flex-direction: column; gap: 8px; }

  /* Plot button */
  .plot-btn {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 0 14px !important;
    height: 40px !important;
    cursor: pointer !important;
    transition: opacity 0.15s, transform 0.1s !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
  }
  .plot-btn:hover  { opacity: 0.85 !important; }
  .plot-btn:active { transform: scale(0.96) !important; }

  /* Override NiceGUI input styles */
  .nicegui-input .q-field__control,
  .nicegui-select .q-field__control {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
  }
  .nicegui-input .q-field__native,
  .nicegui-select .q-field__native {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
  }
  .q-field__label { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; }
  .q-field--focused .q-field__control { border-color: var(--accent) !important; }
  .q-slider__thumb { color: var(--accent) !important; }
  .q-slider__track-container { color: var(--accent) !important; }

  /* Preset buttons */
  .preset-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .preset-btn {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    padding: 8px 6px !important;
    cursor: pointer !important;
    text-align: center !important;
    transition: border-color 0.15s, background 0.15s !important;
  }
  .preset-btn:hover { border-color: var(--accent) !important; background: #1f2535 !important; }

  /* Info cards */
  .info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
  }
  .info-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .info-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
  }
  .info-card .value.green  { color: var(--accent); }
  .info-card .value.red    { color: var(--accent2); }
  .info-card .value.blue   { color: var(--accent3); }

  /* Chart area */
  .dv-chart-area {
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 4px;
    flex: 1;
  }

  /* Error banner */
  .error-banner {
    background: rgba(255,107,107,0.12);
    border: 1px solid rgba(255,107,107,0.4);
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent2);
    display: none;
  }
  .error-banner.visible { display: block; }

  /* Tangent info bar */
  .tangent-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }
  .tangent-bar .tb-item { display: flex; flex-direction: column; gap: 2px; }
  .tangent-bar .tb-label { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
  .tangent-bar .tb-val   { font-family: 'Space Mono', monospace; font-size: 0.9rem; font-weight: 700; }
  .sep { width: 1px; height: 32px; background: var(--border); }
</style>
""")

# ── App state ────────────────────────────────────────────────────────────────
state = {
    "expr": "x**2",
    "x_min": -5.0,
    "x_max": 5.0,
    "x_pt": 1.0,
    "show_deriv": True,
    "show_tangent": True,
}

PRESETS = [
    ("x²",       "x**2"),
    ("x³",       "x**3"),
    ("sin(x)",   "sin(x)"),
    ("cos(x)",   "cos(x)"),
    ("eˣ",       "exp(x)"),
    ("|x|",      "abs(x)"),
    ("ln(x)",    "log(x)"),
    ("1/x",      "1/x"),
]

N_POINTS = 400

def compute_chart_data():
    expr = state["expr"]
    x_min, x_max = state["x_min"], state["x_max"]
    x_pt = state["x_pt"]
    x_vals = np.linspace(x_min, x_max, N_POINTS).tolist()

    f_vals, d_vals, errors = [], [], []
    for x in x_vals:
        v = safe_eval(expr, x)
        if v is None or not np.isfinite(v):
            f_vals.append(None)
        else:
            f_vals.append(round(v, 6))

    valid = [(x, f) for x, f in zip(x_vals, f_vals) if f is not None]
    if len(valid) < 2:
        return None, "Cannot evaluate function on this range."

    # derivative
    vx, vf = zip(*valid)
    vx, vf = list(vx), list(vf)
    vd = numerical_derivative(vf, vx)

    # interpolate derivative at x_pt
    x_pt_c = max(x_min, min(x_max, x_pt))
    f_at_pt = safe_eval(expr, x_pt_c)
    if f_at_pt is None or not np.isfinite(f_at_pt):
        return None, f"Function undefined at x = {x_pt_c}"

    # numeric deriv at point
    h = (x_max - x_min) / N_POINTS
    fp = safe_eval(expr, x_pt_c + h)
    fm = safe_eval(expr, x_pt_c - h)
    if fp is None or fm is None:
        dydx = 0.0
    else:
        dydx = (fp - fm) / (2 * h)

    # tangent line: y = f(x_pt) + f'(x_pt)*(x - x_pt)
    t_vals = [round(f_at_pt + dydx * (x - x_pt_c), 6) for x in x_vals]

    # y-axis scale
    finite_f = [v for v in f_vals if v is not None and abs(v) < 1e6]
    y_pad = (max(finite_f) - min(finite_f)) * 0.15 if finite_f else 1
    y_min = min(finite_f) - y_pad if finite_f else -5
    y_max = max(finite_f) + y_pad if finite_f else 5

    return {
        "x_vals": x_vals,
        "f_vals": f_vals,
        "d_vals_x": list(vx),
        "d_vals_y": vd,
        "t_vals": t_vals,
        "x_pt": x_pt_c,
        "f_pt": round(f_at_pt, 5),
        "dydx": round(dydx, 5),
        "y_min": y_min,
        "y_max": y_max,
    }, None

# ── Build UI ─────────────────────────────────────────────────────────────────
with ui.element("div").classes("dv-header"):
    with ui.element("div"):
        ui.html('<div class="dv-title">deriv<span>.</span>viz</div>')
    ui.html('<div class="dv-subtitle">Derivative Visualizer &nbsp;·&nbsp; Calculus Tool</div>')

with ui.element("div").classes("dv-layout"):

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with ui.element("div").classes("dv-sidebar"):

        # Function input + Plot button
        ui.html('<div class="dv-section-label">Function  f(x)</div>')
        with ui.row().classes('w-full gap-2 items-center'):
            func_input = ui.input(value=state["expr"], placeholder="e.g. x**2, sin(x)").props('dense outlined').classes('flex-1')
            plot_btn = ui.html('<button class="plot-btn">&#9654; Plot</button>')
        error_el = ui.html('<div class="error-banner" id="err-banner">Invalid expression</div>')

        # Presets
        ui.html('<div class="dv-section-label" style="margin-top:4px">Presets</div>')
        with ui.element("div").classes("preset-grid"):
            for label, expr in PRESETS:
                def make_preset(e):
                    def handler():
                        func_input.value = e
                        state["expr"] = e
                        refresh_chart()
                    return handler
                ui.html(f'<div class="preset-btn" onclick="">{label}</div>').on("click", make_preset(expr))

        # Range
        ui.html('<div class="dv-section-label" style="margin-top:4px">x-range</div>')
        with ui.row().classes("gap-3 w-full"):
            xmin_input = ui.number(label="x min", value=state["x_min"], step=0.5).props('dense outlined').classes("flex-1")
            xmax_input = ui.number(label="x max", value=state["x_max"], step=0.5).props('dense outlined').classes("flex-1")

        # Tangent point — dual slider (normal = integer snap, Shift = fine 0.05 snap)
        ui.html('<div class="dv-section-label" style="margin-top:4px">Tangent point  x₀ &nbsp;<span style="font-size:0.6rem;color:#6b7280">(hold Shift for fine steps)</span></div>')
        with ui.element('div').style('position:relative; width:100%'):
            xpt_slider       = ui.slider(min=state["x_min"], max=state["x_max"], step=1,    value=state["x_pt"]).props('color=teal').classes('w-full xpt-coarse')
            xpt_slider_fine  = ui.slider(min=state["x_min"], max=state["x_max"], step=0.05, value=state["x_pt"]).props('color=teal').classes('w-full xpt-fine').style('position:absolute;top:0;left:0;display:none')
        xpt_label  = ui.html(f'<div style="font-family:Space Mono,monospace;font-size:0.78rem;color:#5cffb0;text-align:center">x₀ = {state["x_pt"]}</div>')

        # Toggles
        ui.html('<div class="dv-section-label" style="margin-top:4px">Display</div>')
        with ui.row().classes("gap-4"):
            toggle_deriv   = ui.checkbox("f ′(x) curve", value=True)
            toggle_tangent = ui.checkbox("Tangent line", value=True)

    # ── CHART AREA ───────────────────────────────────────────────────────────
    with ui.element("div").classes("dv-chart-area"):

        tangent_bar = ui.html("")

        with ui.element("div").classes("chart-card"):
            chart = ui.echart({
                "backgroundColor": "transparent",
                "animation": False,
                "grid": {"top": 24, "right": 24, "bottom": 48, "left": 56},
                "xAxis": {"type": "value", "axisLine": {"lineStyle": {"color": "#2a2f40"}},
                          "splitLine": {"lineStyle": {"color": "#1e2330"}},
                          "axisLabel": {"color": "#6b7280", "fontFamily": "Space Mono"}},
                "yAxis": {"type": "value", "axisLine": {"lineStyle": {"color": "#2a2f40"}},
                          "splitLine": {"lineStyle": {"color": "#1e2330"}},
                          "axisLabel": {"color": "#6b7280", "fontFamily": "Space Mono"}},
                "tooltip": {"trigger": "axis",
                            "backgroundColor": "#1a1e2b",
                            "borderColor": "#2a2f40",
                            "textStyle": {"color": "#e8ecf5", "fontFamily": "Space Mono", "fontSize": 12}},
                "legend": {"data": ["f(x)", "f ′(x)", "Tangent"],
                           "textStyle": {"color": "#6b7280", "fontFamily": "Space Mono", "fontSize": 11},
                           "top": 0},
                "series": []
            }).classes("w-full").style("height: 480px")

# ── Refresh logic ─────────────────────────────────────────────────────────────
def refresh_chart():
    data, err = compute_chart_data()

    if err or data is None:
        ui.run_javascript(f"""
            document.getElementById('err-banner').textContent = '{err or "Error"}';
            document.getElementById('err-banner').className = 'error-banner visible';
        """)
        return

    ui.run_javascript("document.getElementById('err-banner').className = 'error-banner';")

    x_vals  = data["x_vals"]
    f_vals  = data["f_vals"]
    d_x     = data["d_vals_x"]
    d_y     = data["d_vals_y"]
    t_vals  = data["t_vals"]
    x_pt    = data["x_pt"]
    f_pt    = data["f_pt"]
    dydx    = data["dydx"]
    y_min   = data["y_min"]
    y_max   = data["y_max"]

    show_d = toggle_deriv.value
    show_t = toggle_tangent.value

    # Tangent line clipped to y range with padding
    y_pad = (y_max - y_min) * 0.3
    t_clipped = [
        [round(x, 4), round(t, 4)]
        for x, t in zip(x_vals, t_vals)
        if (y_min - y_pad) <= t <= (y_max + y_pad)
    ]

    series = [
        {
            "name": "f(x)",
            "type": "line",
            "data": [[round(x, 4), v] for x, v in zip(x_vals, f_vals) if v is not None],
            "showSymbol": False,
            "lineStyle": {"color": "#5cffb0", "width": 2.5},
            "itemStyle": {"color": "#5cffb0"},
        },
    ]

    if show_d:
        series.append({
            "name": "f ′(x)",
            "type": "line",
            "data": [[round(x, 4), round(y, 4)] for x, y in zip(d_x, d_y)],
            "showSymbol": False,
            "lineStyle": {"color": "#6bbaff", "width": 1.8, "type": "dashed"},
            "itemStyle": {"color": "#6bbaff"},
        })

    if show_t:
        series.append({
            "name": "Tangent",
            "type": "line",
            "data": t_clipped,
            "showSymbol": False,
            "lineStyle": {"color": "#ff6b6b", "width": 1.8, "type": "dotted"},
            "itemStyle": {"color": "#ff6b6b"},
        })

    # Mark the tangent point
    series.append({
        "name": "x₀",
        "type": "scatter",
        "data": [[x_pt, f_pt]],
        "symbolSize": 10,
        "itemStyle": {"color": "#ff6b6b", "borderColor": "#fff", "borderWidth": 2},
        "showSymbol": True,
        "legend": {"show": False},
    })

    slope_sign = "+" if dydx >= 0 else "−"
    slope_str  = f"y = {f_pt} {slope_sign} {abs(dydx)}·(x − {x_pt})"

    # Update echart
    chart.options["series"] = series
    chart.options["yAxis"]["min"] = round(y_min, 2)
    chart.options["yAxis"]["max"] = round(y_max, 2)
    chart.update()

    behavior = "Increasing ↑" if dydx > 0.01 else ("Decreasing ↓" if dydx < -0.01 else "Flat →")
    bcolor   = "#5cffb0" if dydx > 0.01 else ("#ff6b6b" if dydx < -0.01 else "#6bbaff")

    tangent_bar.set_content(f"""
    <div class="tangent-bar">
      <div class="tb-item">
        <span class="tb-label">Point x₀</span>
        <span class="tb-val" style="color:#e8ecf5">{x_pt}</span>
      </div>
      <div class="sep"></div>
      <div class="tb-item">
        <span class="tb-label">f(x₀)</span>
        <span class="tb-val" style="color:#5cffb0">{f_pt}</span>
      </div>
      <div class="sep"></div>
      <div class="tb-item">
        <span class="tb-label">f ′(x₀)  — slope</span>
        <span class="tb-val" style="color:#ff6b6b">{dydx}</span>
      </div>
      <div class="sep"></div>
      <div class="tb-item">
        <span class="tb-label">Tangent equation</span>
        <span class="tb-val" style="color:#6bbaff;font-size:0.78rem">{slope_str}</span>
      </div>
      <div class="sep"></div>
      <div class="tb-item">
        <span class="tb-label">Behavior</span>
        <span class="tb-val" style="color:{bcolor}">{behavior}</span>
      </div>
    </div>
    """)

# ── Event bindings ─────────────────────────────────────────────────────────────
def on_func_change(_=None):
    state["expr"] = func_input.value
    refresh_chart()

def on_range_change():
    try:
        lo = float(xmin_input.value)
        hi = float(xmax_input.value)
        if lo >= hi:
            return
        state["x_min"] = lo
        state["x_max"] = hi
        xpt_slider.min = lo
        xpt_slider.max = hi
        xpt_slider_fine.min = lo
        xpt_slider_fine.max = hi
        state["x_pt"] = max(lo, min(hi, state["x_pt"]))
        xpt_slider.value = state["x_pt"]
        xpt_slider_fine.value = state["x_pt"]
        xpt_slider.update()
        xpt_slider_fine.update()
        refresh_chart()
    except Exception:
        pass

def on_xpt_change(_=None):
    # Read whichever slider is currently active
    val = xpt_slider_fine.value if state.get('fine_mode') else xpt_slider.value
    state["x_pt"] = round(float(val), 3)
    # Keep both sliders in sync
    xpt_slider.value = state["x_pt"]
    xpt_slider_fine.value = state["x_pt"]
    xpt_label.set_content(f'<div style="font-family:Space Mono,monospace;font-size:0.78rem;color:#5cffb0;text-align:center">x₀ = {state["x_pt"]}</div>')
    refresh_chart()

def on_toggle(_):
    state["show_deriv"]   = toggle_deriv.value
    state["show_tangent"] = toggle_tangent.value
    refresh_chart()

func_input.on("keyup.enter", on_func_change)
plot_btn.on("click", on_func_change)
xmin_input.on("blur", lambda _: on_range_change())
xmax_input.on("blur", lambda _: on_range_change())
xmin_input.on("keyup.enter", lambda _: on_range_change())
xmax_input.on("keyup.enter", lambda _: on_range_change())
xpt_slider.on("update:model-value", on_xpt_change)
xpt_slider_fine.on("update:model-value", on_xpt_change)
toggle_deriv.on("update:model-value", on_toggle)
toggle_tangent.on("update:model-value", on_toggle)

# Shift key swaps which slider is visible/active
def on_key(e):
    if e.key == 'Shift':
        fine = e.action == 'keydown'
        state['fine_mode'] = fine
        coarse_display = 'none' if fine else 'block'
        fine_display   = 'block' if fine else 'none'
        ui.run_javascript(
            f'document.querySelectorAll(".xpt-coarse")[0].style.display = "{coarse_display}";'
            f'document.querySelectorAll(".xpt-fine")[0].style.display   = "{fine_display}";'
        )

ui.keyboard(on_key=on_key, ignore=['input', 'select', 'button', 'textarea'])

# Initial draw
ui.timer(0.1, refresh_chart, once=True)

ui.run(dark=True, title="deriv.viz — Derivative Visualizer", port=8080, reload=False)