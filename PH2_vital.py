from pathlib import Path
from math import ceil
from typing import IO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


CSV_PATH = Path(__file__).with_name("lh2_testdata.csv")
MAX_METRICS = 9


def _detect_encoding(file_source: str | IO[bytes]) -> str:
    """UTF-8 → cp932 → latin-1 の順で読めるエンコーディングを返す。"""
    for enc in ("utf-8", "cp932", "latin-1"):
        try:
            if isinstance(file_source, str):
                with open(file_source, encoding=enc) as f:
                    f.read(4096)
            else:
                file_source.seek(0)
                file_source.read(4096).decode(enc)
                file_source.seek(0)
            return enc
        except (UnicodeDecodeError, AttributeError):
            if not isinstance(file_source, str):
                try:
                    file_source.seek(0)
                except Exception:
                    pass
    return "latin-1"


@st.cache_data(show_spinner=False)
def load_units(file_source: str | IO[bytes]) -> dict[str, str]:
    """CSVの2行目が単位行であればカラム名→単位の辞書を返す。Parquetは常に空辞書。"""
    # Parquet には単位行なし
    is_parquet = (
        isinstance(file_source, str) and file_source.endswith(".parquet")
    ) or (
        hasattr(file_source, "name") and getattr(file_source, "name", "").endswith(".parquet")
    )
    if is_parquet:
        return {}
    enc = _detect_encoding(file_source)
    raw = pd.read_csv(file_source, nrows=2, encoding=enc)
    if len(raw) < 1:
        return {}
    first_row = raw.iloc[0]
    first_col = raw.columns[0]
    if pd.isna(pd.to_numeric(first_row.get(first_col, ""), errors="coerce")):
        return {col: str(val) for col, val in first_row.items() if pd.notna(val)}
    return {}


@st.cache_data(show_spinner=False)
def load_data(file_source: str | IO[bytes]) -> pd.DataFrame:
    is_parquet = (
        isinstance(file_source, str) and file_source.endswith(".parquet")
    ) or (
        hasattr(file_source, "name") and getattr(file_source, "name", "").endswith(".parquet")
    )
    if is_parquet:
        dataframe = pd.read_parquet(file_source)
        if len(dataframe.columns) == 0:
            raise ValueError("Parquetに列がありません。")
        for col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
    else:
        enc = _detect_encoding(file_source)
        dataframe = pd.read_csv(file_source, encoding=enc)
        if len(dataframe.columns) == 0:
            raise ValueError("CSVに列がありません。")
        first_col = dataframe.columns[0]
        numeric_mask = pd.to_numeric(dataframe[first_col], errors="coerce").notna()
        dataframe = dataframe[numeric_mask].reset_index(drop=True)
        for col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")

    dataframe = dataframe.sort_values("time").reset_index(drop=True)
    return dataframe


def metric_summary(dataframe: pd.DataFrame, metric_name: str) -> tuple[str, str, str]:
    series = pd.to_numeric(dataframe[metric_name], errors="coerce")
    if series.notna().sum() == 0:
        return "N/A", "min N/A", "max N/A"
    latest_value = series.dropna().iloc[-1]
    minimum_value = series.min()
    maximum_value = series.max()
    return (
        f"{latest_value:.3f}",
        f"min {minimum_value:.3f}",
        f"max {maximum_value:.3f}",
    )


def build_threshold_segments(
    dataframe: pd.DataFrame,
    metric_name: str,
    threshold: float,
    x_col: str = "time",
) -> pd.DataFrame:
    time_values = dataframe[x_col].tolist()
    y_values = pd.to_numeric(dataframe[metric_name], errors="coerce").tolist()

    segmented_rows: list[dict[str, float | None]] = []

    def append_point(time_value: float, y_value: float, force_exceed: bool | None = None) -> None:
        is_exceed = force_exceed if force_exceed is not None else y_value > threshold
        segmented_rows.append(
            {
                x_col: time_value,
                "normal": None if is_exceed else y_value,
                "exceed": y_value if is_exceed else None,
            }
        )

    if not time_values:
        return pd.DataFrame(columns=[x_col, "normal", "exceed"])

    append_point(time_values[0], y_values[0])

    for index in range(len(time_values) - 1):
        start_time = time_values[index]
        end_time = time_values[index + 1]
        start_value = y_values[index]
        end_value = y_values[index + 1]

        if pd.notna(start_value) and pd.notna(end_value):
            crossed_threshold = (start_value - threshold) * (end_value - threshold) < 0
            if crossed_threshold and end_time != start_time:
                ratio = (threshold - start_value) / (end_value - start_value)
                cross_time = start_time + (end_time - start_time) * ratio
                start_exceeds = start_value > threshold
                append_point(cross_time, threshold, force_exceed=start_exceeds)
                append_point(cross_time, threshold, force_exceed=not start_exceeds)

        append_point(end_time, end_value)

    return pd.DataFrame(segmented_rows)


def render_chart(
    dataframe: pd.DataFrame,
    metric_name: str,
    show_markers: bool,
    threshold: float,
    units: dict[str, str] | None = None,
    x_col: str = "time",
) -> None:
    segmented_df = build_threshold_segments(dataframe, metric_name, threshold, x_col)
    marker_style = dict(size=5) if show_markers else None

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=segmented_df[x_col],
            y=segmented_df["normal"],
            mode="lines+markers" if show_markers else "lines",
            line=dict(color="#137dad", width=2.5),
            marker=marker_style,
            name="閾値以下",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=segmented_df[x_col],
            y=segmented_df["exceed"],
            mode="lines+markers" if show_markers else "lines",
            line=dict(color="#d62839", width=2.5),
            marker=marker_style,
            name="閾値超過",
            showlegend=False,
        )
    )
    has_data = pd.to_numeric(dataframe[metric_name], errors="coerce").notna().any()

    if has_data:
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#0b2940",
            line_width=1.5,
            annotation_text=f"最大閾値 {threshold:.3f}",
            annotation_position="top left",
        )

    layout_annotations = []
    if not has_data:
        layout_annotations.append(
            dict(
                text="No Data",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28, color="rgba(200, 80, 80, 0.45)"),
            )
        )

    _units = units or {}
    x_unit = _units.get(x_col, "")
    metric_unit = _units.get(metric_name, "")
    xaxis_label = f"{x_col} [{x_unit}]" if x_unit else x_col
    yaxis_label = f"{metric_name} [{metric_unit}]" if metric_unit else metric_name

    figure.update_layout(
        template="plotly_white",
        title=metric_name,
        margin=dict(l=16, r=16, t=56, b=12),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=xaxis_label,
        yaxis_title=yaxis_label,
        annotations=layout_annotations,
    )
    st.plotly_chart(figure, width="stretch")


st.set_page_config(page_title="LH2 Public Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(210, 240, 255, 0.9), transparent 28%),
            radial-gradient(circle at top right, rgba(255, 232, 203, 0.9), transparent 24%),
            linear-gradient(180deg, #f7fbff 0%, #eef4f7 100%);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(9, 74, 108, 0.96), rgba(19, 125, 173, 0.9));
        color: #ffffff;
        box-shadow: 0 20px 50px rgba(9, 74, 108, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .hero p {
        margin: 0.4rem 0 0;
        color: rgba(255, 255, 255, 0.88);
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(9, 74, 108, 0.08);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }
    .metric-card p {
        margin: 0;
        color: #305164;
        font-size: 0.92rem;
    }
    .metric-card h3 {
        margin: 0.2rem 0;
        color: #0b2940;
        font-size: 1.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>LH2 Public Dashboard</h1>
        <p>time を基準に、CSVの主要系列をすぐ共有できる形で可視化します。サンプルCSVの表示にも、任意CSVのアップロードにも対応します。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("表示設定")
    uploaded_file = st.file_uploader("CSV / Parquetをアップロード", type=["csv", "parquet"])
    chart_columns = st.slider("1行あたりのグラフ数", min_value=1, max_value=3, value=3)
    use_markers = st.toggle("折れ線にマーカーを表示", value=False)
    with st.expander("描画最適化", expanded=False):
        enable_downsample = st.toggle("大容量時に間引いて描画", value=True)
        max_plot_points = st.slider("表示点数の上限", min_value=300, max_value=5000, value=1200, step=100)

try:
    data_source = uploaded_file if uploaded_file is not None else str(CSV_PATH)
    df = load_data(data_source)
    units = load_units(data_source)
except Exception as exc:
    st.error(f"CSVの読み込みに失敗しました: {exc}")
    st.stop()

x_col = "time"

metric_columns = [column for column in df.columns if column != x_col and column != "lap"]
default_metrics = metric_columns[:MAX_METRICS]

with st.sidebar:
    selected_metrics = st.multiselect(
        "表示する系列",
        options=metric_columns,
        default=default_metrics,
        max_selections=MAX_METRICS,
        help="公開画面向けに最大9系列まで選択できます。",
    )

if not selected_metrics:
    st.warning("少なくとも1系列を選択してください。")
    st.stop()

thresholds: dict[str, float] = {}
with st.sidebar.expander("系列ごとの最大閾値", expanded=True):
    def sync_from_input(value_key: str, input_key: str, slider_key: str) -> None:
        st.session_state[value_key] = float(st.session_state[input_key])
        st.session_state[slider_key] = float(st.session_state[input_key])

    def sync_from_slider(value_key: str, input_key: str, slider_key: str) -> None:
        st.session_state[value_key] = float(st.session_state[slider_key])
        st.session_state[input_key] = float(st.session_state[slider_key])

    for metric_name in selected_metrics:
        series = pd.to_numeric(df[metric_name], errors="coerce")
        metric_min = float(series.min()) if series.notna().any() else 0.0
        metric_max = float(series.max()) if series.notna().any() else 1.0
        if metric_min == metric_max:
            metric_min -= 1.0
            metric_max += 1.0

        value_key = f"threshold_value_{metric_name}"
        input_key = f"threshold_input_{metric_name}"
        slider_key = f"threshold_slider_{metric_name}"

        if value_key not in st.session_state:
            st.session_state[value_key] = float(series.max()) if series.notna().any() else 0.0
        if input_key not in st.session_state:
            st.session_state[input_key] = st.session_state[value_key]
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state[value_key]

        st.markdown(f"**{metric_name}**")
        input_col, slider_col = st.columns([1, 2])

        with input_col:
            st.number_input(
                "直接入力",
                key=input_key,
                format="%.3f",
                label_visibility="collapsed",
                on_change=sync_from_input,
                args=(value_key, input_key, slider_key),
            )
        with slider_col:
            st.slider(
                "スライダー",
                min_value=metric_min,
                max_value=metric_max,
                key=slider_key,
                step=(metric_max - metric_min) / 200 if metric_max > metric_min else 0.01,
                label_visibility="collapsed",
                on_change=sync_from_slider,
                args=(value_key, input_key, slider_key),
            )

        thresholds[metric_name] = float(st.session_state[value_key])

plot_df = df
if enable_downsample and len(df) > max_plot_points:
    step = ceil(len(df) / max_plot_points)
    plot_df = df.iloc[::step].copy()
    if len(plot_df) > 0 and plot_df.iloc[-1][x_col] != df.iloc[-1][x_col]:
        plot_df = pd.concat([plot_df, df.tail(1)], ignore_index=True)
    st.caption(f"表示高速化のため {len(df):,} 行を {len(plot_df):,} 点に間引いて表示しています。")

METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("LH2Tank",   ["pLH2Tank", "xLH2TankLevel", "pLH2BufferTank", "TLH2BufferTank"]),
    ("Vaporizer", ["TLH2VaporizerIn", "TGH2VaporizerOut", "pLH2VaporizerIn"]),
    ("Regulator", ["pGH2RegIn", "pGH2regOut"]),
]

overview_tab, data_tab = st.tabs(["ダッシュボード", "データ"])


def _render_metric_grid(metrics: list[str]) -> None:
    for start_index in range(0, len(metrics), chart_columns):
        grid_columns = st.columns(chart_columns)
        for chart_container, metric_name in zip(
            grid_columns,
            metrics[start_index:start_index + chart_columns],
        ):
            latest_value, min_label, max_label = metric_summary(df, metric_name)
            exceed_count = int((pd.to_numeric(df[metric_name], errors="coerce") > thresholds[metric_name]).sum())
            with chart_container:
                st.caption(
                    f"最新値 {latest_value} | {min_label} | {max_label} | 超過点数 {exceed_count}"
                )
                render_chart(plot_df, metric_name, use_markers, thresholds[metric_name], units, x_col)


with overview_tab:
    if uploaded_file is None:
        st.info("現在は同梱サンプルCSVを表示しています。共有時はサイドバーから任意CSVへ切り替えできます。")

    rendered: set[str] = set()

    for group_name, group_members in METRIC_GROUPS:
        metrics_in_group = [m for m in group_members if m in selected_metrics]
        if not metrics_in_group:
            continue
        st.markdown(
            f"<p style='font-weight:900; color:#000; font-size:1.6rem; margin:1.2rem 0 0.1rem;'>{group_name}</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        _render_metric_grid(metrics_in_group)
        rendered.update(metrics_in_group)

    ungrouped = [m for m in selected_metrics if m not in rendered]
    if ungrouped:
        st.markdown(
            "<p style='font-weight:900; color:#000; font-size:1.6rem; margin:1.2rem 0 0.1rem;'>その他</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        _render_metric_grid(ungrouped)

with data_tab:
    st.subheader("プレビュー")
    st.dataframe(df.head(20), width="stretch")
    st.download_button(
        label="現在のCSVをダウンロード",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="lh2_dashboard_export.csv",
        mime="text/csv",
    )