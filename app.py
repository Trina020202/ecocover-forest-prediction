from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "data" / "DatasetC_Forest_Cover_Cleaned.csv"

NUMERIC_FEATURES = ["Elevation", "avg_hillshade"]
CATEGORICAL_FEATURES = ["Soil_Type", "Area"]
TARGET = "Forest_Cover"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]

CLASS_PROFILES = [
    {
        "Forest_Cover": "Aspen",
        "count": 246,
        "share": 0.0170,
        "elevation_mu": 2880,
        "elevation_sd": 320,
        "shade_mu": 205,
        "shade_sd": 22,
        "soils": [29, 30, 31, 32, 33, 34],
        "areas": {"Comanche Peak": 0.35, "Rawah": 0.20, "Cache la Poudre": 0.10, "Neota": 0.35},
        "color": "#b7791f",
    },
    {
        "Forest_Cover": "Cottonwood/Willow",
        "count": 94,
        "share": 0.0065,
        "elevation_mu": 2140,
        "elevation_sd": 210,
        "shade_mu": 191,
        "shade_sd": 25,
        "soils": [1, 2, 3, 4, 5, 6],
        "areas": {"Cache la Poudre": 0.55, "Comanche Peak": 0.18, "Rawah": 0.12, "Neota": 0.15},
        "color": "#2f6fa7",
    },
    {
        "Forest_Cover": "Douglas-fir",
        "count": 652,
        "share": 0.0450,
        "elevation_mu": 2510,
        "elevation_sd": 330,
        "shade_mu": 193,
        "shade_sd": 25,
        "soils": [10, 11, 12, 13, 14, 15, 16, 17],
        "areas": {"Cache la Poudre": 0.35, "Comanche Peak": 0.30, "Rawah": 0.20, "Neota": 0.15},
        "color": "#7b4aa0",
    },
    {
        "Forest_Cover": "Krummholz",
        "count": 500,
        "share": 0.0345,
        "elevation_mu": 3450,
        "elevation_sd": 350,
        "shade_mu": 181,
        "shade_sd": 28,
        "soils": [35, 36, 37, 38, 39, 40],
        "areas": {"Neota": 0.48, "Rawah": 0.31, "Comanche Peak": 0.17, "Cache la Poudre": 0.04},
        "color": "#1f8a9b",
    },
    {
        "Forest_Cover": "Lodgepole Pine",
        "count": 6667,
        "share": 0.4598,
        "elevation_mu": 2860,
        "elevation_sd": 420,
        "shade_mu": 205,
        "shade_sd": 21,
        "soils": [18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32],
        "areas": {"Comanche Peak": 0.45, "Rawah": 0.34, "Cache la Poudre": 0.16, "Neota": 0.05},
        "color": "#1f7a52",
    },
    {
        "Forest_Cover": "Ponderosa Pine",
        "count": 1308,
        "share": 0.0902,
        "elevation_mu": 2290,
        "elevation_sd": 285,
        "shade_mu": 184,
        "shade_sd": 27,
        "soils": [6, 9, 10, 11, 12, 13, 14, 17],
        "areas": {"Cache la Poudre": 0.50, "Comanche Peak": 0.24, "Rawah": 0.16, "Neota": 0.10},
        "color": "#c4553d",
    },
    {
        "Forest_Cover": "Spruce/Fir",
        "count": 5033,
        "share": 0.3471,
        "elevation_mu": 3180,
        "elevation_sd": 390,
        "shade_mu": 197,
        "shade_sd": 23,
        "soils": [22, 23, 24, 29, 30, 31, 32, 33, 38, 39, 40],
        "areas": {"Rawah": 0.43, "Comanche Peak": 0.36, "Neota": 0.17, "Cache la Poudre": 0.04},
        "color": "#6d7d2f",
    },
]

MODEL_RESULTS = pd.DataFrame(
    [
        {
            "Model": "Random Forest",
            "Accuracy": 0.617241,
            "Precision_macro": 0.458425,
            "Recall_macro": 0.662787,
            "F1_macro": 0.493775,
        },
        {
            "Model": "Decision Tree",
            "Accuracy": 0.532069,
            "Precision_macro": 0.412443,
            "Recall_macro": 0.624582,
            "F1_macro": 0.434131,
        },
        {
            "Model": "Logistic Regression",
            "Accuracy": 0.506897,
            "Precision_macro": 0.408942,
            "Recall_macro": 0.588029,
            "F1_macro": 0.420206,
        },
    ]
)

CLEANING_RESULTS = pd.DataFrame(
    [
        {"Feature": "Elevation", "Before": 16.0, "After": 0.0},
        {"Feature": "Hillshade_9am", "Before": 30.0, "After": 0.0},
        {"Feature": "Hillshade_Noon", "Before": 12.0, "After": 0.0},
        {"Feature": "Hillshade_3pm", "Before": 28.0, "After": 0.0},
        {"Feature": "Unknown Soil_Type rows", "Before": 33.47, "After": 0.0},
    ]
)

COLOR_MAP = {profile["Forest_Cover"]: profile["color"] for profile in CLASS_PROFILES}
AREAS = ["Comanche Peak", "Rawah", "Cache la Poudre", "Neota"]
SOILS = [f"Soil_Type{i}" for i in range(1, 41)]


st.set_page_config(
    page_title="EcoCover | Forest Cover Prediction",
    page_icon="EC",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --forest: #1f7a52;
            --forest-dark: #14543a;
            --ink: #17201b;
            --muted: #607064;
            --soft: #edf2ea;
            --border: #d8e0d5;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f5f7f4, #edf2ea);
        }
        .hero {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            background: linear-gradient(135deg, rgba(31, 122, 82, .10), rgba(47, 111, 167, .07));
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            color: var(--ink);
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: var(--muted);
            max-width: 880px;
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
            background: #ffffff;
            box-shadow: 0 12px 32px rgba(32, 54, 41, .08);
            min-height: 112px;
        }
        .metric-card span {
            display: block;
            color: var(--muted);
            font-size: .86rem;
            margin-bottom: .3rem;
        }
        .metric-card strong {
            display: block;
            color: var(--ink);
            font-size: 1.55rem;
            line-height: 1.1;
        }
        .metric-card small {
            color: var(--muted);
        }
        .status-note {
            border-left: 4px solid var(--forest);
            background: var(--soft);
            padding: .8rem .95rem;
            border-radius: 8px;
            color: var(--ink);
            margin: .4rem 0 1rem 0;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #14543a;
            background: #14543a;
            color: #ffffff;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: var(--forest-dark);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def make_demo_dataset(seed: int = 42, target_rows: int = 14500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for profile in CLASS_PROFILES:
        n_rows = int(round(profile["share"] * target_rows))
        n_rows = max(n_rows, 90)
        area_names = list(profile["areas"].keys())
        area_probs = np.array(list(profile["areas"].values()), dtype=float)
        area_probs = area_probs / area_probs.sum()

        elevations = rng.normal(profile["elevation_mu"], profile["elevation_sd"], n_rows)
        shades = rng.normal(profile["shade_mu"], profile["shade_sd"], n_rows)
        soil_choices = rng.choice(profile["soils"], size=n_rows, replace=True)
        area_choices = rng.choice(area_names, size=n_rows, replace=True, p=area_probs)

        for elevation, shade, soil, area in zip(elevations, shades, soil_choices, area_choices):
            rows.append(
                {
                    "Elevation": float(np.clip(elevation, 1871, 6704)),
                    "avg_hillshade": float(np.clip(shade, 0, 254)),
                    "Soil_Type": f"Soil_Type{int(soil)}",
                    "Area": str(area),
                    "Forest_Cover": profile["Forest_Cover"],
                }
            )

    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def normalize_uploaded_data(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    data = pd.read_csv(uploaded_file)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        st.sidebar.error(f"Uploaded CSV is missing: {', '.join(missing)}")
        return None

    data = data[REQUIRED_COLUMNS].dropna().copy()
    data["Soil_Type"] = data["Soil_Type"].astype(str)
    data["Area"] = data["Area"].astype(str)
    data["Forest_Cover"] = data["Forest_Cover"].astype(str)
    data["Elevation"] = pd.to_numeric(data["Elevation"], errors="coerce")
    data["avg_hillshade"] = pd.to_numeric(data["avg_hillshade"], errors="coerce")
    return data.dropna()


@st.cache_data
def load_default_data() -> tuple[pd.DataFrame, str]:
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        data = data[REQUIRED_COLUMNS].dropna().copy()
        return data, "real"
    return make_demo_dataset(), "demo"


@st.cache_resource
def train_model(data: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    x = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    test_size = 0.2 if len(data) >= 250 else 0.3
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
    }
    return pipeline, metrics


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def prediction_frame(model: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["model"].classes_
    output = pd.DataFrame({"Forest_Cover": classes, "Probability": probabilities})
    output["Color"] = output["Forest_Cover"].map(COLOR_MAP).fillna("#607064")
    return output.sort_values("Probability", ascending=False).reset_index(drop=True)


def feature_group_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    importances = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})

    def group_name(feature: str) -> str:
        if feature.startswith("num__Elevation"):
            return "Elevation"
        if feature.startswith("num__avg_hillshade"):
            return "avg_hillshade"
        if feature.startswith("cat__Soil_Type"):
            return "Soil_Type"
        if feature.startswith("cat__Area"):
            return "Area"
        return "Other"

    importances["group"] = importances["feature"].map(group_name)
    grouped = importances.groupby("group", as_index=False)["importance"].sum()
    grouped["importance"] = grouped["importance"] / grouped["importance"].sum()
    return grouped.sort_values("importance", ascending=False)


def metric_cards(data: pd.DataFrame, source: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Clean samples", f"{len(data):,}", "model-ready rows"),
        ("Cover classes", f"{data[TARGET].nunique()}", "multi-class target"),
        ("Reported accuracy", "61.72%", "best Random Forest"),
        ("Reported macro recall", "66.28%", "class-balanced metric"),
    ]
    for column, (label, value, caption) in zip([col1, col2, col3, col4], cards):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>{label}</span>
                    <strong>{value}</strong>
                    <small>{caption}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if source == "demo":
        st.markdown(
            """
            <div class="status-note">
                Live predictor is running in demo mode because the cleaned CSV/model artifact is not bundled.
                The reported model metrics come from the submitted COMP5310 notebook; add
                <code>data/DatasetC_Forest_Cover_Cleaned.csv</code> to train from the real cleaned dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )


def probability_chart(probabilities: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        probabilities.sort_values("Probability"),
        x="Probability",
        y="Forest_Cover",
        orientation="h",
        text=probabilities.sort_values("Probability")["Probability"].map(lambda value: f"{value:.1%}"),
        color="Forest_Cover",
        color_discrete_map=COLOR_MAP,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=360,
        showlegend=False,
        xaxis_title="Predicted probability",
        yaxis_title=None,
        xaxis_tickformat=".0%",
        margin=dict(l=10, r=60, t=20, b=35),
    )
    return fig


def class_distribution_chart(data: pd.DataFrame) -> go.Figure:
    dist = data[TARGET].value_counts().rename_axis("Forest_Cover").reset_index(name="Samples")
    dist["Share"] = dist["Samples"] / dist["Samples"].sum()
    fig = px.bar(
        dist.sort_values("Samples"),
        x="Samples",
        y="Forest_Cover",
        orientation="h",
        text=dist.sort_values("Samples")["Share"].map(lambda value: f"{value:.2%}"),
        color="Forest_Cover",
        color_discrete_map=COLOR_MAP,
    )
    fig.update_layout(
        height=420,
        showlegend=False,
        xaxis_title="Samples",
        yaxis_title=None,
        margin=dict(l=10, r=50, t=20, b=35),
    )
    return fig


def feature_space_chart(data: pd.DataFrame, current_row: pd.DataFrame, predicted_class: str) -> go.Figure:
    sample = data.sample(min(len(data), 1600), random_state=7)
    fig = px.scatter(
        sample,
        x="Elevation",
        y="avg_hillshade",
        color="Forest_Cover",
        color_discrete_map=COLOR_MAP,
        opacity=0.55,
        hover_data=["Soil_Type", "Area"],
    )
    fig.add_trace(
        go.Scatter(
            x=[current_row["Elevation"].iloc[0]],
            y=[current_row["avg_hillshade"].iloc[0]],
            mode="markers+text",
            marker=dict(size=18, color=COLOR_MAP.get(predicted_class, "#14543a"), line=dict(width=3, color="#ffffff")),
            text=["Current input"],
            textposition="top center",
            name="Current input",
        )
    )
    fig.update_layout(
        height=520,
        legend_title_text="Forest cover",
        margin=dict(l=10, r=10, t=20, b=35),
    )
    return fig


def model_comparison_chart() -> go.Figure:
    long_df = MODEL_RESULTS.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=long_df["Score"].map(lambda value: f"{value:.1%}"),
        color_discrete_map={
            "Random Forest": "#1f7a52",
            "Decision Tree": "#2f6fa7",
            "Logistic Regression": "#b7791f",
        },
    )
    fig.update_layout(
        height=460,
        yaxis_tickformat=".0%",
        yaxis_title="Score",
        xaxis_title=None,
        legend_title_text=None,
        margin=dict(l=10, r=10, t=20, b=35),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def cleaning_chart() -> go.Figure:
    long_df = CLEANING_RESULTS.melt(id_vars="Feature", var_name="Stage", value_name="Missing rate")
    fig = px.bar(
        long_df,
        x="Feature",
        y="Missing rate",
        color="Stage",
        barmode="group",
        text=long_df["Missing rate"].map(lambda value: f"{value:.1f}%"),
        color_discrete_map={"Before": "#c4553d", "After": "#1f7a52"},
    )
    fig.update_layout(
        height=420,
        yaxis_title="Missing / affected rows",
        xaxis_title=None,
        yaxis_ticksuffix="%",
        legend_title_text=None,
        margin=dict(l=10, r=10, t=20, b=80),
    )
    return fig


def sidebar_inputs(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Prediction input")
    uploaded_file = st.sidebar.file_uploader("Optional cleaned CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state["uploaded_data"] = normalize_uploaded_data(uploaded_file)

    st.sidebar.divider()
    elevation = st.sidebar.slider("Elevation (m)", 1871, 6704, 3016, 1)
    avg_hillshade = st.sidebar.slider("Average hillshade", 0.0, 254.0, 197.0, 0.5)

    soil_options = sorted(data["Soil_Type"].astype(str).unique(), key=lambda value: int(value.replace("Soil_Type", "")) if value.replace("Soil_Type", "").isdigit() else 999)
    area_options = [area for area in AREAS if area in set(data["Area"])] or sorted(data["Area"].unique())

    soil_type = st.sidebar.selectbox("Soil type", soil_options, index=soil_options.index("Soil_Type29") if "Soil_Type29" in soil_options else 0)
    area = st.sidebar.selectbox("Wilderness area", area_options, index=area_options.index("Comanche Peak") if "Comanche Peak" in area_options else 0)

    return pd.DataFrame(
        [
            {
                "Elevation": elevation,
                "avg_hillshade": avg_hillshade,
                "Soil_Type": soil_type,
                "Area": area,
            }
        ]
    )


def render_predict_tab(data: pd.DataFrame, model: Pipeline, metrics: dict[str, float], current_row: pd.DataFrame) -> None:
    probabilities = prediction_frame(model, current_row)
    predicted = probabilities.iloc[0]

    left, right = st.columns([0.95, 1.25], gap="large")
    with left:
        st.subheader("Prediction")
        st.metric("Predicted cover type", predicted["Forest_Cover"], format_percent(predicted["Probability"]))
        st.dataframe(
            current_row.rename(
                columns={
                    "Elevation": "Elevation (m)",
                    "avg_hillshade": "Average hillshade",
                    "Soil_Type": "Soil type",
                    "Area": "Area",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.plotly_chart(probability_chart(probabilities), use_container_width=True)

    with right:
        st.subheader("Feature space")
        st.plotly_chart(feature_space_chart(data, current_row, predicted["Forest_Cover"]), use_container_width=True)

    st.subheader("Feature importance")
    importance = feature_group_importance(model)
    fig = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="group",
        orientation="h",
        text=importance.sort_values("importance")["importance"].map(lambda value: f"{value:.1%}"),
        color_discrete_sequence=["#1f7a52"],
    )
    fig.update_layout(height=290, xaxis_tickformat=".0%", xaxis_title="Aggregated importance", yaxis_title=None, margin=dict(l=10, r=40, t=10, b=35))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Runtime model check"):
        st.write(
            pd.DataFrame(
                [
                    {"Metric": "Accuracy", "Score": metrics["accuracy"]},
                    {"Metric": "Precision macro", "Score": metrics["precision_macro"]},
                    {"Metric": "Recall macro", "Score": metrics["recall_macro"]},
                    {"Metric": "F1 macro", "Score": metrics["f1_macro"]},
                ]
            ).assign(Score=lambda frame: frame["Score"].map(lambda value: f"{value:.4f}"))
        )


def render_data_tab(data: pd.DataFrame) -> None:
    col1, col2 = st.columns([1.15, 0.85], gap="large")
    with col1:
        st.subheader("Class distribution")
        st.plotly_chart(class_distribution_chart(data), use_container_width=True)
    with col2:
        st.subheader("Cleaned data profile")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Rows", "Value": f"{len(data):,}"},
                    {"Field": "Columns", "Value": f"{len(REQUIRED_COLUMNS)}"},
                    {"Field": "Forest cover classes", "Value": f"{data[TARGET].nunique()}"},
                    {"Field": "Soil types", "Value": f"{data['Soil_Type'].nunique()}"},
                    {"Field": "Wilderness areas", "Value": f"{data['Area'].nunique()}"},
                    {"Field": "Elevation range", "Value": f"{data['Elevation'].min():.0f}-{data['Elevation'].max():.0f} m"},
                    {"Field": "Hillshade range", "Value": f"{data['avg_hillshade'].min():.1f}-{data['avg_hillshade'].max():.1f}"},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("Data cleaning impact")
        st.plotly_chart(cleaning_chart(), use_container_width=True)


def render_model_tab() -> None:
    st.subheader("Reported model comparison")
    st.plotly_chart(model_comparison_chart(), use_container_width=True)
    st.dataframe(
        MODEL_RESULTS.assign(
            Accuracy=lambda frame: frame["Accuracy"].map(lambda value: f"{value:.6f}"),
            Precision_macro=lambda frame: frame["Precision_macro"].map(lambda value: f"{value:.6f}"),
            Recall_macro=lambda frame: frame["Recall_macro"].map(lambda value: f"{value:.6f}"),
            F1_macro=lambda frame: frame["F1_macro"].map(lambda value: f"{value:.6f}"),
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Best estimator")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("n_estimators", "200")
    c2.metric("max_depth", "20")
    c3.metric("min_samples_leaf", "10")
    c4.metric("max_features", "sqrt")
    st.info("Notebook result: Random Forest reached 61.72% accuracy and 66.28% macro recall, outperforming the logistic regression baseline by 11.03 and 7.48 percentage points.")


def render_resume_tab() -> None:
    st.subheader("Resume-ready summary")
    st.markdown(
        """
        **EcoCover: Forest Cover Type Prediction Dashboard**  
        Built an interactive machine learning dashboard for a seven-class forest cover prediction task using environmental features including elevation, hillshade, soil type and wilderness area.

        - Processed multi-source COMP5310 data with quality auditing, missing-value handling and feature engineering.
        - Converted 40 soil one-hot fields and 4 wilderness one-hot fields into interpretable categorical predictors.
        - Compared Logistic Regression, Decision Tree and Random Forest with class weighting and macro recall.
        - Achieved 61.72% test accuracy and 66.28% macro recall with the tuned Random Forest model.
        - Deployed an interactive Streamlit portfolio app with Plotly visual analytics.
        """
    )

    st.subheader("Deployment checklist")
    st.markdown(
        """
        1. Create a public GitHub repository, for example `ecocover-forest-prediction`.
        2. Put the files in this `ecocover-streamlit` folder at the repository root.
        3. On Streamlit Community Cloud, create a new app from that GitHub repo.
        4. Set the main file path to `app.py`.
        5. Deploy and use the generated `streamlit.app` URL on your resume.
        """
    )


def main() -> None:
    inject_css()
    default_data, source = load_default_data()

    uploaded_data = st.session_state.get("uploaded_data")
    data = uploaded_data if uploaded_data is not None and not uploaded_data.empty else default_data
    source = "uploaded" if uploaded_data is not None and not uploaded_data.empty else source

    current_row = sidebar_inputs(data)
    model, runtime_metrics = train_model(data)

    st.markdown(
        """
        <div class="hero">
            <h1>EcoCover</h1>
            <p>Interactive forest cover prediction dashboard built with Python, Pandas, scikit-learn, Streamlit and Plotly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cards(data, source)

    tab_predict, tab_data, tab_models, tab_resume = st.tabs(["Predictor", "Data story", "Model results", "Portfolio notes"])

    with tab_predict:
        render_predict_tab(data, model, runtime_metrics, current_row)
    with tab_data:
        render_data_tab(data)
    with tab_models:
        render_model_tab()
    with tab_resume:
        render_resume_tab()


if __name__ == "__main__":
    main()
