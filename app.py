
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

TEST_DATA_PATH = DATA_DIR / "test_FD001_official.csv"
METRICS_PATH = REPORT_DIR / "official_test_results.csv"

MODEL_FILES = {
    "Extra Trees": "extra_trees_fd001.joblib",
    "Random Forest": "random_forest_fd001.joblib",
}


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance | NASA FD001",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 5px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-title">⚙️ AI-Based Machine Fault Detection</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Remaining Useful Life (RUL) Prediction '
    '| NASA C-MAPSS FD001</p>',
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# Data and model loading
# --------------------------------------------------

@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_data
def load_feature_columns():
    with open(
        MODEL_DIR / "feature_columns.json",
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    return joblib.load(model_path)


@st.cache_data
def load_official_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return None


# --------------------------------------------------
# Check required files
# --------------------------------------------------

required_files = [
    TEST_DATA_PATH,
    MODEL_DIR / "feature_columns.json",
]

required_files.extend(
    MODEL_DIR / filename
    for filename in MODEL_FILES.values()
)

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error("Required project files are missing:")
    for file_path in missing_files:
        st.write(f"- {file_path}")
    st.stop()


# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------

try:
    test_data = load_test_data()
    feature_columns = load_feature_columns()
except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

required_columns = ["engine_id", "cycle", "RUL"] + feature_columns

missing_columns = [
    column
    for column in required_columns
    if column not in test_data.columns
]

if missing_columns:
    st.error(f"Dataset is missing columns: {missing_columns}")
    st.stop()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.title("Prediction Settings")
st.sidebar.caption("NASA C-MAPSS FD001")

engine_ids = sorted(test_data["engine_id"].unique())

selected_engine = st.sidebar.selectbox(
    "Select test engine",
    engine_ids,
    format_func=lambda engine_id: f"Engine {engine_id}",
)

st.sidebar.divider()
st.sidebar.info(
    "Both trained models will predict the selected engine's "
    "Remaining Useful Life."
)

st.sidebar.caption("Models: Extra Trees and Random Forest")


# --------------------------------------------------
# Get selected engine data
# --------------------------------------------------

engine_data = test_data[
    test_data["engine_id"] == selected_engine
].copy()

if engine_data.empty:
    st.error("No data found for the selected engine.")
    st.stop()

# One final observed row per engine in official FD001 test data.
engine_row = engine_data.iloc[[-1]]

X_input = engine_row[feature_columns]

actual_rul = float(engine_row["RUL"].iloc[0])
observed_cycle = int(engine_row["cycle"].iloc[0])


# --------------------------------------------------
# Predict using both saved models
# --------------------------------------------------

predictions = {}

try:
    for model_name in MODEL_FILES:
        model = load_model(model_name)

        predicted_rul = float(model.predict(X_input)[0])
        predicted_rul = max(0.0, predicted_rul)

        predictions[model_name] = {
            "prediction": predicted_rul,
            "error": abs(predicted_rul - actual_rul),
        }

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()


# --------------------------------------------------
# Engine overview
# --------------------------------------------------

st.header(f"Engine {selected_engine} — RUL Prediction")

overview1, overview2, overview3 = st.columns(3)

overview1.metric("Observed Cycle", f"{observed_cycle}")
overview2.metric("Actual RUL", f"{actual_rul:.0f} cycles")
overview3.metric("Input Features", f"{len(feature_columns)}")

st.caption(
    "Actual RUL is shown only for retrospective evaluation. "
    "It is not used as a model input."
)

st.divider()


# --------------------------------------------------
# Side-by-side model predictions
# --------------------------------------------------

st.header("Model Prediction Comparison")

extra_trees = predictions["Extra Trees"]
random_forest = predictions["Random Forest"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌳 Extra Trees")
    st.metric(
        "Predicted RUL",
        f"{extra_trees['prediction']:.2f} cycles",
    )
    st.metric(
        "Absolute Error",
        f"{extra_trees['error']:.2f} cycles",
    )

with col2:
    st.subheader("🌲 Random Forest")
    st.metric(
        "Predicted RUL",
        f"{random_forest['prediction']:.2f} cycles",
    )
    st.metric(
        "Absolute Error",
        f"{random_forest['error']:.2f} cycles",
    )

if extra_trees["error"] < random_forest["error"]:
    st.success(
        "Extra Trees has the lower absolute error "
        "for this selected engine."
    )
elif random_forest["error"] < extra_trees["error"]:
    st.success(
        "Random Forest has the lower absolute error "
        "for this selected engine."
    )
else:
    st.info("Both models have the same absolute error.")

if extra_trees["error"] < random_forest["error"]:
    st.success(
        "Extra Trees has the lower absolute error "
        "for this selected engine."
    )
elif random_forest["error"] < extra_trees["error"]:
    st.success(
        "Random Forest has the lower absolute error "
        "for this selected engine."
    )
else:
    st.info("Both models have the same absolute error.")



# --------------------------------------------------
# Download selected engine prediction report
# --------------------------------------------------

st.divider()
st.subheader("Download Prediction Report")

report_data = pd.DataFrame([{
    "Engine ID": selected_engine,
    "Observed Cycle": observed_cycle,
    "Actual RUL (cycles)": round(actual_rul, 2),
    "Extra Trees Predicted RUL": round(extra_trees["prediction"], 2),
    "Extra Trees Absolute Error": round(extra_trees["error"], 2),
    "Random Forest Predicted RUL": round(random_forest["prediction"], 2),
    "Random Forest Absolute Error": round(random_forest["error"], 2),
}])

st.download_button(
    label="Download CSV Prediction Report",
    data=report_data.to_csv(index=False),
    file_name=f"engine_{selected_engine}_prediction_report.csv",
    mime="text/csv",
    width="stretch",
)


# --------------------------------------------------
# Professional PDF prediction report
# --------------------------------------------------

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# Create PDF in memory
pdf_buffer = BytesIO()

pdf_doc = SimpleDocTemplate(
    pdf_buffer,
    pagesize=A4,
    rightMargin=20 * mm,
    leftMargin=20 * mm,
    topMargin=20 * mm,
    bottomMargin=25 * mm,
)

# --------------------------------------------------
# PDF colors and styles
# --------------------------------------------------

navy = colors.HexColor("#1e3a5f")
blue = colors.HexColor("#2563eb")
light_blue = colors.HexColor("#eff6ff")
light_gray = colors.HexColor("#f1f5f9")
dark_gray = colors.HexColor("#334155")
green = colors.HexColor("#dcfce7")

pdf_styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ProjectTitle",
    parent=pdf_styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=19,
    leading=24,
    alignment=TA_CENTER,
    textColor=navy,
    spaceAfter=8,
)

subtitle_style = ParagraphStyle(
    "ReportSubtitle",
    parent=pdf_styles["Normal"],
    fontSize=11,
    leading=15,
    alignment=TA_CENTER,
    textColor=dark_gray,
)

section_style = ParagraphStyle(
    "SectionHeading",
    parent=pdf_styles["Heading2"],
    fontSize=12,
    leading=16,
    textColor=navy,
    spaceBefore=8,
    spaceAfter=8,
)

body_style = ParagraphStyle(
    "ReportBody",
    parent=pdf_styles["Normal"],
    fontSize=9,
    leading=14,
    textColor=dark_gray,
)

cell_style = ParagraphStyle(
    "TableCell",
    parent=body_style,
    fontSize=8.5,
    leading=11,
)

header_style = ParagraphStyle(
    "TableHeader",
    parent=cell_style,
    fontName="Helvetica-Bold",
    textColor=colors.white,
)

# --------------------------------------------------
# Footer and page number
# --------------------------------------------------

def add_pdf_footer(canvas, doc):
    canvas.saveState()

    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.6)

    canvas.line(
        doc.leftMargin,
        18 * mm,
        A4[0] - doc.rightMargin,
        18 * mm,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(dark_gray)

    canvas.drawString(
        doc.leftMargin,
        12 * mm,
        "NASA C-MAPSS FD001 | Predictive Maintenance FYP",
    )

    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        12 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


# --------------------------------------------------
# Build PDF content
# --------------------------------------------------

pdf_story = []

pdf_story.append(
    Paragraph(
        "AI-Based Machine Fault Detection<br/>"
        "with Limited Failure Data",
        title_style,
    )
)

pdf_story.append(
    Paragraph(
        "Remaining Useful Life (RUL) Prediction Report",
        subtitle_style,
    )
)

pdf_story.append(Spacer(1, 15))

# Project information
pdf_story.append(
    Paragraph("PROJECT INFORMATION", section_style)
)

project_data = [
    [
        Paragraph("<b>Dataset</b>", cell_style),
        Paragraph("NASA C-MAPSS FD001", cell_style),
    ],
    [
        Paragraph("<b>Prediction Task</b>", cell_style),
        Paragraph("Remaining Useful Life (RUL) Regression", cell_style),
    ],
    [
        Paragraph("<b>Models</b>", cell_style),
        Paragraph("Extra Trees and Random Forest", cell_style),
    ],
    [
        Paragraph("<b>Report Generated</b>", cell_style),
        Paragraph(
            datetime.now().strftime("%d %B %Y, %H:%M"),
            cell_style,
        ),
    ],
]

project_table = Table(
    project_data,
    colWidths=[45 * mm, 125 * mm],
)

project_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), light_blue),
        ("BACKGROUND", (1, 0), (1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])
)

pdf_story.append(project_table)
pdf_story.append(Spacer(1, 15))

# Selected engine information
pdf_story.append(
    Paragraph("SELECTED ENGINE INFORMATION", section_style)
)

engine_data = [
    [
        Paragraph("<b>Field</b>", header_style),
        Paragraph("<b>Value</b>", header_style),
    ],
    [
        Paragraph("Engine ID", cell_style),
        Paragraph(str(selected_engine), cell_style),
    ],
    [
        Paragraph("Observed Cycle", cell_style),
        Paragraph(str(observed_cycle), cell_style),
    ],
    [
        Paragraph("Actual RUL", cell_style),
        Paragraph(f"{actual_rul:.2f} cycles", cell_style),
    ],
]

engine_table = Table(
    engine_data,
    colWidths=[85 * mm, 85 * mm],
)

engine_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("BACKGROUND", (0, 1), (-1, -1), light_gray),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ])
)

pdf_story.append(engine_table)
pdf_story.append(Spacer(1, 15))

# Model comparison
pdf_story.append(
    Paragraph("MODEL PREDICTION COMPARISON", section_style)
)

model_data = [
    [
        Paragraph("<b>Model</b>", header_style),
        Paragraph("<b>Predicted RUL</b>", header_style),
        Paragraph("<b>Absolute Error</b>", header_style),
    ],
    [
        Paragraph("Extra Trees", cell_style),
        Paragraph(f"{extra_trees['prediction']:.2f}", cell_style),
        Paragraph(f"{extra_trees['error']:.2f}", cell_style),
    ],
    [
        Paragraph("Random Forest", cell_style),
        Paragraph(f"{random_forest['prediction']:.2f}", cell_style),
        Paragraph(f"{random_forest['error']:.2f}", cell_style),
    ],
]

model_table = Table(
    model_data,
    colWidths=[55 * mm, 55 * mm, 60 * mm],
)

model_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("BACKGROUND", (0, 1), (-1, -1), light_gray),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ])
)

pdf_story.append(model_table)
pdf_story.append(Spacer(1, 15))

# Identify the lower-error model
if extra_trees["error"] < random_forest["error"]:
    better_model = "Extra Trees"
elif random_forest["error"] < extra_trees["error"]:
    better_model = "Random Forest"
else:
    better_model = "Both models have equal absolute error"

best_model_data = [[
    Paragraph(
        f"<b>Lower-error model for Engine {selected_engine}:</b> "
        f"{better_model}",
        body_style,
    )
]]

best_model_table = Table(
    best_model_data,
    colWidths=[170 * mm],
)

best_model_table.setStyle(
    TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), green),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#86b99a")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ])
)

pdf_story.append(best_model_table)
pdf_story.append(Spacer(1, 15))

# Evaluation note
pdf_story.append(
    Paragraph("EVALUATION NOTE", section_style)
)

pdf_story.append(
    Paragraph(
        "Actual RUL is included only for retrospective evaluation. "
        "It is not supplied as an input feature to either machine "
        "learning model. Both models generate their predictions "
        "using the engine's 24 operating-setting and sensor features.",
        body_style,
    )
)

# Generate PDF
pdf_doc.build(
    pdf_story,
    onFirstPage=add_pdf_footer,
    onLaterPages=add_pdf_footer,
)

pdf_buffer.seek(0)

# Download button
st.download_button(
    label="Download PDF Prediction Report",
    data=pdf_buffer.getvalue(),
    file_name=f"engine_{selected_engine}_prediction_report.pdf",
    mime="application/pdf",
    width="stretch",
)


# --------------------------------------------------
# Prediction chart
# --------------------------------------------------

st.divider()
st.header("Predicted RUL vs Actual RUL")

comparison_data = pd.DataFrame(
    {
        "Measurement": [
            "Extra Trees",
            "Random Forest",
            "Actual RUL",
        ],
        "RUL (cycles)": [
            extra_trees["prediction"],
            random_forest["prediction"],
            actual_rul,
        ],
    }
)

st.bar_chart(
    comparison_data.set_index("Measurement"),
    y="RUL (cycles)",
    width="stretch",
)


# --------------------------------------------------
# Official 100-engine model performance
# --------------------------------------------------

st.divider()
st.header("Official Test Performance")

official_metrics = load_official_metrics()

if official_metrics is not None:
    expected_metric_columns = [
        "model",
        "MAE",
        "RMSE",
        "NASA_score",
    ]

    missing_metric_columns = [
        column
        for column in expected_metric_columns
        if column not in official_metrics.columns
    ]

    if missing_metric_columns:
        st.warning(
            "The official metrics file is missing columns: "
            f"{missing_metric_columns}"
        )
    else:
        performance_rows = []

        for model_name in MODEL_FILES:
            selected_rows = official_metrics[
                official_metrics["model"] == model_name
            ]

            if not selected_rows.empty:
                row = selected_rows.iloc[0]

                performance_rows.append(
                    {
                        "Model": model_name,
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "NASA Score": row["NASA_score"],
                    }
                )

        if performance_rows:
            performance_table = pd.DataFrame(performance_rows)

            st.dataframe(
                performance_table,
                width="stretch",
                hide_index=True,
            )
            st.subheader("Official Model Performance Charts")

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.write("MAE Comparison — Lower is Better")
                st.bar_chart(
                    performance_table.set_index("Model")[["MAE"]],
                    width="stretch",
                )

            with chart_col2:
                st.write("RMSE Comparison — Lower is Better")
                st.bar_chart(
                    performance_table.set_index("Model")[["RMSE"]],
                    width="stretch",
                )

            st.write("NASA Score Comparison — Lower is Better")
            st.bar_chart(
                performance_table.set_index("Model")[["NASA Score"]],
                width="stretch",
            )

            st.caption(
                "These saved metrics represent the complete official "
                "100-engine test evaluation. They are not calculated "
                "from the selected engine alone."
            )
        else:
            st.warning("No official model metrics were found.")
else:
    st.warning(
        "Official evaluation metrics were not found. "
        "Predictions are still available."
    )


# --------------------------------------------------
# Sensor readings and operating settings
# --------------------------------------------------

st.divider()
st.header("Engine Sensor Readings and Operating Settings")

feature_table = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Value": engine_row[feature_columns].iloc[0].values,
    }
)

with st.expander("View all 24 input features"):
    st.dataframe(
        feature_table,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# Academic interpretation
# --------------------------------------------------

st.divider()

with st.expander("How to interpret this demonstration"):
    st.write(
        "The application selects an engine from the official "
        "NASA C-MAPSS FD001 test dataset."
    )

    st.write(
        "Both pre-trained models use the same 24 operating-setting "
        "and sensor features to estimate Remaining Useful Life."
    )

    st.write(
        "The official test dataset provides one final observed row "
        "for each of its 100 test engines."
    )

    st.write(
        "Actual RUL is displayed only for retrospective comparison. "
        "It is not passed to either model as an input."
    )

    st.write(
        "The application loads saved models for inference. "
        "It does not train or modify any model."
    )

st.caption(
    "Academic demonstration | NASA C-MAPSS FD001 | "
    "Random Forest and Extra Trees"
)

# --------------------------------------------------
# Project overview and methodology
# --------------------------------------------------

st.divider()

st.header("Project Overview and Methodology")

with st.expander("Project Overview", expanded=True):
    st.write(
        "This project investigates Remaining Useful Life (RUL) "
        "prediction for aircraft engines using the NASA C-MAPSS FD001 dataset."
    )

    st.write(
        "The main objective is to study how machine learning models "
        "perform when only a limited number of training engines are available."
    )

    st.write(
        "The project uses 24 operating-setting and sensor features "
        "to estimate the remaining useful life of an engine."
    )

with st.expander("Experimental Methodology"):
    st.markdown("""
    **Dataset:** NASA C-MAPSS FD001

    **Prediction task:** Remaining Useful Life (RUL) regression

    **Machine learning models:**
    - Random Forest
    - Extra Trees

    **Limited-data experiment sizes:**
    - 20 training engines
    - 40 training engines
    - 60 training engines
    - 80 training engines

    **Evaluation metrics:**
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - NASA scoring function
    """)

    st.write(
        "For the official test evaluation, both saved models were "
        "trained using all 100 training engines and evaluated on "
        "the 100 official test engines."
    )

with st.expander("Prediction Workflow"):
    st.markdown("""
    1. Load the processed NASA C-MAPSS FD001 test data.
    2. Select an engine.
    3. Extract its 24 input features.
    4. Generate RUL predictions using both saved models.
    5. Compare predictions with actual RUL for retrospective evaluation.
    6. Review the official performance metrics.
    """)

    st.info(
        "This dashboard performs inference using saved models. "
        "It does not retrain the models."
    )


# --------------------------------------------------
# System architecture and prediction pipeline
# --------------------------------------------------

st.divider()

st.header("System Architecture and Prediction Pipeline")

st.write(
    "The following diagram shows how engine sensor readings "
    "pass through the saved machine learning models to generate "
    "Remaining Useful Life predictions."
)

pipeline_graph = """
digraph PredictiveMaintenance {
    graph [rankdir=TB, bgcolor="transparent", nodesep=0.45, ranksep=0.65]
    node [shape=box, style="rounded,filled", fontname="Arial",
          fontsize=11, color="#64748b", fillcolor="#e2e8f0",
          fontcolor="#0f172a", margin="0.18,0.12"]
    edge [color="#64748b", penwidth=1.5, arrowsize=0.8]

    A [label="NASA C-MAPSS FD001\\nProcessed Test Data"]
    B [label="Select Engine\\n24 Input Features"]
    C [label="Saved Random Forest Model\\nPre-trained"]
    D [label="Saved Extra Trees Model\\nPre-trained"]
    E [label="Random Forest\\nRUL Prediction"]
    F [label="Extra Trees\\nRUL Prediction"]
    G [label="Compare Predictions\\nand Model Performance"]
    H [label="Actual RUL\\nEvaluation Only", fillcolor="#fef3c7"]
    I [label="Results Dashboard\\nPredicted RUL, Errors and Charts",
       fillcolor="#dcfce7"]

    A -> B
    B -> C
    B -> D
    C -> E
    D -> F
    E -> G
    F -> G
    H -> G [style=dashed, color="#d97706", label="Retrospective comparison"]
    G -> I
}
"""

st.graphviz_chart(pipeline_graph, width="stretch")

st.caption(
    "Note: Actual RUL is used only for retrospective evaluation. "
    "It is not supplied to either model as an input feature."
)
