from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="EpiSentinel",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"


# ==================================================
# LIGHT, CLEAN UI STYLING
# ==================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #F8FAFC;
            color: #0F172A;
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] * {
            color: #0F172A;
        }

        h1, h2, h3 {
            color: #0F172A;
        }

        .hero {
            background: linear-gradient(135deg, #E0F2FE 0%, #EEF2FF 100%);
            border: 1px solid #D8EAFE;
            border-radius: 22px;
            padding: 28px 30px;
            margin-bottom: 20px;
        }

        .hero h1 {
            margin: 0;
            color: #0F172A;
            font-size: 2.2rem;
        }

        .hero p {
            margin-top: 10px;
            margin-bottom: 0;
            color: #334155;
            font-size: 1.05rem;
        }

        .card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
            margin-bottom: 16px;
        }

        .subtle {
            color: #64748B;
            font-size: 0.95rem;
        }

        .risk-box {
            background: #FFFFFF;
            border: 1px solid #DBEAFE;
            border-radius: 18px;
            padding: 22px;
            margin-top: 16px;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: #2563EB;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.2rem;
            font-weight: 700;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #1D4ED8;
            color: white;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            padding: 14px 16px;
            border-radius: 14px;
        }

        .footer-note {
            margin-top: 22px;
            padding: 14px 16px;
            background: #FFF7ED;
            border: 1px solid #FED7AA;
            border-radius: 12px;
            color: #7C2D12;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HELPERS
# ==================================================

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path)


def safe_load_csv(path: Path):
    return load_csv(path) if path.exists() else None


def patient_dataframe(
    age,
    sex,
    fever,
    cough,
    dyspnoea,
    respiratory_distress,
    low_oxygen,
    cardiopathy,
    diabetes,
    lung_disease,
    renal,
    immunodepression,
    asthma,
    neurological,
    obesity,
):
    binary_map = {
        "Yes": 1,
        "No": 0,
        "Unknown": np.nan,
    }

    return pd.DataFrame(
        [
            {
                "age_years": age,
                "CS_SEXO": sex,
                "FEBRE": binary_map[fever],
                "TOSSE": binary_map[cough],
                "DISPNEIA": binary_map[dyspnoea],
                "DESC_RESP": binary_map[respiratory_distress],
                "SATURACAO": binary_map[low_oxygen],
                "CARDIOPATI": binary_map[cardiopathy],
                "DIABETES": binary_map[diabetes],
                "PNEUMOPATI": binary_map[lung_disease],
                "RENAL": binary_map[renal],
                "IMUNODEPRE": binary_map[immunodepression],
                "ASMA": binary_map[asthma],
                "NEUROLOGIC": binary_map[neurological],
                "OBESIDADE": binary_map[obesity],
            }
        ]
    )


# ==================================================
# LOAD FILES
# ==================================================

clinical_model_path = MODEL_DIR / "clinical_model.joblib"
multimodal_model_path = MODEL_DIR / "multimodal_model.joblib"

model_results_path = OUTPUT_DIR / "model_results.csv"
weekly_surveillance_path = OUTPUT_DIR / "weekly_surveillance.csv"

clinical_model = (
    load_model(clinical_model_path)
    if clinical_model_path.exists()
    else None
)

multimodal_model = (
    load_model(multimodal_model_path)
    if multimodal_model_path.exists()
    else None
)

model_results = safe_load_csv(model_results_path)
weekly_cases = safe_load_csv(weekly_surveillance_path)

if weekly_cases is not None and "week" in weekly_cases.columns:
    weekly_cases["week"] = pd.to_datetime(
        weekly_cases["week"],
        errors="coerce",
    )


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.markdown("## 🫁 EpiSentinel")
    st.caption("Severe Acute Respiratory Infection (SARI) AI Surveillance Prototype")

    page = st.radio(
        "Navigate",
        [
            "Live Prediction",
            "Model Results",
            "Surveillance",
            "About",
        ],
        index=0,
    )

    st.divider()

    st.markdown("### Prototype status")

    if clinical_model is not None:
        st.success("Clinical model loaded")
    else:
        st.error("Clinical model missing")

    if multimodal_model is not None:
        st.success("Radiology model loaded")
    else:
        st.info("Radiology model optional")

    st.caption(
        "Built as a research proof-of-concept for AI-driven "
        "respiratory surveillance."
    )


# ==================================================
# PAGE 1 — LIVE PREDICTION
# ==================================================

if page == "Live Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🫁 Live Patient Risk Prediction</h1>
            <p>
                Enter a Severe Acute Respiratory Infection (SARI) patient profile and run the trained AI model
                to generate a model-estimated mortality risk score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if clinical_model is None:
        st.error(
            "The trained clinical model is missing. "
            "Save it as models/clinical_model.joblib."
        )
        st.stop()

    available_models = ["Clinical + Comorbidities"]

    if multimodal_model is not None:
        available_models.append(
            "Clinical + Comorbidities + Radiology"
        )

    model_mode = st.radio(
        "Choose prediction model",
        available_models,
        horizontal=True,
    )

    st.markdown(
        """
        <div class="subtle">
        Change the inputs and press <b>Predict Risk</b>.
        The app sends the values to the trained AI Model pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.form("patient_prediction_form"):

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("### Patient details")

            age = st.slider(
                "Age",
                min_value=0,
                max_value=100,
                value=65,
            )

            sex = st.selectbox(
                "Sex",
                ["M", "F"],
            )

            st.markdown("### Respiratory symptoms")

            fever = st.selectbox(
                "Fever",
                ["Yes", "No", "Unknown"],
            )

            cough = st.selectbox(
                "Cough",
                ["Yes", "No", "Unknown"],
            )

            dyspnoea = st.selectbox(
                "Dyspnoea / shortness of breath",
                ["Yes", "No", "Unknown"],
            )

            respiratory_distress = st.selectbox(
                "Respiratory distress",
                ["Yes", "No", "Unknown"],
            )

            low_oxygen = st.selectbox(
                "Oxygen saturation <95%",
                ["Yes", "No", "Unknown"],
            )

        with right:
            st.markdown("### Comorbidities")

            cardiopathy = st.selectbox(
                "Cardiovascular disease",
                ["Yes", "No", "Unknown"],
            )

            diabetes = st.selectbox(
                "Diabetes",
                ["Yes", "No", "Unknown"],
            )

            lung_disease = st.selectbox(
                "Chronic lung disease",
                ["Yes", "No", "Unknown"],
            )

            renal = st.selectbox(
                "Renal disease",
                ["Yes", "No", "Unknown"],
            )

            immunodepression = st.selectbox(
                "Immunodepression",
                ["Yes", "No", "Unknown"],
            )

            asthma = st.selectbox(
                "Asthma",
                ["Yes", "No", "Unknown"],
            )

            neurological = st.selectbox(
                "Neurological disease",
                ["Yes", "No", "Unknown"],
            )

            obesity = st.selectbox(
                "Obesity",
                ["Yes", "No", "Unknown"],
            )

            xray_result = None

            if model_mode == "Clinical + Comorbidities + Radiology":
                st.markdown("### Radiology")

                xray_result = st.selectbox(
                    "Chest X-ray finding",
                    [
                        "Normal",
                        "Interstitial infiltrate",
                        "Consolidation",
                        "Mixed",
                        "Other",
                    ],
                )

        submitted = st.form_submit_button(
            "Predict Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        patient = patient_dataframe(
            age=age,
            sex=sex,
            fever=fever,
            cough=cough,
            dyspnoea=dyspnoea,
            respiratory_distress=respiratory_distress,
            low_oxygen=low_oxygen,
            cardiopathy=cardiopathy,
            diabetes=diabetes,
            lung_disease=lung_disease,
            renal=renal,
            immunodepression=immunodepression,
            asthma=asthma,
            neurological=neurological,
            obesity=obesity,
        )

        if model_mode == "Clinical + Comorbidities":

            risk_score = clinical_model.predict_proba(
                patient
            )[0, 1]

            model_name = "Clinical + Comorbidities"

        else:
            xray_map = {
                "Normal": 1,
                "Interstitial infiltrate": 2,
                "Consolidation": 3,
                "Mixed": 4,
                "Other": 5,
            }

            patient["RAIOX_RES"] = xray_map[xray_result]

            risk_score = multimodal_model.predict_proba(
                patient
            )[0, 1]

            model_name = "Clinical + Comorbidities + Radiology"

        st.markdown(
            '<div class="risk-box">',
            unsafe_allow_html=True,
        )

        st.success("Prediction completed")

        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(
                "Model-estimated risk score",
                f"{risk_score * 100:.1f}%",
            )

        with result_col2:
            st.markdown(f"**Model used:** {model_name}")
            st.progress(
                min(max(float(risk_score), 0.0), 1.0)
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="footer-note">
            <b>Research prototype only.</b>
            This score is not a clinically validated probability
            and must not be used for diagnosis, treatment, or
            medical decision-making.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================
# PAGE 2 — MODEL RESULTS
# ==================================================

elif page == "Model Results":

    st.markdown(
        """
        <div class="hero">
            <h1>📊 Model Results</h1>
            <p>
                Summary of the patient-level modelling experiments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Clinical model — what information is used?")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AUROC", "0.857")
    col2.metric("AUPRC", "0.329")
    col3.metric("Recall", "87.6%")
    col4.metric("Precision", "22.6%")

    st.write(
        """
        This model uses **age, sex, respiratory symptoms and comorbidities**
        to predict the patient's mortality outcome: **recovered (0) or died (1)**.

        Adding comorbidities produced a small but consistent improvement
        over using demographics and symptoms alone.
        """
    )

    st.divider()

    st.subheader("Does adding chest X-ray information help?")

    st.write(
        """
        We trained two models on the **same patients** so the comparison is fair.

        **Target we want to predict:**  
        whether the patient **recovered (0)** or **died (1)**.

        This is a patient-level mortality **outcome**, not a population mortality rate.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model A — Clinical features only")
        st.write(
            """
            **Features used**
            - Age
            - Sex
            - Fever
            - Cough
            - Shortness of breath (dyspnoea)
            - Respiratory distress
            - Oxygen saturation <95%
            - Cardiovascular disease
            - Diabetes
            - Chronic lung disease
            - Renal disease
            - Immunodepression
            - Asthma
            - Neurological disease
            - Obesity

            **Prediction:** Recovered or died
            """
        )
        st.metric("AUROC", "0.784")
        st.metric("AUPRC", "0.222")
        st.metric("Recall", "60.0%")
        st.metric("Precision", "18.8%")

    with col2:
        st.markdown("### Model B — Clinical + Radiology features")
        st.write(
            """
            **Features used**

            All the same clinical features from Model A, **plus**:

            - Chest X-ray finding:
              - Normal
              - Interstitial infiltrate
              - Consolidation
              - Mixed
              - Other

            **Prediction:** Recovered or died
            """
        )
        st.metric("AUROC", "0.801", "+0.017")
        st.metric("AUPRC", "0.206")
        st.metric("Recall", "54.3%")
        st.metric("Precision", "21.1%")

    st.info(
        """
        **Simple interpretation:**  
        Model B receives one extra type of information — the chest X-ray result.
        Its AUROC increased from **0.784 to 0.801**, which means the radiology
        information added some predictive signal. However, not every metric improved.
        """
    )

    if model_results is not None:
        st.divider()
        st.subheader("All model runs")
        st.dataframe(
            model_results,
            use_container_width=True,
        )


# ==================================================
# PAGE 3 — SURVEILLANCE
# ==================================================

elif page == "Surveillance":

    st.markdown(
        """
        <div class="hero">
            <h1>📈 Severe Acute Respiratory Infection (SARI) Surveillance</h1>
            <p>
                Weekly respiratory activity monitoring with
                anomaly detection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if weekly_cases is None:
        st.warning(
            "outputs/weekly_surveillance.csv was not found."
        )

    else:
        required_columns = {
            "week",
            "cases",
            "baseline_mean",
            "alert",
            "z_score",
        }

        missing = required_columns.difference(
            weekly_cases.columns
        )

        if missing:
            st.error(
                "weekly_surveillance.csv is missing: "
                + ", ".join(sorted(missing))
            )

        else:
            alerts = weekly_cases[
                weekly_cases["alert"] == True
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Weeks monitored",
                str(len(weekly_cases)),
            )

            col2.metric(
                "Anomaly weeks",
                str(len(alerts)),
            )

            if len(weekly_cases) > 0:
                latest_cases = int(
                    weekly_cases["cases"].iloc[-1]
                )
                col3.metric(
                    "Latest weekly cases",
                    f"{latest_cases:,}",
                )

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )

            ax.plot(
                weekly_cases["week"],
                weekly_cases["cases"],
                label="Weekly SARI cases",
                linewidth=2,
            )

            ax.plot(
                weekly_cases["week"],
                weekly_cases["baseline_mean"],
                linestyle="--",
                label="4-week baseline",
                linewidth=2,
            )

            ax.scatter(
                alerts["week"],
                alerts["cases"],
                s=75,
                label="Anomaly alert",
            )

            ax.set_title(
                "Weekly Severe Acute Respiratory Infection (SARI) Surveillance"
            )

            ax.set_xlabel("Week")
            ax.set_ylabel("Number of cases")
            ax.legend(frameon=False)

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            st.subheader("Detected anomaly weeks")

            st.dataframe(
                alerts[
                    [
                        "week",
                        "cases",
                        "baseline_mean",
                        "z_score",
                    ]
                ].round(2),
                use_container_width=True,
            )

            st.caption(
                """
                An alert is generated when the current weekly
                Severe Acute Respiratory Infection (SARI) count exceeds the previous four-week baseline
                by more than two standard deviations.
                """
            )


# ==================================================
# PAGE 4 — ABOUT
# ==================================================

elif page == "About":

    st.markdown(
        """
        <div class="hero">
            <h1>ℹ️ About EpiSentinel</h1>
            <p>
                A compact proof-of-concept for AI-driven respiratory
                infection surveillance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### What this prototype demonstrates

        **Patient-level AI**
        - Demographics
        - Respiratory symptoms
        - Comorbidities
        - Structured chest X-ray findings
        - XGBoost classification

        **Population-level surveillance**
        - Weekly Severe Acute Respiratory Infection (SARI) case aggregation
        - Four-week rolling baseline
        - Simple anomaly alerts

        ### Data

        SIVEP-Gripe SARI surveillance data were used for the
        proof-of-concept.

        ### Key limitations

        - Mortality is used as a proxy for severe outcome.
        - Radiology uses structured chest X-ray findings rather than raw images.
        - Risk scores are research outputs and are not clinically validated.
        - This application is a research prototype and not a clinical tool.
        """
    )
