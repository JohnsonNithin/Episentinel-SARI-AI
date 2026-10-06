# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# st.set_page_config(
#     page_title="EpiSentinel",
#     layout="wide"
# )

# st.title("EpiSentinel")
# st.caption(
#     "AI-driven prototype for SARI risk prediction "
#     "and respiratory surveillance"
# )

# tab1, tab2 = st.tabs([
#     "Patient Risk",
#     "Surveillance"
# ])


# # --------------------------------------------------
# # TAB 1 — PATIENT RISK
# # --------------------------------------------------

# with tab1:

#     st.header("Patient-level severe outcome prediction")

#     st.write(
#         "Models predict mortality as a proxy "
#         "for severe SARI outcome."
#     )

#     results = pd.DataFrame({
#         "Model": [
#             "Clinical baseline",
#             "Clinical + comorbidities",
#             "Clinical only (X-ray cohort)",
#             "Clinical + radiology"
#         ],

#         "AUROC": [
#             0.851,
#             0.857,
#             0.784,
#             0.801
#         ],

#         "AUPRC": [
#             0.318,
#             0.329,
#             0.222,
#             0.206
#         ]
#     })

#     st.dataframe(
#         results,
#         use_container_width=True
#     )

#     col1, col2, col3 = st.columns(3)

#     col1.metric(
#         "Best Clinical AUROC",
#         "0.857"
#     )

#     col2.metric(
#         "Radiology Cohort AUROC",
#         "0.801",
#         "+0.017 vs clinical-only"
#     )

#     col3.metric(
#         "Clinical Recall",
#         "87.6%"
#     )

#     st.info(
#         "Structured chest X-ray findings showed "
#         "incremental discrimination, although improvements "
#         "were not consistent across every metric."
#     )


# # --------------------------------------------------
# # TAB 2 — SURVEILLANCE
# # --------------------------------------------------

# with tab2:

#     st.header("Population-level SARI surveillance")

#     weekly_cases = pd.read_csv(
#         "outputs/weekly_surveillance.csv"
#     )

#     weekly_cases["week"] = pd.to_datetime(
#         weekly_cases["week"]
#     )

#     fig, ax = plt.subplots(figsize=(12, 5))

#     ax.plot(
#         weekly_cases["week"],
#         weekly_cases["cases"],
#         label="Weekly SARI cases"
#     )

#     ax.plot(
#         weekly_cases["week"],
#         weekly_cases["baseline_mean"],
#         linestyle="--",
#         label="4-week baseline"
#     )

#     alerts = weekly_cases[
#         weekly_cases["alert"] == True
#     ]

#     ax.scatter(
#         alerts["week"],
#         alerts["cases"],
#         s=80,
#         label="Anomaly alert"
#     )

#     ax.set_title(
#         "Weekly SARI Surveillance"
#     )

#     ax.set_xlabel("Week")
#     ax.set_ylabel("Number of cases")

#     ax.legend()

#     st.pyplot(fig)

#     st.subheader("Detected anomaly weeks")

#     st.dataframe(
#         alerts[
#             [
#                 "week",
#                 "cases",
#                 "baseline_mean",
#                 "z_score"
#             ]
#         ],
#         use_container_width=True
#     )


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EpiSentinel",
    page_icon="🫁",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🫁 EpiSentinel")

st.subheader(
    "AI-driven SARI risk prediction and respiratory surveillance"
)

st.write(
    """
    A proof-of-concept system using real SIVEP-Gripe SARI data
    to combine patient-level severe-outcome prediction with
    population-level respiratory surveillance.
    """
)


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

model_results = pd.read_csv(
    "outputs/model_results.csv"
)

weekly_cases = pd.read_csv(
    "outputs/weekly_surveillance.csv"
)

weekly_cases["week"] = pd.to_datetime(
    weekly_cases["week"]
)


# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------

overview_tab, risk_tab, surveillance_tab = st.tabs(
    [
        "Overview",
        "Patient Risk",
        "Surveillance"
    ]
)


# ==================================================
# TAB 1 — OVERVIEW
# ==================================================

with overview_tab:

    st.header("Project Overview")

    st.markdown(
        """
        ### Problem

        Severe Acute Respiratory Infection (SARI)
        surveillance requires information at two levels:

        **Patient level**
        - Which patients are at higher risk of severe outcome?

        **Population level**
        - Is respiratory disease activity becoming unusually high?

        ### Prototype

        **Clinical data + comorbidities + structured radiology**
        → mortality-risk prediction

        **Weekly SARI case counts**
        → trend monitoring and anomaly detection
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "SARI records",
        "240,435"
    )

    col2.metric(
        "Best Clinical AUROC",
        "0.857"
    )

    col3.metric(
        "Clinical Recall",
        "87.6%"
    )

    st.info(
        """
        Mortality was used as a proof-of-concept proxy
        for severe SARI outcome.
        """
    )


# ==================================================
# TAB 2 — PATIENT RISK
# ==================================================

with risk_tab:

    st.header(
        "Patient-level Severe Outcome Prediction"
    )

    st.write(
        """
        The objective was to determine whether increasingly
        rich patient information improves mortality prediction.
        """
    )

    st.markdown(
        """
        **Model progression**

        Clinical symptoms  
        ↓  
        + Comorbidities  
        ↓  
        + Structured chest X-ray findings
        """
    )

    st.dataframe(
        model_results,
        use_container_width=True
    )


    # -----------------------------
    # CLINICAL MODEL
    # -----------------------------

    st.subheader(
        "1. Clinical Model"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AUROC", "0.857")
    col2.metric("AUPRC", "0.329")
    col3.metric("Recall", "87.6%")
    col4.metric("Precision", "22.6%")

    st.write(
        """
        Adding patient comorbidities produced a small but
        consistent improvement over symptoms and demographics
        alone.
        """
    )


    # -----------------------------
    # MULTIMODAL EXPERIMENT
    # -----------------------------

    st.subheader(
        "2. Clinical + Radiology"
    )

    st.write(
        """
        To evaluate the incremental value of radiology,
        clinical-only and clinical+radiology models were
        compared using the same radiology-eligible patients.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Clinical only"
        )

        st.metric(
            "AUROC",
            "0.784"
        )

        st.metric(
            "AUPRC",
            "0.222"
        )

        st.metric(
            "Recall",
            "60.0%"
        )


    with col2:

        st.markdown(
            "#### Clinical + Radiology"
        )

        st.metric(
            "AUROC",
            "0.801",
            "+0.017"
        )

        st.metric(
            "AUPRC",
            "0.206"
        )

        st.metric(
            "Recall",
            "54.3%"
        )


    st.info(
        """
        Structured radiological findings increased AUROC
        from 0.784 to 0.801 and improved precision,
        although AUPRC and recall decreased.

        This suggests additional radiological signal,
        but not a uniform improvement across all metrics.
        """
    )


# ==================================================
# TAB 3 — SURVEILLANCE
# ==================================================

with surveillance_tab:

    st.header(
        "Population-level SARI Surveillance"
    )

    st.write(
        """
        Patient symptom-onset dates were aggregated by week
        to monitor respiratory disease activity over time.
        """
    )


    # -----------------------------
    # SURVEILLANCE FIGURE
    # -----------------------------

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        weekly_cases["week"],
        weekly_cases["cases"],
        label="Weekly SARI cases"
    )

    ax.plot(
        weekly_cases["week"],
        weekly_cases["baseline_mean"],
        linestyle="--",
        label="4-week baseline"
    )

    alerts = weekly_cases[
        weekly_cases["alert"] == True
    ]

    ax.scatter(
        alerts["week"],
        alerts["cases"],
        s=80,
        label="Anomaly alert"
    )

    ax.set_title(
        "Weekly SARI Surveillance"
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Number of cases")

    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


    # -----------------------------
    # ALERTS
    # -----------------------------

    st.subheader(
        "Detected Anomaly Weeks"
    )

    st.dataframe(
        alerts[
            [
                "week",
                "cases",
                "baseline_mean",
                "z_score"
            ]
        ].round(2),
        use_container_width=True
    )

    st.write(
        """
        An alert is generated when the current weekly
        SARI count is more than 2 standard deviations
        above the previous four-week baseline.
        """
    )


# --------------------------------------------------
# METHODOLOGY / LIMITATIONS
# --------------------------------------------------

st.divider()

with st.expander(
    "Methodology and limitations"
):

    st.markdown(
        """
        **Data**
        - Real SIVEP-Gripe SARI surveillance records.
        - Mortality used as the severe-outcome proxy.

        **Patient-level model**
        - Demographics
        - Respiratory symptoms
        - Comorbidities
        - Structured chest X-ray findings
        - XGBoost classifier

        **Radiology**
        - Radiology represents structured chest X-ray findings.
        - Raw radiograph images were not available.

        **Surveillance**
        - Cases aggregated using symptom-onset date.
        - Previous four weeks form the expected baseline.
        - Z-score > 2 generates an anomaly alert.

        **Limitations**
        - Prototype, not a clinical decision-support system.
        - Radiology uses structured findings rather than raw images.
        - The anomaly detector does not explicitly model seasonality.
        - Longer historical data would be required for robust
          outbreak validation.
        """
    )