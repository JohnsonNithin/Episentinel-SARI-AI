                    SARI PATIENT DATA
                           │
             ┌─────────────┴──────────────┐
             │                            │
      CLINICAL FEATURES            RADIOLOGY FEATURES
      age, symptoms, etc.          X-ray / CT result
             │                            │
             └─────────────┬──────────────┘
                           ↓
                 MULTIMODAL FUSION
                           ↓
                 SEVERITY PREDICTION
                           ↓
                   probability of death
                           │
             ┌─────────────┴──────────────┐
             ↓                            ↓
         SHAP / WHY?               PATIENT REPRESENTATION
                                          ↓
                                    CLUSTERING
                                          ↓
                                   SARI ENDOTYPES


All patients over time
        ↓
Aggregate weekly
        ↓
Cases + mortality + predicted high-risk %
        ↓
Trend / anomaly detection
        ↓
⚠ EARLY-WARNING SIGNAL