# Seasonal Agriculture Performance Analysis
**VOIS AICTE Emerging Technologies Internship — Major Project (Batch 1, 2026–2027)**  
*Author:* **Dhanish Ladwani**  
*Repository:* [https://github.com/dhanish0711/Seasonal-Agriculture-Performance-Analysis.git](https://github.com/dhanish0711/Seasonal-Agriculture-Performance-Analysis.git)  

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-2ea44f.svg?style=for-the-badge)]()

---

## 📌 Project Overview
This major project conducts an end-to-end data analytics and machine learning investigation into **4,000 multi-regional agricultural operations** across **8 Indian States** (*Punjab, Maharashtra, Tamil Nadu, Karnataka, Gujarat, Andhra Pradesh, Telangana, Madhya Pradesh*) and **10 Districts**, covering **8 major crop varieties** (*Rice, Wheat, Maize, Cotton, Pulses, Groundnut, Chilli, Sugarcane*) observed across **Kharif (Monsoon)**, **Rabi (Winter)**, and **Zaid (Summer)** cropping cycles.

### The Problem
Agricultural activities are intensely vulnerable to seasonal variations in environmental conditions, water availability, farming practices, and market prices. Raw farm data records inputs and outputs but does not explain *how*, *why*, and *by what magnitude* agricultural performance diverges across seasons.

This project decrypts the macro and micro drivers of seasonal performance divergence, establishes statistical significance through parametric and non-parametric tests, builds high-accuracy machine learning predictors, and formulates evidence-based planning models for stakeholders.

---

## 🏗️ System & Solution Architecture

```mermaid
flowchart TD
    subgraph S1["1. Data Ingestion & Quality Control"]
        RawData["Raw Dataset (4,000 Records, 28 Features)"] --> Audit["Data Schema Audit & Null Profiling"]
        Audit --> ReconYield["Mathematical Yield Reconstruction: Yield = Production / Area"]
        Audit --> StratImpute["Stratified Group Imputation: Rainfall & Soil Moisture"]
    end

    subgraph S2["2. Feature Engineering Pipeline"]
        ReconYield --> CleanData["Cleaned Dataset (0 Null Values)"]
        StratImpute --> CleanData
        CleanData --> Metrics["Domain Features: Profit Margin %, Cost/Ha, Water Productivity (INR/m³), Profit Tiers"]
    end

    subgraph S3["3. Statistical Inference & Machine Learning"]
        Metrics --> StatsTests["Statistical Hypothesis Testing: One-Way ANOVA, Intra-Crop Tests, Kruskal-Wallis, Welch t-test"]
        Metrics --> MLModels["Supervised ML: Random Forest Regressors (Yield R²=0.962, Profit R²=0.792)"]
        Metrics --> Unsupervised["Unsupervised Typologies: K-Means Clustering (3 Typologies) & 2D PCA"]
    end

    subgraph S4["4. User Interfaces & Submission Deliverables"]
        StatsTests --> StreamlitApp["Interactive Streamlit Web App (app.py): Live Predictor & Simulator"]
        MLModels --> StreamlitApp
        Unsupervised --> StreamlitApp
        StatsTests --> HTMLDash["Standalone Interactive HTML Dashboard (dashboard.html)"]
        StatsTests --> Notebook["Executed Jupyter Notebook (Seasonal_Agriculture_Performance_Analysis.ipynb)"]
        MLModels --> Notebook
        Unsupervised --> Notebook
        StatsTests --> PPTDeck["14-Slide Presentation Deck for VOIS AICTE"]
    end

    style S1 fill:#f0f4f8,stroke:#1B365D,stroke-width:2px
    style S2 fill:#eefaf0,stroke:#27ae60,stroke-width:2px
    style S3 fill:#fef6ec,stroke:#f39c12,stroke-width:2px
    style S4 fill:#f5f0fb,stroke:#8e44ad,stroke-width:2px
```

---

## 🌟 Key Discoveries & Highlights

1. **Macro Seasonal Divergence:**
   - **Kharif (Monsoon):** Highest average yield (5.64 t/ha) and net profit (**+₹1,78,915**), but suffers peak pest/disease risk (**54.5%**) due to high rainfall (852 mm) and humidity (71.8%).
   - **Rabi (Winter):** Most stable operating cycle with the lowest production costs (**₹5,13,837**) and consistent profitability (**+₹87,689**).
   - **Zaid (Summer):** Severe economic distress with highest water consumption (**6,420 m³**) and highest cost (**₹5,43,977**), resulting in a net negative average return (**-₹24,805**) and a 54.4% farm failure rate.
2. **The Staple Food Grain Paradox:**
   - Commercial cash crops (**Sugarcane** and **Chilli**) sustain massive profitability (+₹4.5L to +₹10.0L) across all three seasons.
   - In contrast, staple food grains (**Rice, Wheat, Maize**) suffer structural operational losses across all cycles due to high operating costs (~₹5.2 Lakhs) relative to farm-gate prices.
3. **The Micro-Irrigation Summer Shield:**
   - In Zaid, conventional **Flood irrigation** incurs massive deficits (**-₹69,787**), while precision **Drip (+₹21,292)** and **Sprinkler (+₹57,909)** irrigation remain profitable while saving over 2,200 m³/ha in water.
4. **Machine Learning Model Accuracy:**
   - **Crop Yield Predictor (Random Forest):** $R^2 = 0.962$, $\text{RMSE} = 2.716\text{ t/ha}$, $\text{MAE} = 0.763\text{ t/ha}$.
   - **Net Profit Predictor (Random Forest):** $R^2 = 0.792$, $\text{RMSE} = ₹2,41,398$, $\text{MAE} = ₹1,54,577$.
   - **Dominant Profit Drivers:** Soil pH (22.6%), Crop Choice (Sugarcane/Cotton/Chilli), Farm Area (11.5%), and Water Volume (5.9%).
5. **Statistical Verification:**
   - One-Way ANOVA across Seasons for Net Profit: $F = 34.292, \; p = 1.71 \times 10^{-15}$.
   - Intra-Crop Yield ANOVA: **Every single crop variety exhibits statistically significant seasonal yield variation ($p < 0.0001$)**.

---

## 📂 Repository Structure

```
├── Seasonal_Agriculture_Performance_Analysis.ipynb  # Fully executed publication-grade Jupyter Notebook (24 cells)
├── app.py                                           # Interactive Streamlit Web Application & Live ML Predictor
├── dashboard.html                                   # Standalone interactive HTML dashboard (Open directly in browser)
├── PROJECT_REPORT.md                                # Comprehensive 9-section technical research report
├── VOIS_Major_Project_PPT_Submission_Template.pptx  # Populated 14-slide presentation deck (Template name)
├── VOIS_Major_Project_PPT_Final_Submission.pptx     # Populated 14-slide presentation deck (Submission name)
├── seasonal_agriculture_performance_dataset.csv     # Raw dataset (4,000 records, 28 columns)
├── data_cleaned/
│   ├── cleaned_seasonal_agriculture_performance.csv # Cleaned, imputed, and clustered dataset (39 columns)
│   └── statistical_ml_summary.json                  # Serialized ANOVA, t-test, and ML evaluation metrics
├── models/
│   └── model_bundle.joblib                          # Pre-trained Random Forest regressors, Scalers, & K-Means models
├── output_visualizations/                           # High-resolution 300 DPI figures
│   ├── fig1_seasonal_macro_metrics.png              # Multi-panel macro climatic and profit distributions
│   ├── fig2_crop_seasonal_economics.png             # Crop profit bars and yield heatmaps across seasons
│   ├── fig3_irrigation_resilience.png               # Irrigation method economics and water efficiency
│   ├── fig4_ml_feature_importance.png               # Random Forest feature importance rankings
│   └── fig5_state_seasonal_dynamics.png             # State-wise seasonal profitability heatmap
├── generate_analysis_and_charts.py                  # Pipeline script to generate figures and metrics
├── build_presentation.py                            # Automated PowerPoint slide builder script
├── build_and_execute_notebook.py                    # Automated Jupyter Notebook builder and executor
└── train_and_save_models.py                         # Model training and clustering script
```

---

## 🚀 How to Run

### 1. View the Jupyter Notebook
Open [`Seasonal_Agriculture_Performance_Analysis.ipynb`](Seasonal_Agriculture_Performance_Analysis.ipynb) in Jupyter Lab, VS Code, or Jupyter Notebook. All 24 cells are pre-executed with interactive tables, statistical ANOVA summaries, and seaborn plots.

```bash
jupyter notebook Seasonal_Agriculture_Performance_Analysis.ipynb
```

### 2. Launch the Interactive Web Dashboard
Run the multi-tab Streamlit web application:

```bash
streamlit run app.py
```
*Features:*
- **Executive KPI Dashboard:** Real-time filtering by state, crop, season, and irrigation technology.
- **Climate & Irrigation Simulator:** What-if calculator estimating profit gain and groundwater conserved.
- **AI Real-Time Predictor:** Sliders for soil pH, weather, fertilizer, and farm size to forecast Yield and Net Profit.
- **Unsupervised Typologies:** Interactive 2D PCA cluster visualization of farm groups.

### 3. Open Standalone HTML Dashboard
Double-click [`dashboard.html`](dashboard.html) in Windows Explorer to immediately open the responsive Chart.js dashboard in Google Chrome, Microsoft Edge, or Firefox without requiring Python.

---

## 📊 Presentation Submission
The PowerPoint presentation [`VOIS_Major_Project_PPT_Submission_Template.pptx`](VOIS_Major_Project_PPT_Submission_Template.pptx) has been populated with:
- **Slide 1:** Title, Student Name (*Dhanish Ladwani*), College Name, and AICTE Student ID.
- **Slide 2:** Problem Statement.
- **Slide 3:** Comprehensive Project Description.
- **Slide 4:** End Users (Farmers, Agronomists, Financial Institutions, Policymakers).
- **Slide 5:** Technology Stack Used (Python, Scikit-Learn, Pandas, SciPy, Matplotlib, Streamlit).
- **Slide 6:** Results Executive Overview.
- **Slides 7–10:** Embedded 300 DPI analytical charts with detailed findings and insights.
- **Slide 11:** Future Scope & Strategic Roadmap.
- **Slide 12:** GitHub Repository Link.
- **Slide 13:** VOIS Data Visualization Course Completion Certificate placeholder.
- **Slide 14:** Conclusion & Acknowledgements.

---

## 📜 Citation & Credits
Developed as part of the **Vodafone Intelligent Solutions (VOIS) & AICTE Emerging Technologies Internship Program (Batch 1, 2026–2027)**.
