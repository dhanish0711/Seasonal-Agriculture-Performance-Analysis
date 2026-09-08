import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Seasonal Agriculture Performance Analytics | VOIS AICTE",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B365D;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1B365D;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        font-size: 15px;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Data & Models
@st.cache_data
def load_data():
    df = pd.read_csv('data_cleaned/cleaned_seasonal_agriculture_performance.csv')
    if 'Profit_Tier' not in df.columns:
        def classify_profit(p):
            if p < 0: return 'Deficit (Loss)'
            elif p < 100000: return 'Low Profit (<₹1L)'
            elif p < 500000: return 'Moderate Profit (₹1L-₹5L)'
            else: return 'High Profit (>₹5L)'
        df['Profit_Tier'] = df['Profit_INR'].apply(classify_profit)
    return df

@st.cache_resource
def load_models():
    bundle = joblib.load('models/model_bundle.joblib')
    return bundle

df = load_data()
models = load_models()

# Header
st.markdown('<div class="main-title">🌾 Seasonal Agriculture Performance Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>VOIS AICTE Batch 1 (2026–2027) Major Project</b> | Author: <i>Dhanish Ladwani</i> | 4,000 Multi-Regional Farm Records</div>', unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.header("🔍 Dynamic Filters")
seasons = st.sidebar.multiselect("Select Season(s):", options=['Kharif', 'Rabi', 'Zaid'], default=['Kharif', 'Rabi', 'Zaid'])
all_crops = sorted(df['Crop'].unique().tolist())
crops = st.sidebar.multiselect("Select Crop(s):", options=all_crops, default=all_crops)
all_states = sorted(df['State'].unique().tolist())
states = st.sidebar.multiselect("Select State(s):", options=all_states, default=all_states)
all_irrs = sorted(df['Irrigation_Method'].unique().tolist())
irrigations = st.sidebar.multiselect("Select Irrigation Method(s):", options=all_irrs, default=all_irrs)

# Filter Data
dff = df[
    (df['Season'].isin(seasons)) &
    (df['Crop'].isin(crops)) &
    (df['State'].isin(states)) &
    (df['Irrigation_Method'].isin(irrigations))
]

if dff.empty:
    st.warning("No farm operations match the selected filter combination. Please broaden your selection.")
    st.stop()

# Key Performance KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Farms Analyzed", f"{len(dff):,}", f"{(len(dff)/len(df)*100):.1f}% of total")
with col2:
    mean_yield = dff['Yield_Tonnes_Ha'].mean()
    st.metric("Mean Yield", f"{mean_yield:.2f} t/ha")
with col3:
    mean_profit = dff['Profit_INR'].mean()
    st.metric("Mean Net Profit", f"₹{mean_profit:,.0f}", delta="Profitable" if mean_profit > 0 else "-Deficit", delta_color="normal" if mean_profit > 0 else "inverse")
with col4:
    mean_water_eff = dff['Water_Efficiency_t_per_1000m3'].mean()
    st.metric("Water Efficiency", f"{mean_water_eff:.2f} t/k m³")
with col5:
    mean_pest = dff['Disease_Pest_Risk_pct'].mean()
    st.metric("Pest / Disease Risk", f"{mean_pest:.1f}%")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive KPI Dashboard",
    "🌾 Crop & Economic Disparities",
    "💧 Climate & Irrigation Simulator",
    "🤖 Real-Time ML Predictor",
    "🔬 Unsupervised Typologies & Stats"
])

# ==================== TAB 1: EXECUTIVE KPI DASHBOARD ====================
with tab1:
    st.subheader("Seasonal Divergence: Environmental Envelope & Financial Distribution")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_scatter = px.scatter(
            dff,
            x='Avg_Temperature_C',
            y='Rainfall_mm',
            color='Season',
            size='Yield_Tonnes_Ha',
            hover_data=['State', 'Crop', 'Profit_INR', 'Irrigation_Method'],
            title="Agro-Climatic Envelopes: Temperature vs. Rainfall (Colored by Season)",
            color_discrete_map={'Kharif': '#1f77b4', 'Rabi': '#2ca02c', 'Zaid': '#d62728'},
            labels={'Avg_Temperature_C': 'Average Temperature (°C)', 'Rainfall_mm': 'Precipitation (mm)'}
        )
        st.plotly_chart(fig_scatter)
        
    with c2:
        fig_box = px.box(
            dff,
            x='Season',
            y='Profit_INR',
            color='Season',
            points=False,
            title="Distribution of Farm Net Profit by Season (₹ INR)",
            color_discrete_map={'Kharif': '#1f77b4', 'Rabi': '#2ca02c', 'Zaid': '#d62728'},
            labels={'Profit_INR': 'Net Profit (₹ INR)'}
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even (₹0)")
        st.plotly_chart(fig_box)

    c3, c4 = st.columns(2)
    with c3:
        # Revenue vs Cost by Season
        rev_cost = dff.groupby('Season')[['Total_Cost_INR', 'Revenue_INR', 'Profit_INR']].mean().reset_index()
        fig_rev = go.Figure(data=[
            go.Bar(name='Total Cost (INR)', x=rev_cost['Season'], y=rev_cost['Total_Cost_INR'], marker_color='#e74c3c'),
            go.Bar(name='Revenue (INR)', x=rev_cost['Season'], y=rev_cost['Revenue_INR'], marker_color='#27ae60')
        ])
        fig_rev.update_layout(
            barmode='group',
            title='Mean Total Cost vs. Gross Revenue by Season',
            yaxis_title='Amount (₹ INR)'
        )
        st.plotly_chart(fig_rev)

    with c4:
        # Pest Risk vs Water Volume
        pest_water = dff.groupby('Season')[['Disease_Pest_Risk_pct', 'Water_Used_m3']].mean().reset_index()
        fig_pw = px.bar(
            pest_water,
            x='Season',
            y='Disease_Pest_Risk_pct',
            color='Season',
            title='Mean Disease & Pest Risk (%) by Season',
            color_discrete_map={'Kharif': '#1f77b4', 'Rabi': '#2ca02c', 'Zaid': '#d62728'},
            text_auto='.1f'
        )
        st.plotly_chart(fig_pw)

# ==================== TAB 2: CROP & ECONOMIC DISPARITIES ====================
with tab2:
    st.subheader("Granular Crop Economics: Commercial Powerhouses vs. Staple Grain Squeeze")
    
    c1, c2 = st.columns([3, 2])
    with c1:
        crop_season_profit = dff.groupby(['Crop', 'Season'])['Profit_INR'].mean().reset_index()
        fig_crop = px.bar(
            crop_season_profit,
            x='Crop',
            y='Profit_INR',
            color='Season',
            barmode='group',
            title='Mean Net Profit (INR) by Crop across Cropping Cycles',
            color_discrete_map={'Kharif': '#1f77b4', 'Rabi': '#2ca02c', 'Zaid': '#d62728'},
            labels={'Profit_INR': 'Net Profit (₹)'}
        )
        fig_crop.add_hline(y=0, line_dash="solid", line_color="black")
        st.plotly_chart(fig_crop)
        
    with c2:
        # Sunburst chart of filtered records
        fig_sun = px.sunburst(
            dff,
            path=['Season', 'Crop', 'Profit_Tier'],
            values='Farm_Area_Hectares',
            color='Profit_INR',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Interactive Hierarchical Decomposition: Season → Crop → Profitability'
        )
        st.plotly_chart(fig_sun)

    # Detailed Summary Table
    st.markdown("#### Empirical Crop Economics Breakdown")
    crop_table = dff.groupby('Crop').agg(
        Mean_Yield=('Yield_Tonnes_Ha', 'mean'),
        Market_Price=('Market_Price_INR_Tonne', 'mean'),
        Mean_Cost=('Total_Cost_INR', 'mean'),
        Mean_Revenue=('Revenue_INR', 'mean'),
        Mean_Profit=('Profit_INR', 'mean'),
        Profitable_Share=('Is_Profitable', lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    crop_table['Mean_Profit'] = crop_table['Mean_Profit'].apply(lambda x: f"₹{x:,.0f}")
    crop_table['Mean_Cost'] = crop_table['Mean_Cost'].apply(lambda x: f"₹{x:,.0f}")
    crop_table['Mean_Revenue'] = crop_table['Mean_Revenue'].apply(lambda x: f"₹{x:,.0f}")
    crop_table['Market_Price'] = crop_table['Market_Price'].apply(lambda x: f"₹{x:,.0f}/t")
    crop_table['Mean_Yield'] = crop_table['Mean_Yield'].apply(lambda x: f"{x:.2f} t/ha")
    st.dataframe(crop_table)

# ==================== TAB 3: CLIMATE & IRRIGATION SIMULATOR ====================
with tab3:
    st.subheader("💧 Climate & Irrigation Modernization Simulator")
    st.markdown("Evaluate how upgrading irrigation technology from traditional surface flooding to precision micro-irrigation transforms summer (Zaid) resilience and water conservation.")
    
    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        st.markdown("### ⚙️ Farm Parameters")
        sim_area = st.slider("Farm Acreage (Hectares):", min_value=1.0, max_value=25.0, value=8.0, step=0.5)
        sim_season = st.selectbox("Cropping Season:", ['Zaid (Summer)', 'Rabi (Winter)', 'Kharif (Monsoon)'])
        sim_crop = st.selectbox("Selected Crop Variety:", all_crops, index=all_crops.index('Rice') if 'Rice' in all_crops else 0)
        curr_irr = st.selectbox("Current Irrigation Method:", ['Flood', 'Rainfed'])
        target_irr = st.selectbox("Proposed Modernization Method:", ['Drip', 'Sprinkler'])

    with sc_col2:
        st.markdown("### 📈 Projected Impact & Payback")
        # Empirical benchmarks from our dataset
        season_clean = sim_season.split()[0]
        curr_subset = df[(df['Season'] == season_clean) & (df['Irrigation_Method'] == curr_irr)]
        target_subset = df[(df['Season'] == season_clean) & (df['Irrigation_Method'] == target_irr)]
        
        curr_profit_ha = curr_subset['Profit_per_Hectare'].mean()
        target_profit_ha = target_subset['Profit_per_Hectare'].mean()
        curr_water_ha = (curr_subset['Water_Used_m3'] / curr_subset['Farm_Area_Hectares']).mean()
        target_water_ha = (target_subset['Water_Used_m3'] / target_subset['Farm_Area_Hectares']).mean()
        
        profit_delta = (target_profit_ha - curr_profit_ha) * sim_area
        water_saved = max(0, (curr_water_ha - target_water_ha) * sim_area)
        
        # Capital expenditure estimation (~₹65,000/ha for Drip, subsidized at 45%)
        capex_per_ha = 65000 * 0.55 if target_irr == 'Drip' else 45000 * 0.55
        total_capex = capex_per_ha * sim_area
        payback_seasons = total_capex / max(1, profit_delta)
        
        st.metric("Net Seasonal Profit Gain", f"₹{profit_delta:,.0f}", delta=f"+₹{profit_delta:,.0f}")
        st.metric("Annual Water Volume Conserved", f"{water_saved:,.0f} m³", delta="Groundwater Saved")
        st.metric("Estimated Installation Net Cost (after subsidy)", f"₹{total_capex:,.0f}")
        st.info(f"💡 **Estimated Payback Period:** **{payback_seasons:.1f} cropping seasons** to fully amortize equipment investment through higher yield and reduced pumping costs.")

# ==================== TAB 4: REAL-TIME ML PREDICTOR ====================
with tab4:
    st.subheader("🤖 Real-Time Machine Learning Prediction Engine")
    st.markdown("Input specific soil, weather, and farm features to predict **Crop Yield** ($R^2 = 0.962$) and **Net Profit** ($R^2 = 0.792$) using pre-trained Random Forest models.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        p_crop = st.selectbox("Target Crop:", all_crops)
        p_season = st.selectbox("Target Season:", ['Kharif', 'Rabi', 'Zaid'])
        p_irr = st.selectbox("Irrigation Technology:", all_irrs)
        p_state = st.selectbox("State:", all_states)
        p_area = st.number_input("Farm Area (Hectares):", min_value=0.5, max_value=50.0, value=7.5, step=0.5)

    with col_input2:
        p_rainfall = st.slider("Seasonal Rainfall (mm):", min_value=100.0, max_value=1500.0, value=600.0, step=10.0)
        p_temp = st.slider("Average Temperature (°C):", min_value=15.0, max_value=42.0, value=26.5, step=0.5)
        p_humidity = st.slider("Humidity (%):", min_value=20.0, max_value=95.0, value=65.0, step=1.0)
        p_sunlight = st.slider("Sunlight Hours / Day:", min_value=3.0, max_value=12.0, value=7.5, step=0.5)
        p_soil_ph = st.slider("Soil pH Level:", min_value=4.5, max_value=9.0, value=6.8, step=0.1)

    with col_input3:
        p_soil_moist = st.slider("Soil Moisture (%):", min_value=5.0, max_value=50.0, value=25.0, step=1.0)
        p_n = st.number_input("Nitrogen (kg/ha):", min_value=20.0, max_value=250.0, value=120.0)
        p_p = st.number_input("Phosphorus (kg/ha):", min_value=10.0, max_value=150.0, value=60.0)
        p_k = st.number_input("Potassium (kg/ha):", min_value=20.0, max_value=250.0, value=100.0)
        p_fert = st.number_input("Total Fertilizer (kg/ha):", min_value=50.0, max_value=500.0, value=185.0)
        p_pest = st.number_input("Pesticides (Litre/ha):", min_value=0.5, max_value=20.0, value=5.0)
        p_seed = st.slider("Seed Quality Score (0 to 1):", min_value=0.5, max_value=1.0, value=0.85, step=0.01)
        p_water = st.number_input("Expected Water Used (m³):", min_value=1000, max_value=50000, value=6000, step=500)
        p_risk = st.slider("Disease/Pest Risk (%):", min_value=10.0, max_value=95.0, value=45.0, step=1.0)

    if st.button("🚀 Predict Agricultural Performance", type="primary"):
        # Build single row dataframe with dummy encoding matching feature_names
        row_dict = {
            'Farm_Area_Hectares': p_area,
            'Rainfall_mm': p_rainfall,
            'Avg_Temperature_C': p_temp,
            'Humidity_pct': p_humidity,
            'Sunlight_Hours_Day': p_sunlight,
            'Soil_pH': p_soil_ph,
            'Soil_Moisture_pct': p_soil_moist,
            'Nitrogen_kg_ha': p_n,
            'Phosphorus_kg_ha': p_p,
            'Potassium_kg_ha': p_k,
            'Fertilizer_kg_ha': p_fert,
            'Pesticide_Litre_ha': p_pest,
            'Seed_Quality_Score': p_seed,
            'Water_Used_m3': p_water,
            'Disease_Pest_Risk_pct': p_risk,
            'Crop': p_crop,
            'Season': p_season,
            'Irrigation_Method': p_irr,
            'State': p_state
        }
        input_df = pd.DataFrame([row_dict])
        input_encoded = pd.get_dummies(input_df)
        
        # Align with model feature names
        full_input = pd.DataFrame(0, index=[0], columns=models['feature_names'])
        for col in input_encoded.columns:
            if col in full_input.columns:
                full_input[col] = input_encoded[col].values[0]
                
        pred_yield = models['rf_yield'].predict(full_input)[0]
        pred_profit = models['rf_profit'].predict(full_input)[0]
        pred_prod = pred_yield * p_area

        st.markdown("### 🎯 Model Forecast Results")
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Predicted Crop Yield", f"{pred_yield:.2f} Tonnes / Ha")
        with res2:
            st.metric("Predicted Total Production", f"{pred_prod:.1f} Tonnes")
        with res3:
            st.metric("Predicted Farm Net Profit", f"₹{pred_profit:,.0f}", delta="Profitable" if pred_profit > 0 else "Financial Deficit", delta_color="normal" if pred_profit > 0 else "inverse")

# ==================== TAB 5: UNSUPERVISED TYPOLOGIES & STATS ====================
with tab5:
    st.subheader("🔬 Unsupervised Farm Typologies & Statistical Validations")
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        fig_pca = px.scatter(
            dff,
            x='PCA1',
            y='PCA2',
            color='Cluster_Name',
            hover_data=['State', 'Crop', 'Season', 'Profit_INR'],
            title='Unsupervised Farm Typology Clusters (PCA Projection)',
            labels={'PCA1': 'Principal Component 1 (Scale & Production)', 'PCA2': 'Principal Component 2 (Cost & Water)'}
        )
        st.plotly_chart(fig_pca)
        
    with col_u2:
        # Cluster Profiles table
        cluster_tbl = df.groupby('Cluster_Name').agg(
            Farms_Count=('Farm_ID', 'count'),
            Mean_Profit=('Profit_INR', 'mean'),
            Mean_Yield=('Yield_Tonnes_Ha', 'mean'),
            Mean_Water=('Water_Used_m3', 'mean'),
            Dominant_Season=('Season', lambda x: x.mode()[0])
        ).reset_index()
        cluster_tbl['Mean_Profit'] = cluster_tbl['Mean_Profit'].apply(lambda x: f"₹{x:,.0f}")
        cluster_tbl['Mean_Yield'] = cluster_tbl['Mean_Yield'].apply(lambda x: f"{x:.2f} t/ha")
        cluster_tbl['Mean_Water'] = cluster_tbl['Mean_Water'].apply(lambda x: f"{x:,.0f} m³")
        st.markdown("#### Farm Typology Profiles (K-Means Clustering)")
        st.dataframe(cluster_tbl)
        
    st.markdown("---")
    st.markdown("#### 📑 Summary of Formal Statistical Hypothesis Tests")
    stat_summary_data = [
        {"Hypothesis Test": "One-Way ANOVA: Net Profit across Seasons", "Test Statistic": "F = 34.292", "p-value": "1.71e-15", "Outcome": "Highly Significant (p < 0.001)"},
        {"Hypothesis Test": "Kruskal-Wallis: Net Profit across Seasons", "Test Statistic": "H = 101.926", "p-value": "7.36e-23", "Outcome": "Highly Significant (p < 0.001)"},
        {"Hypothesis Test": "One-Way ANOVA: Disease/Pest Risk across Seasons", "Test Statistic": "F = 1049.467", "p-value": "< 1e-300", "Outcome": "Extreme Significance (p < 0.001)"},
        {"Hypothesis Test": "One-Way ANOVA: Water Efficiency across Seasons", "Test Statistic": "F = 6.948", "p-value": "0.00097", "Outcome": "Significant (p < 0.01)"},
        {"Hypothesis Test": "Intra-Crop Yield ANOVA (All 8 Crops)", "Test Statistic": "F ∈ [10.84, 24.65]", "p-value": "All p < 0.0001", "Outcome": "Every single crop varies seasonally"},
        {"Hypothesis Test": "Welch's t-test: Zaid Drip vs Flood Profitability", "Test Statistic": "t = 1.835", "p-value": "0.0678", "Outcome": "Drip provides +₹91,078 profit buffer"}
    ]
    st.table(pd.DataFrame(stat_summary_data))
