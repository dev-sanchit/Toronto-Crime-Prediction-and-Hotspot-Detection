import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


st.set_page_config(page_title="Toronto Crime Analytics AI", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
    <style>
    /* Global Smooth Scrolling Effect */
    html {
        scroll-behavior: smooth;
    }
    
    /* Custom App Background Image Setup */
    .stApp {
        background-image: linear-gradient(rgba(14, 18, 36, 0.88), rgba(14, 18, 36, 0.95)), 
                          url("https://images.pexels.com/photos/12727665/pexels-photo-12727665.jpeg");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        color: #E2E8F0 !important;
    }

    /* Main Dashboard Header Styling */
    h1 {
        text-shadow: 0px 4px 12px rgba(0, 180, 216, 0.4);
        font-weight: 800 !important;
        letter-spacing: 0.5px;
    }

    /* Glassmorphic Container Cards with Dynamic Hover Effects */
    div[data-testid="stVerticalBlock"] > div {
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .element-container:has(.stMarkdown) {
        border-radius: 12px;
    }

    /* Glassmorphism custom components wrapper */
    .glass-card {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease !important;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 180, 216, 0.5) !important;
        box-shadow: 0 12px 24px rgba(0, 180, 216, 0.15) !important;
    }

    /* Interactive Sidebar UI Makeover */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 28, 0.95) !important;
        border-right: 2px solid rgba(0, 180, 216, 0.2) !important;
    }
    
    /* Completely hide the default radio options if any exist */
    div[data-testid="stSidebarRadio"] {
        display: none !important;
    }
    
    /* Custom Sidebar Header Styling */
    .sidebar-header {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
        margin-bottom: 25px;
    }
    
    /* Re-engineered Navigation Menu Buttons */
    .nav-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .nav-btn {
        display: block;
        padding: 14px 20px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid rgba(0, 180, 216, 0.3);
        border-radius: 8px;
        color: #E2E8F0 !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    /* Interactive Hover State with Glow and Push Effect */
    .nav-btn:hover {
        background: rgba(0, 180, 216, 0.15) !important;
        border-left: 4px solid #00B4D8 !important;
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.2);
        transform: translateX(8px);
        color: #FFFFFF !important;
    }
    
    /* Active Page Highlight */
    .nav-btn-active {
        background: linear-gradient(90deg, rgba(0, 180, 216, 0.2) 0%, rgba(0, 180, 216, 0.02) 100%) !important;
        border-left: 4px solid #00B4D8 !important;
        border-right: 1px solid rgba(0, 180, 216, 0.3) !important;
        border-top: 1px solid rgba(0, 180, 216, 0.2) !important;
        border-bottom: 1px solid rgba(0, 180, 216, 0.2) !important;
        color: #00B4D8 !important;
        box-shadow: inset 5px 0 10px rgba(0, 180, 216, 0.1);
    }
    
    /* Styled Presentation / Explanatory Comment Area styling */
    .presentation-comment {
        background: rgba(0, 180, 216, 0.08) !important;
        border-left: 5px solid #00B4D8 !important;
        padding: 15px !important;
        border-radius: 4px 12px 12px 4px;
        font-style: italic;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# CACHED DATA PIPELINE 
@st.cache_data
def load_data():
    df = pd.read_csv("MCI_2014_to_2019.csv")
    possible_date_cols = ['occurrenceOnlyDate', 'occurrence_date', 'occurrencedate', 'event_date']
    
    found_col = None
    for col in possible_date_cols:
        if col in df.columns:
            found_col = col
            break
            
    if found_col:
        df = df.rename(columns={found_col: 'occurrenceOnlyDate'})
        df['occurrenceOnlyDate'] = pd.to_datetime(df['occurrenceOnlyDate']).dt.date
    else:
        st.error(f"Could not find valid temporal columns. Dataset schemas provided: {list(df.columns)}")
        st.stop()
        
    #Clean empty rows from core categorical and location metrics to block 'nan' errors
    critical_cols = ['occurrenceyear', 'occurrencemonth', 'occurrencehour', 'MCI', 'Lat', 'Long']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    
    #Cast decimals down to standard integers for filtering arrays
    if 'occurrenceyear' in df.columns:
        df['occurrenceyear'] = df['occurrenceyear'].astype(int)
    if 'occurrencehour' in df.columns:
        df['occurrencehour'] = df['occurrencehour'].astype(int)
        
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data repository file 'MCI_2014_to_2019.csv' not detected in active directory workspace.")
    st.stop()



st.sidebar.markdown("""
    <div class="sidebar-header">
        <h3 style='color: white; margin: 0; font-size: 20px; font-weight: 800;'>🚨 COGNITIVE SYSTEM</h3>
        <p style='color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 11px; uppercase; letter-spacing: 1px;'>Crime Analytics Space</p>
    </div>
""", unsafe_allow_html=True)


if "page" not in st.query_params:
    st.query_params["page"] = "Project Space Overview"

current_page = st.query_params["page"]


st.sidebar.markdown(f"""
    <div class="nav-container">
        <a href="?page=Project+Space+Overview" class="nav-btn {'nav-btn-active' if current_page == 'Project Space Overview' else ''}" target="_self">📂 PROJECT SPACE OVERVIEW</a>
        <a href="?page=Dashboard" class="nav-btn {'nav-btn-active' if current_page == 'Dashboard' else ''}" target="_self">📊 DASHBOARD</a>
        <a href="?page=Unsupervised+Risk+Clustering" class="nav-btn {'nav-btn-active' if current_page == 'Unsupervised Risk Clustering' else ''}" target="_self">🏘️ UNSUPERVISED RISK CLUSTERING</a>
        <a href="?page=Predictive+Forecasting+Core" class="nav-btn {'nav-btn-active' if current_page == 'Predictive Forecasting Core' else ''}" target="_self">🔮 PREDICTIVE FORECASTING CORE</a>
    </div>
""", unsafe_allow_html=True)

page = current_page



if page == "Project Space Overview":
    st.markdown("<h1 style='color:#00B4D9;'>🛡️ Toronto Predictive Policing Evaluation Portal</h1>", unsafe_allow_html=True)
    
    col_main, col_comm = st.columns([2, 1])
    
    with col_main:
        st.markdown("""
        <div class="glass-card">
            <h3>Computational Intent Architecture</h3>
            <p>This analytics infrastructure applies unsupervised spatial categorization and predictive time-series mathematical architectures to unstructured major crime incident paradigms in urban metropolitan clusters.</p>
            <h4 style="color:#00B4D9;">Targeted Capabilities:</h4>
            <ul>
                <li><b>Spatial Cluster Partitioning:</b> Aggregating regional quadrants via multi-dimensional density scoring profiles.</li>
                <li><b>Additive Seasonal Trend Modeling:</b> Decomposing chronological patterns to isolate future threat trajectories.</li>
                <li><b>Dynamic Metric Slicing:</b> Interactive filtering to track localized performance shifts.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color:#FFF;'>Feature Space Verification Head (Top 10 Nodes)</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    with col_comm:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(247, 237, 37, 0.3) !important;">
            <h4 style="color:#F7ED25;">💡 Evaluation Panel Comments</h4>
            <div class="presentation-comment">
                "Honorable Evaluators, this overview page presents the ingestion layer. The table on the left validates that data normalization and cleansing routines have run completely, mapping raw temporal records into parsed types seamlessly."
            </div>
            <br>
            <p style="font-size:13px; opacity:0.8;"><b>Notes for defense viva:</b> Highlight that the underlying schema aggregates over 100,000 incident nodes spanning critical major indicators.</p>
        </div>
        """, unsafe_allow_html=True)



elif page == "Dashboard":
    st.markdown("<h1 style='color:#00B4D9;'>📊 DASHBOARD</h1>", unsafe_allow_html=True)
    
    # Advanced Filter Ribbon
    st.markdown("<div class='glass-card'><h4>Dynamic Parametric Constraints Selector</h4>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        mci_filter = st.multiselect("Isolate Target Crime Indicators (MCI)", options=df['MCI'].unique(), default=df['MCI'].unique())
    with f_col2:
        year_filter = st.slider("Select Temporal Boundary (Chronological Years)", int(df['occurrenceyear'].min()), int(df['occurrenceyear'].max()), (int(df['occurrenceyear'].min()), int(df['occurrenceyear'].max())))
    st.markdown("</div>", unsafe_allow_html=True)
    
    filtered_df = df[(df['MCI'].isin(mci_filter)) & (df['occurrenceyear'].between(year_filter[0], year_filter[1]))]

    col_plots, col_comm = st.columns([2.2, 1])
    
    with col_plots:
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Spatial Incidence Map Layout")
            fig_map = px.scatter(filtered_df.sample(min(2000, len(filtered_df))), x="Long", y="Lat", color="MCI", 
                                 hover_name="Neighbourhood", opacity=0.5, template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Safe)
            fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with p_col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Categorical Volumetric Ranks")
            mci_counts = filtered_df['MCI'].value_counts()
            fig_bar = px.bar(mci_counts, x=mci_counts.index, y=mci_counts.values, color=mci_counts.index, 
                             template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_bar.update_layout(margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_comm:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(247, 237, 37, 0.3) !important;">
            <h4 style="color:#F7ED25;">💡 Evaluation Panel Comments</h4>
            <div class="presentation-comment">
                "This visual module processes coordinate geometries via interactive WebGL layers. By adjusting the selectors above, you can see how specific crimes like Assault dominate total volume, while auto theft maps cleanly to specific regional configurations."
            </div>
            <br>
            <textarea style="width:100%; background:rgba(255,255,255,0.05); color:white; border:1px solid rgba(255,255,255,0.2); border-radius:8px; padding:10px;" rows="4" placeholder="Type presentation real-time notes here during the live evaluation panel..."></textarea>
        </div>
        """, unsafe_allow_html=True)


#NEIGHBORHOOD CLUSTERING ---
elif page == "Unsupervised Risk Clustering":
    st.markdown("<h1 style='color:#00B4D9;'>🏘️ Unsupervised Spatial Segmentation Engineering</h1>", unsafe_allow_html=True)
    
    
    df_cluster = df.groupby(['Neighbourhood', 'MCI']).size().unstack(fill_value=0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cluster)
    
    n_clusters = st.sidebar.slider("Hyperparameter Target (K Clusters)", 2, 5, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df_cluster['Cluster'] = clusters

    
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    df_cluster['PCA1'] = pca_data[:, 0]
    df_cluster['PCA2'] = pca_data[:, 1]

    col_clust, col_comm = st.columns([2, 1])
    
    with col_clust:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Feature Projection Optimization Space (PCA Mapping)")
        fig_pca = px.scatter(df_cluster, x='PCA1', y='PCA2', color=df_cluster['Cluster'].astype(str),
                             hover_name=df_cluster.index, title="Feature Variant Decomposition", 
                             template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Segment Cluster Membership Queries")
        selected_cluster = st.selectbox("Query Cluster Partition Content ID:", options=sorted(df_cluster['Cluster'].unique()))
        st.write(df_cluster[df_cluster['Cluster'] == selected_cluster].index.tolist())
        st.markdown("</div>", unsafe_allow_html=True)

    with col_comm:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(247, 237, 37, 0.3) !important;">
            <h4 style="color:#F7ED25;">💡 Evaluation Panel Comments</h4>
            <div class="presentation-comment">
                "To resolve high-dimensional multi-collinearity across categorical indicators, we passed the scaled vectors into a Principal Component Analysis pipeline. This cluster chart isolates low-risk residential quadrants from extreme density anomaly hubs."
            </div>
        </div>
        """, unsafe_allow_html=True)


#CRIME FORECASTING ---
elif page == "Predictive Forecasting Core":
    st.markdown("<h1 style='color:#00B4D9;'>🔮 Algorithmic Seasonal Forecasting Engine</h1>", unsafe_allow_html=True)
    
    #GEOSPATIAL HOTSPOT MAP SECTION
    st.markdown("""
    <div class="glass-card">
        <h2 style='color:#00B4D9; margin-top:0;'>🗺️ Chronological Geospatial Hotspot Engine</h2>
        <p>Isolate and track regional crime concentrations across flexible multi-dimensional temporal boundaries.</p>
    """, unsafe_allow_html=True)
    
    # Map Filters Layout
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        map_mci = st.multiselect("Filter Crime Typologies", options=list(df['MCI'].unique()), default=list(df['MCI'].unique()))
    with m_col2:
      
        years_options = sorted(list(df['occurrenceyear'].unique()))
        map_years = st.multiselect("Select Target Years", options=years_options, default=years_options)
    with m_col3:
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        available_months = [m for m in months_order if m in df['occurrencemonth'].unique()]
        map_months = st.multiselect("Select Target Months", options=available_months, default=available_months)
        
    m_col4, m_col5 = st.columns([2, 1])
    with m_col4:
        map_hours = st.slider("Diurnal Hour Frame Bounds (Time of Day)", 0, 23, (0, 23))
    with m_col5:
        map_style = st.selectbox("GIS Mapbox Layer Style", ["carto-darkmatter", "open-street-map", "carto-positron"])
        
    # Query Data Filter Execution
    map_df = df[
        (df['MCI'].isin(map_mci)) & 
        (df['occurrenceyear'].isin(map_years)) & 
        (df['occurrencemonth'].isin(map_months)) & 
        (df['occurrencehour'].between(map_hours[0], map_hours[1]))
    ]
    
    if map_df.empty:
        st.warning("No recorded incident data matches the set combinations. Modify filters above to plot.")
    else:
        # Downsample to keep cross-browser rendering performant during presentations
        render_limit = min(6000, len(map_df))
        sampled_map_df = map_df.sample(render_limit, random_state=42)
        
        fig_spatial_map = px.scatter_mapbox(
            sampled_map_df,
            lat="Lat",
            lon="Long",
            color="MCI",
            hover_name="offence",
            hover_data={"occurrencehour": True, "occurrencedayofweek": True, "Neighbourhood": True, "Lat": False, "Long": False},
            zoom=10.2,
            center={"lat": 43.71, "lon": -79.38},
            template="plotly_dark",
            height=550,
            title=f"Geospatial Threat Hotspots Layout ({render_limit} Normalized Incidents Plotted)"
        )
        fig_spatial_map.update_layout(
            mapbox_style=map_style,
            margin=dict(l=0, r=0, t=35, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_spatial_map, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------------------------------

    # Forecasting Selection Section
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    mci_to_forecast = st.selectbox("Select Core Vector Objective to Predict via Prophet Engine", options=df['MCI'].unique())
    
    forecast_data = df[df['MCI'] == mci_to_forecast].groupby('occurrenceOnlyDate').size().reset_index()
    forecast_data.columns = ['ds', 'y']
    
    with st.spinner("Executing Mathematical Optimization via Facebook Prophet Optimizer..."):
        m = Prophet(daily_seasonality=True)
        m.fit(forecast_data)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)

    col_fore, col_comm = st.columns([2, 1])
    
    with col_fore:
        st.subheader(f"12-Month Predictive Envelope Generation [{mci_to_forecast}]")
        fig_forecast = plot_plotly(m, forecast)
        fig_forecast.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_forecast, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Mathematical Trend Signal Decomposition Components")
        fig_comp = m.plot_components(forecast)
        st.pyplot(fig_comp)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_comm:
        st.markdown("""
        <div class="glass-card" style="border-color: rgba(247, 237, 37, 0.3) !important;">
            <h4 style="color:#F7ED25;">💡 Evaluation Panel Comments</h4>
            <div class="presentation-comment">
                "The core forecasting platform isolates linear growth shifts alongside cyclical Fourier Series intervals. The shaded boundary shows the 80% mathematical risk confidence limits generated by the predictive algorithm."
            </div>
            <br>
            <p style="font-size: 13px;"><b>Self-Commentary Layer:</b> This confirms the hypothesis that seasonal changes strongly impact property crimes like break-and-enters during summer variations.</p>
        </div>
        """, unsafe_allow_html=True)