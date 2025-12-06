import streamlit as st 
from streamlit_ux_utils import get_data
from streamlit_ux_results import display_results_ux
from streamlit_ux_monitoring import display_monitoring

# --- loading data ---
df = get_data("forecast")
metris = get_data("metrics")



page = st.sidebar.selectbox("Navigation", ["Résultats", "Monitoring", "Paramètres"])

if page ==  "Résultats":
    display_results_ux(df,metris)
elif page == "Monitoring":
    display_monitoring(df,metris)
