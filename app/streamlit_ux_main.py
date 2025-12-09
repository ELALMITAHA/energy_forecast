import streamlit as st 
from streamlit_ux_utils import get_data
from streamlit_ux_results import display_results_ux
from streamlit_ux_monitoring import display_monitoring


st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Remove top and side padding ---
st.markdown("""
    <style>
        /* Remove padding around the main content */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Optional: remove the big gap above the title */
        header[data-testid="stHeader"] {
            height: 5px;
        }

        /* Optional: tighten spacing between widgets */
        div.stButton > button {
            margin-top: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- loading data ---
df = get_data("forecast")
metris = get_data("metrics")

page = st.sidebar.selectbox("Navigation", ["Résultats", "Monitoring"])

if page ==  "Résultats":
    display_results_ux(df,metris)
elif page == "Monitoring":
    display_monitoring(df,metris)
