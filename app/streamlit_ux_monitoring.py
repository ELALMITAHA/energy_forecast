import pandas as pd 
import streamlit as st 

import plotly.express as px 


import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def display_monitoring(df, metrics):
    st.title("Dashboard Monitoring du Modèle")

    # --- Colonnes pour les métriques clés ---
    col1, col2, col3 = st.columns(3)

    wape = metrics.get("wape", None)
    mae = metrics.get("mae", None)
    rmse = metrics.get("rmse", None)

    if wape is not None:
        col1.metric("WAPE", f"{wape:.2f} %")
        col1.caption("Weighted Absolute Percentage Error (plus faible = mieux)")

    if mae is not None:
        col2.metric("MAE", f"{mae:.0f}")
        col2.caption("Mean Absolute Error (erreur moyenne absolue)")

    if rmse is not None:
        col3.metric("RMSE", f"{rmse:.0f}")
        col3.caption("Root Mean Squared Error (erreur quadratique moyenne)")

    st.markdown("---")  # séparation visuelle

    # --- Graphique de la consommation réelle vs prédite ---
    df_plot = df.copy()
    last_days = 90  # visualiser les derniers jours
    df_plot = df_plot[df_plot['ds'] >= df_plot['ds'].max() - pd.Timedelta(days=last_days)]

    fig = go.Figure()

    # Intervalle de confiance (fond)
    if 'yhat_lower' in df_plot.columns and 'yhat_upper' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['ds'], y=df_plot['yhat_upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_plot['ds'], y=df_plot['yhat_lower'],
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)',
            name='Intervalle de confiance'
        ))

    # Courbes réel vs prévision
    fig.add_trace(go.Scatter(
        x=df_plot['ds'], y=df_plot['yhat'],
        mode='lines+markers', name='Prévision', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['ds'], y=df_plot['y'],
        mode='lines+markers', name='Réel', line=dict(color='orange')
    ))

    # Ajuster l’échelle y pour plus de lisibilité
    y_min = min(df_plot[['y','yhat','yhat_lower']].min()) * 0.95
    y_max = max(df_plot[['y','yhat','yhat_upper']].max()) * 1.05
    fig.update_yaxes(range=[y_min, y_max], title_text="Consommation (MW)")
    fig.update_xaxes(title_text="Date")

    st.plotly_chart(fig, use_container_width=True)

    # --- Section Monitoring avancé ---
    st.subheader("Analyse avancée")
    st.write("Ici on pourra ajouter tests de drift, alertes d'anomalies ou comparaison multi-modèles.")
