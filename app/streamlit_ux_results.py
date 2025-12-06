import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def display_results_ux(df_result, metrics):
    st.title("Prévisions de consommation électrique - Bordeaux")

    # --- Nom du modèle ---
    st.subheader(f"Modèle utilisé : {metrics.get('model', 'Inconnu')}")

    # --- Metrics simples ---
    st.subheader("Performance du modèle")
    wape = metrics.get("wape", None)
    if wape is not None:
        st.metric("WAPE", f"{wape:.2f} %")
        st.markdown(
            "*WAPE signifie `Weighted Absolute Percentage Error`. Plus il est faible, plus la prédiction est proche de la consommation réelle.*"
        )

    # --- Filtrer les derniers jours pour la visualisation ---
    last_days = 90
    forecast_days = 6
    df_plot = df_result.copy()
    df_plot = df_plot[df_plot['ds'] >= df_plot['ds'].max() - pd.Timedelta(days=last_days + forecast_days)]

    # --- Graphique avec intervalle de confiance ---
    fig = go.Figure()

    # Intervalle de confiance en arrière-plan
    if 'yhat_lower' in df_plot.columns and 'yhat_upper' in df_plot.columns:
        fig.add_traces(go.Scatter(
            x=df_plot['ds'],
            y=df_plot['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_traces(go.Scatter(
            x=df_plot['ds'],
            y=df_plot['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(0, 176, 246, 0.2)',
            name='Intervalle de confiance'
        ))

    # Courbe prédite
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['yhat'],
        mode='lines+markers',
        name='Prévision',
        line=dict(color='blue')
    ))

    # Courbe réelle
    fig.add_trace(go.Scatter(
        x=df_plot['ds'],
        y=df_plot['y'],
        mode='lines+markers',
        name='Réel',
        line=dict(color='orange')
    ))

    # Ajuster automatiquement l'échelle Y pour voir les variations
    y_min = min(df_plot[['y','yhat','yhat_lower']].min()) * 0.95
    y_max = max(df_plot[['y','yhat','yhat_upper']].max()) * 1.05
    fig.update_yaxes(range=[y_min, y_max], title_text="Consommation (MW)")
    fig.update_xaxes(title_text="Date")

    st.plotly_chart(fig, use_container_width=True)

    # --- Prévisions futures ---
    st.subheader("Prévisions pour les prochains jours")
    st.dataframe(df_plot[['ds', 'y', 'yhat']].tail(forecast_days))

