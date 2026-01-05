from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


from streamlit_extras.metric_cards import style_metric_cards


def display_results_ux(df_result, metrics):
    st.title("Prévisions de consommation électrique - Bordeaux")

    col_model_wap, col_pred_vs_reel = st.columns((2, 5))

    with col_model_wap:
        col_model, col_wape = st.columns(2)
        # --- Metric : Nom du modèle ---

        with col_model:
            model_name = metrics.get("model", "Inconnu")
            st.metric(" Modèle", model_name.upper())

        # --- Metric : WAPE ---
        with col_wape:
            wape = metrics.get("wape", None)
            if wape is not None:
                st.metric(" WAPE", f"{wape:.2f} %")

        # --- Style global des cards ---
        style_metric_cards(
            background_color="#111827",  # plus sombre -> plus premium
            border_color="#1F2937",
            border_left_color="#3B82F6",  # bleu electric
            border_size_px=5,
            border_radius_px=14,  # arrondi plus moderne
            box_shadow=True,
        )

        forecast_days = 6
        df_forecast_6 = (
            df_result.sort_values("ds").tail(forecast_days)[["ds", "yhat"]].copy()
        )
        df_forecast_6["yhat"] = df_forecast_6["yhat"].map(lambda x: f"{x:,.2f} kW")

        # --- CSS dark theme ---
        table_style = """
        <style>
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: #1F2937;
            color: #E2E8F0;
            border-radius: 10px;
            overflow: hidden;
        }
        th {
            background-color: #111827;
            padding: 10px;
            text-align: center;
        }
        td {
            padding: 8px;
            text-align: center;
            border-bottom: 1px solid #2D3748;
        }
        tr:last-child td {
            border-bottom: none;
        }
        </style>
        """

        # --- Construire le HTML manuellement ---
        table_html = "<table><tr><th>Date</th><th>Prévision (kW)</th></tr>"
        for _, row in df_forecast_6.iterrows():
            table_html += f"<tr><td>{row['ds']}</td><td>{row['yhat']}</td></tr>"
        table_html += "</table>"

        # --- Affichage ---
        st.markdown("### 📅 Prévisions J → J+5", unsafe_allow_html=True)
        st.markdown(table_style + table_html, unsafe_allow_html=True)

    with col_pred_vs_reel:
        with st.container(border=True):
            # --- Filtrer les derniers jours pour la visualisation ---
            last_days = 70
            forecast_days = 6
            df_plot = df_result.copy()
            df_plot = df_plot[
                df_plot["ds"]
                >= df_plot["ds"].max() - pd.Timedelta(days=last_days + forecast_days)
            ]

            # Date où commence la prédiction J → J+6
            forecast_start_date = df_plot["ds"].max() - pd.Timedelta(
                days=forecast_days - 1
            )

            # --- Graphique ---
            fig = go.Figure()

            # Intervalle de confiance
            if "yhat_lower" in df_plot.columns and "yhat_upper" in df_plot.columns:
                fig.add_traces(
                    go.Scatter(
                        x=df_plot["ds"],
                        y=df_plot["yhat_upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_traces(
                    go.Scatter(
                        x=df_plot["ds"],
                        y=df_plot["yhat_lower"],
                        fill="tonexty",
                        mode="lines",
                        line=dict(width=0),
                        fillcolor="rgba(0, 176, 246, 0.2)",
                        name="Intervalle de confiance",
                    )
                )

            # Courbe réelle jusqu'à J-1
            historical_mask = df_plot["ds"] < forecast_start_date
            fig.add_trace(
                go.Scatter(
                    x=df_plot.loc[historical_mask, "ds"],
                    y=df_plot.loc[historical_mask, "y"],
                    mode="lines+markers",
                    name="Réel",
                    line=dict(color="orange"),
                )
            )

            # Courbe prédite : historique (si tu veux montrer yhat sur les mêmes dates)
            fig.add_trace(
                go.Scatter(
                    x=df_plot.loc[historical_mask, "ds"],
                    y=df_plot.loc[historical_mask, "yhat"],
                    mode="lines+markers",
                    name="Prévision historique",
                    line=dict(color="deepskyblue"),
                )
            )

            # Courbe prédite : horizon J → J+6
            forecast_mask = df_plot["ds"] >= forecast_start_date
            fig.add_trace(
                go.Scatter(
                    x=df_plot.loc[forecast_mask, "ds"],
                    y=df_plot.loc[forecast_mask, "yhat"],
                    mode="lines+markers",
                    name="Prévision 6 jours",
                    line=dict(color="limegreen", width=3, dash="dash"),
                    marker=dict(size=8),
                )
            )

            # Optionnel : background léger pour horizon J→J+6
            fig.add_vrect(
                x0=forecast_start_date,
                x1=df_plot["ds"].max(),
                fillcolor="rgba(50,205,50,0.1)",
                layer="below",
                line_width=0,
            )

            # Ajustement axes
            y_min = min(df_plot[["y", "yhat", "yhat_lower"]].min()) * 0.95
            y_max = max(df_plot[["y", "yhat", "yhat_upper"]].max()) * 1.05
            fig.update_yaxes(range=[y_min, y_max], title_text="Consommation (kW)")
            fig.update_xaxes(title_text="Date")

            # Layout dark theme + marges réduites
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=5, b=10),
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                font=dict(color="#E2E8F0", size=14),
                height=440,
            )

            st.plotly_chart(fig, width="stretch")
