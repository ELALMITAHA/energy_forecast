import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from streamlit_extras.metric_cards import style_metric_cards


def display_results_ux(df_result: pd.DataFrame, metrics: dict):
    # ============================================================
    # TITLE
    # ============================================================
    st.title("Electricity Consumption Forecast — Bordeaux")


            # ------------------------------------------------------------
        # EXPLANATION MESSAGE (IMPORTANT)
        # ------------------------------------------------------------
    st.warning(
            "⚠️ Forecasting beyond **January 8th, 2026** is temporarily disabled.\n\n"
            "The official electricity consumption Open Data source is no longer updated, "
            "while weather data continues to refresh. "
            "To prevent temporal misalignment, only historical predictions are displayed."
    )

    # ============================================================
    # METRICS + TABLE
    # ============================================================
    col_model_wap, col_plot = st.columns((2, 5))

    
    with col_model_wap:
        col_model, col_wape = st.columns(2)
        st.metric("Baseline","Naive 7 days")
        
        with col_model:
            model_name = metrics.get("model", "Prophet")
            st.metric("Model", model_name.upper())
            
            st.metric("Window size",60)

        with col_wape:
            
            st.metric("MAE",  "85 002 kwh")
            st.metric("MASE","54%")

        style_metric_cards(
            background_color="#111827",
            border_color="#1F2937",
            border_left_color="#3B82F6",
            border_size_px=5,
            border_radius_px=14,
            box_shadow=True,
        )

    # ============================================================
    # DATA PREPARATION FOR VISUALIZATION
    # ============================================================
    # Last valid real observation
    last_real_date = df_result.loc[
        df_result["daily_conso_kwh"].notna(), "ds"
    ].max()

    # Filter strictly to valid historical range
    df_plot = df_result[df_result["ds"] <= last_real_date].copy()

    # Keep last N days for readability
    last_days = 70
    df_plot = df_plot[
        df_plot["ds"] >= df_plot["ds"].max() - pd.Timedelta(days=last_days)
    ]

    # ============================================================
    # PLOT
    # ============================================================
    with col_plot:
        with st.container(border=True):
            fig = go.Figure()

            # -----------------------------
            # Confidence interval (historical)
            # -----------------------------
            if {"yhat_lower", "yhat_upper"}.issubset(df_plot.columns):
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["ds"],
                        y=df_plot["yhat_upper"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["ds"],
                        y=df_plot["yhat_lower"],
                        fill="tonexty",
                        mode="lines",
                        line=dict(width=0),
                        fillcolor="rgba(0,176,246,0.2)",
                        name="Confidence interval",
                    )
                )

            # -----------------------------
            # Observed consumption
            # -----------------------------
            fig.add_trace(
                go.Scatter(
                    x=df_plot["ds"],
                    y=df_plot["daily_conso_kwh"],
                    mode="lines+markers",
                    name="Observed consumption",
                    line=dict(color="orange"),
                )
            )

            # -----------------------------
            # Model prediction (historical)
            # -----------------------------
            fig.add_trace(
                go.Scatter(
                    x=df_plot["ds"],
                    y=df_plot["yhat"],
                    mode="lines",
                    name="Model prediction (historical)",
                    line=dict(color="deepskyblue"),
                )
            )

            # -----------------------------
            # Axes & layout
            # -----------------------------
            y_min = (
                df_plot[["daily_conso_kwh", "yhat", "yhat_lower"]]
                .min()
                .min()
                * 0.95
            )
            y_max = (
                df_plot[["daily_conso_kwh", "yhat", "yhat_upper"]]
                .max()
                .max()
                * 1.05
            )

            fig.update_yaxes(
                range=[y_min, y_max],
                title_text="Daily electricity consumption (kWh)",
            )
            fig.update_xaxes(title_text="Date")

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                font=dict(color="#E2E8F0", size=14),
                height=440,
            )

            st.plotly_chart(fig, use_container_width=True)

