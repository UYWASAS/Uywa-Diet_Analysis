import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Gestión y Análisis de Dietas", layout="wide")

# --- FONDO CORPORATIVO Y SIDEBAR LEGIBLE ---
st.markdown("""
    <style>
    html, body, .stApp, .main, .block-container {
        background: linear-gradient(120deg, #f3f6fa 0%, #e3ecf7 100%) !important;
        background-color: #f3f6fa !important;
    }
    section[data-testid="stSidebar"] {
        background: #19345c !important;
        color: #fff !important;
    }
    section[data-testid="stSidebar"] * {
        color: #fff !important;
    }
    .block-container {
        background: transparent !important;
    }
    section.main {
        background: transparent !important;
    }
    .stFileUploader, .stMultiSelect, .stSelectbox, .stNumberInput, .stTextInput {
        background-color: #f4f8fa !important;
        border-radius: 6px !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* Sidebar radio buttons y labels */
    .stRadio label, .stRadio div[role="radiogroup"] label, .stRadio div[role="radiogroup"] span {
        color: #fff !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    .stRadio div[role="radiogroup"] > div {
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("nombre_archivo_logo.png", width=110)
    st.markdown(
        """
        <div style='text-align: center; margin-bottom:10px;'>
            <div style='font-size:32px;font-family:Montserrat,Arial;color:#fff; margin-top: 10px;letter-spacing:1px; font-weight:700; line-height:1.1;'>
                UYWA-<br>NUTRITION<sup>®</sup>
            </div>
            <div style='font-size:16px;color:#fff; margin-top: 5px; font-family:Montserrat,Arial; line-height: 1.1;'>
                Nutrición de Precisión Basada en Evidencia
            </div>
            <hr style='border-top:1px solid #2e4771; margin: 18px 0;'>
            <div style='font-size:14px;color:#fff; margin-top: 8px;'>
                <b>Contacto:</b> uywasas@gmail.com<br>
                Derechos reservados © 2025
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Selecciona una sección",
        [
            "Análisis de Dieta",
            "Simulador Productivo",
            "Simulador Económico",
            "Comparador de Escenarios"
        ],
        key="menu_radio"
    )

st.title("Gestión y Análisis de Dietas")

# ============ ANÁLISIS DE DIETA MEJORADO =============
if menu == "Análisis de Dieta":
    archivo_excel = "Ingredientes1.xlsx"
    df_ing = None
    unidades_dict = {}
    if os.path.exists(archivo_excel):
        df_full = pd.read_excel(archivo_excel, header=None)
        headers = df_full.iloc[0].values
        unidades = df_full.iloc[1].values
        data = df_full.iloc[2:].copy()
        data.columns = headers
        df_ing = data.reset_index(drop=True)
        unidades_dict = {headers[i]: unidades[i] for i in range(len(headers))}
    else:
        archivo_subido = st.file_uploader("Sube tu archivo de ingredientes (.xlsx)", type=["xlsx"])
        if archivo_subido is not None:
            df_full = pd.read_excel(archivo_subido, header=None)
            headers = df_full.iloc[0].values
            unidades = df_full.iloc[1].values
            data = df_full.iloc[2:].copy()
            data.columns = headers
            df_ing = data.reset_index(drop=True)
            unidades_dict = {headers[i]: unidades[i] for i in range(len(headers))}
        else:
            st.warning("No se encontró el archivo Ingredientes1.xlsx. Sube uno para continuar.")

    if df_ing is not None:
        ingredientes_lista = df_ing["Ingrediente"].dropna().unique().tolist()
        ingredientes_seleccionados = st.multiselect(
            "Selecciona tus ingredientes", ingredientes_lista, default=[]
        )

        columnas_fijas = ["Ingrediente", "% Inclusión", "precio"]
        columnas_nut = [col for col in df_ing.columns if col not in columnas_fijas]
        nutrientes_seleccionados = st.multiselect(
            "Selecciona nutrientes a analizar",
            columnas_nut,
            default=columnas_nut[:4] if len(columnas_nut) >= 4 else columnas_nut
        )

        data_formula = []
        total_inclusion = 0
        st.markdown("### Ajusta % inclusión y precio (USD/tonelada) de cada ingrediente")
        for ing in ingredientes_seleccionados:
            fila = df_ing[df_ing["Ingrediente"] == ing].iloc[0].to_dict()
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.write(f"**{ing}**")
            with cols[1]:
                porcentaje = st.number_input(
                    f"% inclusión para {ing}",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1,
                    key=f"porc_{ing}"
                )
            with cols[2]:
                precio_kg = float(fila["precio"]) if "precio" in fila and pd.notnull(fila["precio"]) else 0.0
                precio_mod = st.number_input(
                    f"Precio {ing} (USD/tonelada)",
                    min_value=0.0,
                    max_value=3000.0,
                    value=precio_kg * 1000,
                    step=1.0,
                    key=f"precio_{ing}"
                )
            total_inclusion += porcentaje
            fila["% Inclusión"] = porcentaje
            fila["precio"] = precio_mod / 1000  # Guardamos en USD/kg para cálculos
            data_formula.append(fila)

        st.markdown(f"#### Suma total de inclusión: **{total_inclusion:.2f}%**")
        if abs(total_inclusion - 100) > 0.01:
            st.warning("La suma de los ingredientes no es 100%. Puede afectar el análisis final.")

        if ingredientes_seleccionados and nutrientes_seleccionados:
            df_formula = pd.DataFrame(data_formula)

            # 1. Construcción de la tabla con aporte real
            tabla = df_formula[["Ingrediente", "% Inclusión", "precio"]].copy()
            for nut in nutrientes_seleccionados:
                tabla[nut] = df_formula.apply(
                    lambda row: pd.to_numeric(row[nut], errors="coerce") * row["% Inclusión"] / 100, axis=1
                )

            # 2. Fila total con la suma final del nutriente en la dieta
            totales = {
                "Ingrediente": "Total en dieta",
                "% Inclusión": tabla["% Inclusión"].sum(),
                "precio": np.nan,
            }
            for nut in nutrientes_seleccionados:
                totales[nut] = tabla[nut].sum()

            tabla = pd.concat([tabla, pd.DataFrame([totales])], ignore_index=True)

            # 3. Mejor formato visual: resaltar total
            def highlight_total(s):
                return ['background-color: #e3ecf7; font-weight: bold' if v == "Total en dieta" else '' for v in s]
            st.subheader("Ingredientes y proporciones de tu dieta (aporte real por nutriente)")
            st.dataframe(
                tabla.style.apply(highlight_total, subset=['Ingrediente']).format(precision=4),
                use_container_width=True
            )

            # 4. (Opcional) Gráfico de barras de aporte por nutriente con referencia al total
            for nut in nutrientes_seleccionados:
                st.markdown(f"##### Aporte de {nut} por ingrediente")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=tabla["Ingrediente"][:-1],
                    y=tabla[nut][:-1],
                    name=f"Aporte {nut}"
                ))
                fig.add_trace(go.Scatter(
                    x=["Total en dieta"],
                    y=[tabla[nut].iloc[-1]],
                    mode="markers+text",
                    marker=dict(size=14, color="red"),
                    text=["Total"],
                    textposition="top center",
                    name="Total dieta"
                ))
                fig.update_layout(
                    xaxis_title="Ingrediente",
                    yaxis_title=f"Aporte de {nut} por kg de dieta",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Selecciona ingredientes y nutrientes para comenzar el análisis y visualización.")
