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

st.title("Gestión y Análisis de Dietas")

# ============ ANÁLISIS DE DIETA ÚNICO APARTADO =============
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

        # --- TABLA MEJORADA: COSTO PROPORCIONAL Y SUMA TOTAL ---
        tabla = df_formula[["Ingrediente", "% Inclusión"]].copy()

        tabla["Costo proporcional (USD/kg)"] = df_formula.apply(
            lambda row: round(row["precio"] * row["% Inclusión"] / 100, 2) if pd.notnull(row["precio"]) else 0,
            axis=1
        )

        for nut in nutrientes_seleccionados:
            tabla[nut] = df_formula.apply(
                lambda row: round(pd.to_numeric(row[nut], errors="coerce") * row["% Inclusión"] / 100, 2), axis=1
            )

        totales = {
            "Ingrediente": "Total en dieta",
            "% Inclusión": round(tabla["% Inclusión"].sum(), 2),
            "Costo proporcional (USD/kg)": round(tabla["Costo proporcional (USD/kg)"].sum(), 2),
        }
        for nut in nutrientes_seleccionados:
            totales[nut] = round(tabla[nut].sum(), 2)

        tabla = pd.concat([tabla, pd.DataFrame([totales])], ignore_index=True)

        def highlight_total(s):
            return ['background-color: #e3ecf7; font-weight: bold' if v == "Total en dieta" else '' for v in s]

        # Formato con dos decimales para todas las columnas numéricas
        fmt_dict = {col: "{:.2f}".format for col in tabla.columns if col != "Ingrediente"}

        st.subheader("Ingredientes y proporciones de tu dieta (aporte real y costo proporcional)")
        st.dataframe(
            tabla.style.apply(highlight_total, subset=['Ingrediente']).format(fmt_dict),
            use_container_width=True
        )

        color_palette = px.colors.qualitative.Plotly
        color_map = {ing: color_palette[idx % len(color_palette)] for idx, ing in enumerate(ingredientes_lista)}

        # Orden de pestañas: Costo total por ingrediente, Aporte por ingrediente a nutrientes, Costo por unidad de nutriente
        tab1, tab2, tab3 = st.tabs([
            "Costo Total por Ingrediente",
            "Aporte por Ingrediente a Nutrientes",
            "Costo por Unidad de Nutriente"
        ])

        with tab1:
            st.markdown("#### Costo total aportado por cada ingrediente (USD/tonelada de dieta, proporcional)")
            costos = [
                round((row["precio"] * row["% Inclusión"] / 100), 2) if pd.notnull(row["precio"]) else 0
                for idx, row in df_formula.iterrows()
            ]
            costos_ton = [round(c * 1000, 2) for c in costos]
            total_costo_ton = round(sum(costos_ton), 2)
            proporciones = [round((c / total_costo_ton * 100), 2) if total_costo_ton > 0 else 0 for c in costos_ton]
            fig2 = go.Figure([go.Bar(
                x=ingredientes_seleccionados,
                y=costos_ton,
                marker_color=[color_map[ing] for ing in ingredientes_seleccionados],
                text=[f"{c:.2f} USD/ton<br>{p:.2f}%" for c, p in zip(costos_ton, proporciones)],
                textposition='auto'
            )])
            fig2.update_layout(
                xaxis_title="Ingrediente",
                yaxis_title="Costo aportado (USD/tonelada de dieta)",
                title="Costo total aportado por ingrediente (USD/tonelada)",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(f"**Costo total de la fórmula:** ${total_costo_ton:.2f} USD/tonelada")
            st.markdown("Cada barra muestra el costo y el porcentaje proporcional de cada ingrediente respecto al costo total de la dieta.")

        with tab2:
            st.markdown("#### Aporte de cada ingrediente a cada nutriente (barras por nutriente)")
            nut_tabs = st.tabs([nut for nut in nutrientes_seleccionados])
            for i, nut in enumerate(nutrientes_seleccionados):
                with nut_tabs[i]:
                    valores = []
                    for ing in ingredientes_seleccionados:
                        valor = pd.to_numeric(df_formula.loc[df_formula["Ingrediente"] == ing, nut], errors="coerce").values[0]
                        porc = df_formula[df_formula["Ingrediente"] == ing]["% Inclusión"].values[0]
                        aporte = round((valor * porc) / 100, 2) if pd.notnull(valor) else 0
                        valores.append(aporte)
                    unidad = unidades_dict.get(nut, "")
                    total_nut = tabla[nut].iloc[-1] if tabla[nut].iloc[-1] != 0 else 1
                    proporciones = [round(v / total_nut * 100, 2) for v in valores]
                    fig = go.Figure()
    fig.add_trace(go.Bar(
    x=ingredientes_seleccionados,
    y=valores,
    marker_color=[color_map[ing] for ing in ingredientes_seleccionados],
    text=[f"{p:.2f}%" for p in proporciones],  # SOLO porcentaje en la barra
    textposition='auto',
    hovertemplate='%{x}<br>Aporte: %{y:.2f} ' + (unidad if unidad else '') + '<br>Proporción: %{text}<extra></extra>'
))
fig.update_layout(
    xaxis_title="Ingrediente",
    yaxis_title=f"Aporte de {nut} ({unidad})" if unidad else f"Aporte de {nut}",
    title=f"Aporte de cada ingrediente a {nut} ({unidad})" if unidad else f"Aporte de cada ingrediente a {nut}"
)

        with tab3:
            st.markdown("#### Costo por unidad de nutriente aportada (USD/tonelada por unidad de nutriente)")
            nut_tabs = st.tabs([nut for nut in nutrientes_seleccionados])
            for i, nut in enumerate(nutrientes_seleccionados):
                with nut_tabs[i]:
                    costos_unit = []
                    for ing in ingredientes_seleccionados:
                        row = df_formula[df_formula["Ingrediente"] == ing].iloc[0]
                        aporte = pd.to_numeric(row[nut], errors="coerce")
                        aporte = round((aporte * row["% Inclusión"]) / 100, 2) if pd.notnull(aporte) else 0
                        costo = round((row["precio"] * row["% Inclusión"] / 100), 2) if pd.notnull(row["precio"]) else 0
                        costo_unitario = round((costo / aporte), 2) if aporte > 0 else np.nan
                        costo_unitario_ton = round(costo_unitario * 1000, 2) if not np.isnan(costo_unitario) else np.nan
                        costos_unit.append(costo_unitario_ton)
                    unidad = unidades_dict.get(nut, "")
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        x=ingredientes_seleccionados,
                        y=costos_unit,
                        marker_color=[color_map[ing] for ing in ingredientes_seleccionados],
                        text=[f"{c:.2f}" if not np.isnan(c) else "-" for c in costos_unit],
                        textposition='auto'
                    ))
                    fig3.update_layout(
                        xaxis_title="Ingrediente",
                        yaxis_title=f"Costo por unidad de {nut} (USD/ton por {unidad})" if unidad else f"Costo por unidad de {nut} (USD/ton)",
                        title=f"Costo por unidad de {nut} (USD/ton por {unidad})" if unidad else f"Costo por unidad de {nut} (USD/ton)"
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        - Puedes modificar los precios de los ingredientes y ver el impacto instantáneamente.
        - Selecciona los nutrientes que más te interesan para un análisis enfocado.
        - Las pestañas te permiten comparar visualmente: el costo total por ingrediente (proporcional), el aporte por nutriente y el costo por unidad de nutriente.
        - **Recuerda:** El precio de cada ingrediente se ingresa y visualiza en USD por tonelada (USD/ton). Los cálculos internos y los resultados de costo total se muestran en USD/tonelada en los gráficos y tablas.
        """)

    else:
        st.info("Selecciona ingredientes y nutrientes para comenzar el análisis y visualización.")
