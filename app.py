import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px

st.set_page_config(page_title="Gestión y Análisis de Dietas", layout="wide")

# --- ESTILO CORPORATIVO ---
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

# --- FUNCIONES AUXILIARES PARA ESCENARIOS ---
def cargar_escenarios():
    if "escenarios_guardados" not in st.session_state:
        if os.path.exists("escenarios_guardados.json"):
            with open("escenarios_guardados.json", "r") as f:
                st.session_state.escenarios_guardados = json.load(f)
        else:
            st.session_state.escenarios_guardados = []
    return st.session_state.escenarios_guardados

def guardar_escenarios(escenarios):
    st.session_state.escenarios_guardados = escenarios
    with open("escenarios_guardados.json", "w") as f:
        json.dump(escenarios, f)

def get_tabla_totales(tabla, nutrientes_seleccionados):
    totales = {
        "Ingrediente": "Total en dieta",
        "% Inclusión": round(tabla["% Inclusión"].sum(), 2),
        "Costo proporcional (USD/kg)": round(tabla["Costo proporcional (USD/kg)"].sum(), 2),
    }
    for nut in nutrientes_seleccionados:
        totales[nut] = round(tabla[nut].sum(), 2)
    return totales

def get_val(esc, nut):
    if "tabla" in esc and nut in esc["tabla"]:
        vals = esc["tabla"][nut]
        return vals[-1] if isinstance(vals, list) and len(vals) > 0 else np.nan
    return np.nan

def get_inclusion(esc, ing):
    if "tabla" in esc and "Ingrediente" in esc["tabla"] and "% Inclusión" in esc["tabla"]:
        ingr_list = esc["tabla"]["Ingrediente"]
        incl_list = esc["tabla"]["% Inclusión"]
        if ingr_list and ing in ingr_list:
            idx = ingr_list.index(ing)
            return incl_list[idx]
    return 0

def comparador_escenarios(escenarios):
    st.header("Comparador de Escenarios Guardados")
    if len(escenarios) < 2:
        st.info("Guarda al menos dos escenarios en el análisis para comparar aquí.")
        return

    opciones = [esc["nombre"] for esc in escenarios]
    seleccionados = st.multiselect(
        "Selecciona escenarios para comparar", opciones, default=opciones[:2], key="comparador_escenarios_main"
    )
    escenarios_sel = [esc for esc in escenarios if esc["nombre"] in seleccionados]
    if len(escenarios_sel) < 2:
        st.info("Selecciona al menos dos escenarios para comparar.")
        return

    pesta1, pesta2, pesta3, pesta4 = st.tabs([
        "Precio por Nutriente Sombra", "Composición Dieta", "Composición Ingredientes", "Dieta Completa"
    ])

    # TAB 1: SHADOW PRICE
    with pesta1:
        st.markdown("#### Precio sombra por nutriente (Shadow Price)")
        nutrientes_disponibles = sorted(list({nut for esc in escenarios_sel for nut in esc["nutrientes"]}))
        shadow_prices = {}
        for nut in nutrientes_disponibles:
            min_price = np.inf
            best_ing = None
            for esc in escenarios_sel:
                for ing in esc["ingredientes"]:
                    idx = esc["ingredientes"].index(ing)
                    row = esc["data_formula"][idx]
                    contenido = pd.to_numeric(row[nut], errors="coerce")
                    precio = row["precio"]
                    if pd.notnull(contenido) and contenido > 0 and pd.notnull(precio):
                        price_per_unit = precio / contenido
                        if price_per_unit < min_price:
                            min_price = price_per_unit
                            best_ing = f'{ing} ({esc["nombre"]})'
            shadow_prices[nut] = (min_price if min_price!=np.inf else np.nan, best_ing)
        unidad = [escenarios_sel[0]["nutrientes"].count(nut) and escenarios_sel[0]["nutrientes"] and escenarios_sel[0]["nutrientes"][0] for nut in nutrientes_disponibles] if escenarios_sel else []
        df_shadow = pd.DataFrame({
            "Nutriente": nutrientes_disponibles,
            "Precio sombra (USD/unidad)": [shadow_prices[nut][0] for nut in nutrientes_disponibles],
            "Ingrediente más barato": [shadow_prices[nut][1] for nut in nutrientes_disponibles],
        })
        st.dataframe(df_shadow.style.format({"Precio sombra (USD/unidad)": "{:.4f}"}), use_container_width=True)
        # CORREGIDO: customdata debe ser una sola lista (usamos np.stack para unir dos columnas)
        fig_shadow = go.Figure()
        fig_shadow.add_trace(go.Bar(
            x=df_shadow["Nutriente"],
            y=df_shadow["Precio sombra (USD/unidad)"],
            text=[f"{v:.4f}" for v in df_shadow["Precio sombra (USD/unidad)"]],
            marker_color='indigo',
            textposition='auto',
            customdata=np.stack([df_shadow.get("Ingrediente más barato", ""), [""]*len(df_shadow)], axis=-1),
            hovertemplate='%{x}<br>Shadow price: %{y:.4f} USD/unidad<br>Mejor ingrediente: %{customdata[0]}<extra></extra>',
        ))
        fig_shadow.update_layout(
            xaxis_title="Nutriente",
            yaxis_title="Precio sombra (USD/unidad)",
            title="Precio sombra por nutriente (Shadow price)",
        )
        st.plotly_chart(fig_shadow, use_container_width=True)
        st.markdown(
            "El precio sombra por nutriente estima el costo mínimo teórico para obtener una unidad de cada nutriente en la dieta, usando el ingrediente más barato en cada caso."
        )

    # TAB 2: COMPOSICIÓN DIETA
    with pesta2:
        nutrientes_disponibles = sorted(list({
            nut for esc in escenarios_sel
            for nut in esc.get("tabla", {}).keys()
            if nut not in ["Ingrediente", "% Inclusión", "Costo proporcional (USD/kg)"]
        }))
        nut_select_2 = st.multiselect(
            "Selecciona nutrientes a comparar",
            nutrientes_disponibles,
            default=nutrientes_disponibles,
            key="nutrientes_comparador_comp_tab"
        )
        if nut_select_2:
            df_nut = pd.DataFrame({
                esc["nombre"]: [get_val(esc, nut) for nut in nut_select_2] for esc in escenarios_sel
            }, index=nut_select_2)
            st.dataframe(df_nut.style.format("{:.2f}"), use_container_width=True)
            fig = go.Figure()
            for esc in escenarios_sel:
                y_vals = [get_val(esc, nut) for nut in nut_select_2]
                fig.add_trace(go.Bar(x=nut_select_2, y=y_vals, name=esc["nombre"]))
            fig.update_layout(barmode='group', xaxis_title="Nutriente", yaxis_title="Valor en dieta")
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: COMPOSICIÓN INGREDIENTES
    with pesta3:
        ingredientes_disponibles = sorted(list({ing for esc in escenarios_sel for ing in esc.get("ingredientes", [])}))
        ing_select = st.multiselect(
            "Selecciona ingredientes a comparar",
            ingredientes_disponibles,
            default=ingredientes_disponibles,
            key="ingredientes_comparador_tab"
        )
        if ing_select:
            df_ing = pd.DataFrame({
                esc["nombre"]: [get_inclusion(esc, ing) for ing in ing_select] for esc in escenarios_sel
            }, index=ing_select)
            st.dataframe(df_ing.style.format("{:.2f}"), use_container_width=True)
            fig = go.Figure()
            for esc in escenarios_sel:
                y_vals = [get_inclusion(esc, ing) for ing in ing_select]
                fig.add_trace(go.Bar(x=ing_select, y=y_vals, name=esc["nombre"]))
            fig.update_layout(barmode='group', xaxis_title="Ingrediente", yaxis_title="% Inclusión")
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: COSTO TOTAL
    with pesta4:
        st.markdown("#### Costo total de cada escenario (USD/tonelada)")
        costos = {esc["nombre"]: esc["costo_total"] for esc in escenarios_sel}
        st.bar_chart(pd.Series(costos))
        st.dataframe(pd.DataFrame({"Costo total (USD/ton)": costos}), use_container_width=True)
        st.markdown("---")
        st.subheader("Resumen rápido de escenarios comparados")
        st.markdown(
            "- Puedes comparar la eficiencia económica y nutricional de cada escenario de forma clara y rápida.\n"
            "- El precio sombra te permite estimar el costo mínimo teórico de cada nutriente en las fórmulas.\n"
            "- La pestaña de composición te permite analizar cómo varían los niveles de nutrientes y la inclusión de ingredientes entre escenarios."
        )

# ============ ANÁLISIS DE DIETA Y ESCENARIOS =============
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

tab1, tab2 = st.tabs(["Análisis y Escenarios", "Comparador de Escenarios"])

with tab1:
    st.header("Análisis y Guardado de Escenarios")
    if df_ing is not None:
        ingredientes_lista = df_ing["Ingrediente"].dropna().unique().tolist()
        ingredientes_seleccionados = st.multiselect(
            "Selecciona tus ingredientes", ingredientes_lista, default=[], key="ingredientes_tab1"
        )

        columnas_fijas = ["Ingrediente", "% Inclusión", "precio"]
        columnas_nut = [col for col in df_ing.columns if col not in columnas_fijas]
        nutrientes_seleccionados = st.multiselect(
            "Selecciona nutrientes a analizar",
            columnas_nut,
            default=columnas_nut[:4] if len(columnas_nut) >= 4 else columnas_nut,
            key="nutrientes_tab1"
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
                    key=f"porc_{ing}_tab1"
                )
            with cols[2]:
                precio_kg = float(fila["precio"]) if "precio" in fila and pd.notnull(fila["precio"]) else 0.0
                precio_mod = st.number_input(
                    f"Precio {ing} (USD/tonelada)",
                    min_value=0.0,
                    max_value=3000.0,
                    value=precio_kg * 1000,
                    step=1.0,
                    key=f"precio_{ing}_tab1"
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

            tabla = df_formula[["Ingrediente", "% Inclusión"]].copy()
            tabla["Costo proporcional (USD/kg)"] = df_formula.apply(
                lambda row: round(row["precio"] * row["% Inclusión"] / 100, 2) if pd.notnull(row["precio"]) else 0,
                axis=1
            )
            for nut in nutrientes_seleccionados:
                tabla[nut] = df_formula.apply(
                    lambda row: round(pd.to_numeric(row[nut], errors="coerce") * row["% Inclusión"] / 100, 2), axis=1
                )
            totales = get_tabla_totales(tabla, nutrientes_seleccionados)
            tabla = pd.concat([tabla, pd.DataFrame([totales])], ignore_index=True)

            def highlight_total(s):
                return ['background-color: #e3ecf7; font-weight: bold' if v == "Total en dieta" else '' for v in s]
            fmt_dict = {col: "{:.2f}".format for col in tabla.columns if col != "Ingrediente"}

            st.subheader("Ingredientes y proporciones de tu dieta (aporte real y costo proporcional)")
            st.dataframe(
                tabla.style.apply(highlight_total, subset=['Ingrediente']).format(fmt_dict),
                use_container_width=True
            )

            color_palette = px.colors.qualitative.Plotly
            color_map = {ing: color_palette[idx % len(color_palette)] for idx, ing in enumerate(ingredientes_lista)}

            subtab1, subtab2, subtab3 = st.tabs([
                "Costo Total por Ingrediente",
                "Aporte por Ingrediente a Nutrientes",
                "Precio Sombra por Nutriente (Shadow Price)"
            ])

            with subtab1:
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
                    text=[f"{c:.2f} USD/ton" for c in costos_ton],
                    textposition='auto',
                    customdata=proporciones,
                    hovertemplate='%{x}<br>Costo: %{y:.2f} USD/ton<br>Proporción: %{customdata:.2f}%<extra></extra>'
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

            with subtab2:
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
                            text=[f"{v:.2f} {unidad}" for v in valores],
                            textposition='auto',
                            customdata=proporciones,
                            hovertemplate='%{x}<br>Aporte: %{y:.2f} ' + (unidad if unidad else '') + '<br>Proporción: %{customdata:.2f}%<extra></extra>',
                        ))
                        fig.update_layout(
                            xaxis_title="Ingrediente",
                            yaxis_title=f"Aporte de {nut} ({unidad})" if unidad else f"Aporte de {nut}",
                            title=f"Aporte de cada ingrediente a {nut} ({unidad})" if unidad else f"Aporte de cada ingrediente a {nut}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"**Total de {nut} en la dieta:** {tabla[nut].iloc[-1]:.2f} {unidad}")

            with subtab3:
                st.markdown("#### Precio sombra por nutriente (Shadow Price)")
                shadow_prices = {}
                for nut in nutrientes_seleccionados:
                    min_price = np.inf
                    best_ing = None
                    for ing in ingredientes_seleccionados:
                        row = df_formula[df_formula["Ingrediente"] == ing].iloc[0]
                        contenido = pd.to_numeric(row[nut], errors="coerce")
                        precio = row["precio"]
                        if pd.notnull(contenido) and contenido > 0 and pd.notnull(precio):
                            price_per_unit = precio / contenido
                            if price_per_unit < min_price:
                                min_price = price_per_unit
                                best_ing = ing
                    shadow_prices[nut] = (min_price if min_price!=np.inf else np.nan, best_ing)
                unidad = [unidades_dict.get(nut, "") for nut in nutrientes_seleccionados]
                df_shadow = pd.DataFrame({
                    "Nutriente": nutrientes_seleccionados,
                    "Precio sombra (USD/unidad)": [shadow_prices[nut][0] for nut in nutrientes_seleccionados],
                    "Ingrediente más barato": [shadow_prices[nut][1] for nut in nutrientes_seleccionados],
                    "Unidad": unidad,
                })
                st.dataframe(df_shadow.style.format({"Precio sombra (USD/unidad)": "{:.4f}"}), use_container_width=True)
                fig_shadow = go.Figure()
                fig_shadow.add_trace(go.Bar(
                    x=df_shadow["Nutriente"],
                    y=df_shadow["Precio sombra (USD/unidad)"],
                    text=[f"{v:.4f}" for v in df_shadow["Precio sombra (USD/unidad)"]],
                    marker_color='indigo',
                    textposition='auto',
                    customdata=np.stack([df_shadow["Unidad"], df_shadow["Ingrediente más barato"]], axis=-1),
                    hovertemplate='%{x}<br>Shadow price: %{y:.4f} USD/%{customdata[0]}<br>Mejor ingrediente: %{customdata[1]}<extra></extra>',
                ))
                fig_shadow.update_layout(
                    xaxis_title="Nutriente",
                    yaxis_title="Precio sombra (USD/unidad)",
                    title="Precio sombra por nutriente (Shadow price)",
                )
                st.plotly_chart(fig_shadow, use_container_width=True)
                st.markdown(
                    "El precio sombra por nutriente estima el costo mínimo teórico para obtener una unidad de cada nutriente en la dieta, usando el ingrediente más barato en cada caso."
                )

            # --- GUARDAR ESCENARIO ---
            st.markdown("---")
            escenarios = cargar_escenarios()
            nombre_escenario = st.text_input("Nombre para guardar este escenario", value="Escenario " + str(len(escenarios)+1), key="nombre_escenario")
            if st.button("Guardar escenario"):
                escenario = {
                    "nombre": nombre_escenario,
                    "ingredientes": ingredientes_seleccionados,
                    "nutrientes": nutrientes_seleccionados,
                    "data_formula": data_formula,
                    "tabla": tabla.to_dict(),
                    "precio_promedio_nutriente": {},  # no usado
                    "costo_total": float(tabla["Costo proporcional (USD/kg)"].iloc[-1]) * 1000  # USD/ton,
                }
                escenarios.append(escenario)
                guardar_escenarios(escenarios)
                st.success(f"Escenario '{nombre_escenario}' guardado exitosamente.")
        else:
            st.info("Selecciona ingredientes y nutrientes para comenzar el análisis y visualización.")

with tab2:
    escenarios = cargar_escenarios()
    comparador_escenarios(escenarios)
