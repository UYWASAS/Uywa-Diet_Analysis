import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import requests

# ---- Configuración de Hugging Face Inference API ----
HF_TOKEN = st.secrets["HF_TOKEN"]
HF_MODEL_URL = "https://api-inference.huggingface.co/models/gpt2"  # Puedes cambiar el modelo aquí (por ejemplo, mistralai/Mistral-7B-Instruct-v0.2)
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def analizar_escenario_con_ia(prompt, datos_escenario):
    texto = f"{prompt}\n\nDatos del escenario:\n{datos_escenario}"
    payload = {"inputs": texto, "parameters": {"max_length": 200}}
    try:
        response = requests.post(HF_MODEL_URL, headers=HF_HEADERS, json=payload, timeout=30)
        # --- DEBUG: muestra la respuesta cruda para diagnosticar ---
        if response.status_code != 200:
            return f"Error Hugging Face (status {response.status_code}): {response.text}"
        try:
            respuesta = response.json()
        except Exception:
            return f"Respuesta no válida de Hugging Face: {response.text}"
        if isinstance(respuesta, list) and 'generated_text' in respuesta[0]:
            return respuesta[0]['generated_text']
        elif isinstance(respuesta, dict) and 'generated_text' in respuesta:
            return respuesta['generated_text']
        elif 'error' in respuesta:
            return f"Error Hugging Face: {respuesta['error']}"
        else:
            return "No se pudo procesar la respuesta de Hugging Face."
    except Exception as e:
        return f"Error de conexión Hugging Face: {e}"

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

def fmt2(x):
    try:
        return round(float(x), 2)
    except:
        return x

def fmt2_df(df):
    return df.applymap(fmt2)

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

# --- FUNCIONES AUXILIARES ---
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
        "% Inclusión": fmt2(tabla["% Inclusión"].sum()),
        "Costo proporcional (USD/kg)": fmt2(tabla["Costo proporcional (USD/kg)"].sum()),
    }
    for nut in nutrientes_seleccionados:
        totales[nut] = fmt2(tabla[nut].sum())
    return totales

def get_val(esc, nut):
    if "tabla" in esc and nut in esc["tabla"]:
        vals = esc["tabla"][nut]
        return fmt2(vals[-1]) if isinstance(vals, list) and len(vals) > 0 else np.nan
    return np.nan

def get_inclusion(esc, ing):
    if "tabla" in esc and "Ingrediente" in esc["tabla"] and "% Inclusión" in esc["tabla"]:
        ingr_list = esc["tabla"]["Ingrediente"]
        incl_list = esc["tabla"]["% Inclusión"]
        if ingr_list and ing in ingr_list:
            idx = ingr_list.index(ing)
            return fmt2(incl_list[idx])
    return 0

def unit_selector(label, options, default, key):
    return st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)

def format_price(value, factor):
    try:
        return fmt2(float(value) * factor)
    except:
        return value

def format_label(unit, factor):
    if factor == 1:
        return f"USD/{unit}"
    elif factor == 100:
        return f"USD/100{unit}"
    elif factor == 1000:
        return f"USD/1000{unit}" if unit != 'kg' else "USD/ton"
    else:
        return f"USD/{factor}{unit}"

def get_unit_factor(unit, manual_unit):
    if manual_unit.lower() in ['unidad', unit]:
        return 1, format_label(unit, 1)
    elif manual_unit.lower().startswith('100'):
        return 100, format_label(unit, 100)
    elif manual_unit.lower().startswith('1000') or manual_unit.lower() == 'ton':
        return 1000, format_label(unit, 1000)
    elif manual_unit.lower() == 'kg':
        return 1, f"USD/kg"
    elif manual_unit.lower() == '100g':
        return 100, "USD/100g"
    elif manual_unit.lower() == '1000kcal':
        return 1000, "USD/1000kcal"
    return 1, f"USD/{unit}"

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
    resumen_comparador = ""
    with pesta1:
        st.markdown("#### Precio sombra por nutriente (Shadow Price)")
        nutrientes_disponibles = sorted(list({nut for esc in escenarios_sel for nut in esc["nutrientes"]}))
        unidades_dict = {}
        for esc in escenarios_sel:
            for nut in esc["nutrientes"]:
                if nut not in unidades_dict:
                    unidades_dict[nut] = esc.get("unidades_dict", {}).get(nut, "unidad")
        unit_options = {
            'kg': ['kg', 'ton'],
            'g': ['g', '100g', 'kg', 'ton'],
            'kcal': ['kcal', '1000kcal'],
            '%': ['%', '100 unidades'],
            'unidad': ['unidad', '100 unidades', '1000 unidades', 'kg', 'ton'],
        }
        shadow_tab = st.tabs(nutrientes_disponibles)
        for idx, nut in enumerate(nutrientes_disponibles):
            with shadow_tab[idx]:
                ingredientes = []
                precios = []
                contenidos = []
                precios_ing = []
                esc_names = []
                for esc in escenarios_sel:
                    for i, ing in enumerate(esc["ingredientes"]):
                        row = esc["data_formula"][i]
                        contenido = fmt2(pd.to_numeric(row.get(nut, 0), errors="coerce"))
                        precio = fmt2(row.get("precio", np.nan))
                        if pd.notnull(contenido) and contenido > 0 and pd.notnull(precio):
                            ingredientes.append(f"{ing} ({esc['nombre']})")
                            contenidos.append(contenido)
                            precios_ing.append(precio)
                            esc_names.append(esc["nombre"])
                unit = unidades_dict.get(nut, "unidad")
                manual_unit = unit_selector(
                    f"Unidad para {nut}",
                    unit_options.get(unit, ["unidad", "100 unidades", "1000 unidades", "kg", "ton"]),
                    unit_options.get(unit, ["unidad"])[0],
                    key=f"unit_selector_{nut}_shadow"
                )
                factor, label = get_unit_factor(unit, manual_unit)
                precios_unit = [fmt2(precio / contenido * factor) if contenido > 0 else np.nan for precio, contenido in zip(precios_ing, contenidos)]
                df_shadow = pd.DataFrame({
                    "Ingrediente": ingredientes,
                    f"Precio por {manual_unit}": precios_unit,
                    f"Contenido de {nut} por kg": contenidos,
                    "Precio ingrediente (USD/kg)": precios_ing,
                    "Escenario": esc_names,
                })
                min_idx = df_shadow[f"Precio por {manual_unit}"].idxmin()
                df_shadow["Es el más barato"] = ["✅" if i == min_idx else "" for i in df_shadow.index]
                bar_colors = ['green' if i == min_idx else 'royalblue' for i in range(len(df_shadow))]
                fig_shadow = go.Figure()
                fig_shadow.add_trace(go.Bar(
                    x=df_shadow["Ingrediente"],
                    y=df_shadow[f"Precio por {manual_unit}"],
                    marker_color=bar_colors,
                    text=[f"{fmt2(v)}" for v in df_shadow[f"Precio por {manual_unit}"]],
                    textposition='auto',
                    customdata=np.stack([df_shadow["Escenario"], df_shadow["Es el más barato"]], axis=-1),
                    hovertemplate='%{x}<br>Escenario: %{customdata[0]}<br>Precio sombra: %{y:.2f} {label}<br>%{customdata[1]}<extra></extra>',
                ))
                fig_shadow.update_layout(
                    xaxis_title="Ingrediente",
                    yaxis_title=label,
                    title=f"Precio sombra y costo por ingrediente para {nut}",
                )
                st.plotly_chart(fig_shadow, use_container_width=True)
                st.dataframe(fmt2_df(df_shadow), use_container_width=True)
                st.markdown(
                    f"**El precio sombra de {nut} es el menor costo posible para obtener una unidad de este nutriente usando el ingrediente más barato en cada fórmula.**\n\n"
                    f"- Puedes ajustar la unidad para mejorar la visualización.\n"
                    f"- El ingrediente marcado con ✅ aporta el precio sombra."
                )
                resumen_comparador += f"\n\nShadow price {nut}:\n{df_shadow.to_csv(index=False)}"

        st.markdown("---")
        st.markdown("### Análisis automático con IA")
        prompt_shadow = st.text_area("Consulta para IA sobre shadow price", value="Genera un análisis experto del precio sombra y su relevancia en estos escenarios.")
        if st.button("Analizar con IA - Shadow Price"):
            respuesta_ia = analizar_escenario_con_ia(prompt_shadow, resumen_comparador)
            st.markdown(f"**IA:** {respuesta_ia}")

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
        unidades_dict = {}
        for esc in escenarios_sel:
            for nut in esc.get("tabla", {}).keys():
                if nut not in unidades_dict:
                    unidades_dict[nut] = esc.get("unidades_dict", {}).get(nut, "unidad")
        unit_options = {
            'kg': ['kg', 'ton'],
            'g': ['g', '100g', 'kg', 'ton'],
            'kcal': ['kcal', '1000kcal'],
            '%': ['%', '100 unidades'],
            'unidad': ['unidad', '100 unidades', '1000 unidades', 'kg', 'ton'],
        }
        nut_tabs = st.tabs(nut_select_2)
        resumen_comparador = ""
        for i, nut in enumerate(nut_select_2):
            with nut_tabs[i]:
                unit = unidades_dict.get(nut, "unidad")
                manual_unit = unit_selector(
                    f"Unidad para {nut}",
                    unit_options.get(unit, ["unidad", "100 unidades", "1000 unidades", "kg", "ton"]),
                    unit_options.get(unit, ["unidad"])[0],
                    key=f"unit_selector_{nut}_comp"
                )
                factor, label = get_unit_factor(unit, manual_unit)
                valores = []
                esc_names = []
                for esc in escenarios_sel:
                    val = get_val(esc, nut)
                    val = fmt2(val * factor) if pd.notnull(val) else np.nan
                    valores.append(val)
                    esc_names.append(esc["nombre"])
                df_nut = pd.DataFrame({
                    "Escenario": esc_names,
                    f"{nut} ({manual_unit})": valores,
                })
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=esc_names,
                    y=valores,
                    marker_color='dodgerblue',
                    text=[f"{fmt2(v)}" for v in valores],
                    textposition='auto',
                    hovertemplate=f'%{{x}}<br>{nut}: %{{y:.2f}} {manual_unit}<extra></extra>',
                ))
                fig.update_layout(
                    xaxis_title="Escenario",
                    yaxis_title=f"{nut} ({manual_unit})",
                    title=f"Comparación de {nut} en cada escenario",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(fmt2_df(df_nut), use_container_width=True)
                st.markdown(
                    f"Puedes ajustar la unidad para comparar valores en la escala más adecuada para cada nutriente."
                )
                resumen_comparador += f"\n\nComparación nutriente {nut}:\n{df_nut.to_csv(index=False)}"

        st.markdown("---")
        st.markdown("### Análisis automático con IA")
        prompt_comp = st.text_area("Consulta para IA sobre la composición y comparación de nutrientes", value="¿Qué escenarios resultan más óptimos y por qué?")
        if st.button("Analizar con IA - Composición Dieta"):
            respuesta_ia = analizar_escenario_con_ia(prompt_comp, resumen_comparador)
            st.markdown(f"**IA:** {respuesta_ia}")

    with pesta3:
        ingredientes_disponibles = sorted(list({ing for esc in escenarios_sel for ing in esc.get("ingredientes", [])}))
        ing_select = st.multiselect(
            "Selecciona ingredientes a comparar",
            ingredientes_disponibles,
            default=ingredientes_disponibles,
            key="ingredientes_comparador_tab"
        )
        unit_options = {
            '%': ['%', '100 unidades'],
            'unidad': ['unidad', '100 unidades', '1000 unidades'],
        }
        manual_unit = unit_selector(
            "Unidad para inclusión",
            unit_options.get('%', ['%']),
            '%',
            key="unit_selector_inclusion_comp"
        )
        factor, label = get_unit_factor('%', manual_unit)
        resumen_comparador = ""
        if ing_select:
            esc_names = []
            valores = {esc["nombre"]: [] for esc in escenarios_sel}
            for esc in escenarios_sel:
                for ing in ing_select:
                    val = get_inclusion(esc, ing)
                    val = fmt2(val * factor)
                    valores[esc["nombre"]].append(val)
            df_ing = pd.DataFrame(valores, index=ing_select)
            fig = go.Figure()
            for esc in escenarios_sel:
                fig.add_trace(go.Bar(
                    x=ing_select,
                    y=valores[esc["nombre"]],
                    name=esc["nombre"],
                    text=[f"{fmt2(v)}" for v in valores[esc["nombre"]]],
                    textposition='auto',
                    hovertemplate=f'%{{x}}<br>Inclusión: %{{y:.2f}} {manual_unit}<extra></extra>',
                ))
            fig.update_layout(
                barmode='group',
                xaxis_title="Ingrediente",
                yaxis_title=f"% Inclusión ({manual_unit})",
                title="Comparación de inclusión de ingredientes entre escenarios"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fmt2_df(df_ing), use_container_width=True)
            st.markdown("Puedes ajustar la unidad para mostrar la inclusión de ingredientes en la escala más adecuada.")
            resumen_comparador += f"\n\nComparación de inclusión de ingredientes:\n{df_ing.to_csv(index=False)}"

        st.markdown("---")
        st.markdown("### Análisis automático con IA")
        prompt_ing = st.text_area("Consulta para IA sobre la comparación de ingredientes", value="¿Qué ingredientes marcan diferencias clave entre los escenarios?")
        if st.button("Analizar con IA - Composición Ingredientes"):
            respuesta_ia = analizar_escenario_con_ia(prompt_ing, resumen_comparador)
            st.markdown(f"**IA:** {respuesta_ia}")

    with pesta4:
        st.markdown("#### Costo total de cada escenario (USD/tonelada)")
        costos = {esc["nombre"]: fmt2(esc["costo_total"]) for esc in escenarios_sel}
        st.bar_chart(pd.Series(costos))
        st.dataframe(fmt2_df(pd.DataFrame({"Costo total (USD/ton)": costos})), use_container_width=True)
        st.markdown("---")
        st.subheader("Resumen rápido de escenarios comparados")
        st.markdown(
            "- Puedes comparar la eficiencia económica y nutricional de cada escenario de forma clara y rápida.\n"
            "- El precio sombra te permite estimar el costo mínimo teórico de cada nutriente en las fórmulas.\n"
            "- Las pestañas permiten ajustar la unidad y comparar valores en la escala más útil para tu análisis."
        )
        prompt_final = st.text_area("Consulta para IA sobre el resumen de escenarios comparados", value="Genera un informe comparativo y recomendaciones finales según los escenarios presentados.")
        resumen_final = f"Resumen comparativo de costos:\n{pd.DataFrame({'Costo total (USD/ton)':costos}).to_csv(index=True)}"
        if st.button("Analizar con IA - Resumen Comparador"):
            respuesta_ia = analizar_escenario_con_ia(prompt_final, resumen_final)
            st.markdown(f"**IA:** {respuesta_ia}")

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
                    value=fmt2(precio_kg * 1000),
                    step=1.0,
                    key=f"precio_{ing}_tab1"
                )
            total_inclusion += porcentaje
            fila["% Inclusión"] = fmt2(porcentaje)
            fila["precio"] = fmt2(precio_mod / 1000)
            data_formula.append(fila)

        st.markdown(f"#### Suma total de inclusión: **{fmt2(total_inclusion)}%**")
        if abs(total_inclusion - 100) > 0.01:
            st.warning("La suma de los ingredientes no es 100%. Puede afectar el análisis final.")

        if ingredientes_seleccionados and nutrientes_seleccionados:
            df_formula = pd.DataFrame(data_formula)

            tabla = df_formula[["Ingrediente", "% Inclusión"]].copy()
            tabla["Costo proporcional (USD/kg)"] = df_formula.apply(
                lambda row: fmt2(row["precio"] * row["% Inclusión"] / 100) if pd.notnull(row["precio"]) else 0,
                axis=1
            )
            for nut in nutrientes_seleccionados:
                tabla[nut] = df_formula.apply(
                    lambda row: fmt2(pd.to_numeric(row[nut], errors="coerce") * row["% Inclusión"] / 100), axis=1
                )
            totales = get_tabla_totales(tabla, nutrientes_seleccionados)
            tabla = pd.concat([tabla, pd.DataFrame([totales])], ignore_index=True)

            def highlight_total(s):
                return ['background-color: #e3ecf7; font-weight: bold' if v == "Total en dieta" else '' for v in s]
            fmt_dict = {col: "{:.2f}".format for col in tabla.columns if col != "Ingrediente"}

            st.subheader("Ingredientes y proporciones de tu dieta (aporte real y costo proporcional)")
            st.dataframe(
                fmt2_df(tabla).style.apply(highlight_total, subset=['Ingrediente']).format(fmt_dict),
                use_container_width=True
            )

            color_palette = px.colors.qualitative.Plotly
            color_map = {ing: color_palette[idx % len(color_palette)] for idx, ing in enumerate(ingredientes_lista)}

            subtab1, subtab2, subtab3 = st.tabs([
                "Costo Total por Ingrediente",
                "Aporte por Ingrediente a Nutrientes",
                "Precio Sombra por Nutriente (Shadow Price)"
            ])
            resumen_escenario = ""

            with subtab1:
                manual_unit = unit_selector(
                    "Unidad para mostrar el costo total por ingrediente",
                    ['USD/kg', 'USD/ton'],
                    'USD/ton',
                    key="unit_selector_costototal_tab1"
                )
                factor = 1 if manual_unit == 'USD/kg' else 1000
                label = manual_unit
                costos = [
                    fmt2((row["precio"] * row["% Inclusión"] / 100) * factor) if pd.notnull(row["precio"]) else 0
                    for idx, row in df_formula.iterrows()
                ]
                total_costo = fmt2(sum(costos))
                suma_inclusion = fmt2(sum(df_formula["% Inclusión"]))
                proporciones = [
                    fmt2(100 * row["% Inclusión"] / suma_inclusion) if suma_inclusion > 0 else 0
                    for idx, row in df_formula.iterrows()
                ]
                fig2 = go.Figure([go.Bar(
                    x=ingredientes_seleccionados,
                    y=costos,
                    marker_color=[color_map[ing] for ing in ingredientes_seleccionados],
                    text=[f"{fmt2(c)} {label}" for c in costos],
                    textposition='auto',
                    customdata=proporciones,
                    hovertemplate='%{x}<br>Costo: %{y:.2f} {label}<br>Proporción dieta: %{customdata:.2f}%<extra></extra>'
                )])
                fig2.update_layout(
                    xaxis_title="Ingrediente",
                    yaxis_title=f"Costo aportado ({label})",
                    title=f"Costo total aportado por ingrediente ({label})",
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)
                df_costos = pd.DataFrame({
                    "Ingrediente": ingredientes_seleccionados,
                    f"Costo aportado ({label})": costos,
                    "% Inclusión": [fmt2(row["% Inclusión"]) for idx, row in df_formula.iterrows()],
                    "Proporción dieta (%)": proporciones,
                    "Precio ingrediente (USD/kg)": [fmt2(row["precio"]) for idx, row in df_formula.iterrows()],
                })
                st.dataframe(fmt2_df(df_costos), use_container_width=True)
                st.markdown(f"**Costo total de la fórmula:** {fmt2(total_costo)} {label} (suma de los ingredientes). Puedes cambiar la unidad.")
                resumen_escenario += f"\n\nCosto total por ingrediente:\n{df_costos.to_csv(index=False)}"

                st.markdown("---")
                st.markdown("### Análisis automático con IA")
                prompt_esc1 = st.text_area("Consulta para IA sobre el costo total por ingrediente", value="Analiza la eficiencia económica y posibles mejoras en la selección de ingredientes de este escenario.")
                if st.button("Analizar con IA - Costo Total Ingrediente"):
                    respuesta_ia = analizar_escenario_con_ia(prompt_esc1, resumen_escenario)
                    st.markdown(f"**IA:** {respuesta_ia}")

            with subtab2:
                unit_options = {
                    'kg': ['kg', 'ton'],
                    'g': ['g', '100g', 'kg', 'ton'],
                    'kcal': ['kcal', '1000kcal'],
                    '%': ['%', '100 unidades'],
                    'unidad': ['unidad', '100 unidades', '1000 unidades', 'kg', 'ton'],
                }
                nut_tabs = st.tabs([nut for nut in nutrientes_seleccionados])
                for i, nut in enumerate(nutrientes_seleccionados):
                    with nut_tabs[i]:
                        unit = unidades_dict.get(nut, "unidad")
                        manual_unit = unit_selector(
                            f"Unidad para {nut}",
                            unit_options.get(unit, ["unidad", "100 unidades", "1000 unidades", "kg", "ton"]),
                            unit_options.get(unit, ["unidad"])[0],
                            key=f"unit_selector_{nut}_aporte_tab1"
                        )
                        factor, label = get_unit_factor(unit, manual_unit)
                        valores = []
                        porc_aporte = []
                        total_nut = fmt2(sum([
                            fmt2(pd.to_numeric(df_formula.loc[df_formula["Ingrediente"] == ing, nut], errors="coerce").values[0] *
                            df_formula[df_formula["Ingrediente"] == ing]["% Inclusión"].values[0] / 100 * factor)
                            if pd.notnull(df_formula.loc[df_formula["Ingrediente"] == ing, nut].values[0]) else 0
                            for ing in ingredientes_seleccionados
                        ]))
                        for ing in ingredientes_seleccionados:
                            valor = pd.to_numeric(df_formula.loc[df_formula["Ingrediente"] == ing, nut], errors="coerce").values[0]
                            porc = df_formula[df_formula["Ingrediente"] == ing]["% Inclusión"].values[0]
                            aporte = fmt2((valor * porc) / 100 * factor) if pd.notnull(valor) else 0
                            valores.append(aporte)
                            porc_aporte.append(fmt2(100 * aporte / total_nut) if total_nut > 0 else 0)
                        df_aporte = pd.DataFrame({
                            "Ingrediente": ingredientes_seleccionados,
                            f"Aporte de {nut} ({label})": valores,
                            "% Inclusión": [fmt2(df_formula[df_formula["Ingrediente"] == ing]["% Inclusión"].values[0]) for ing in ingredientes_seleccionados],
                            "Contenido por kg": [fmt2(df_formula[df_formula["Ingrediente"] == ing][nut].values[0]) for ing in ingredientes_seleccionados],
                            f"Proporción aporte {nut} (%)": porc_aporte,
                        })
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=ingredientes_seleccionados,
                            y=valores,
                            marker_color=[color_map[ing] for ing in ingredientes_seleccionados],
                            text=[f"{fmt2(v)}" for v in valores],
                            textposition='auto',
                            customdata=porc_aporte,
                            hovertemplate=f'%{{x}}<br>Aporte: %{{y:.2f}} {label}<br>Proporción aporte: %{{customdata:.2f}}%<extra></extra>',
                        ))
                        fig.update_layout(
                            xaxis_title="Ingrediente",
                            yaxis_title=f"Aporte de {nut} ({label})",
                            title=f"Aporte de cada ingrediente a {nut} ({label})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(fmt2_df(df_aporte), use_container_width=True)
                        st.markdown(
                            f"Puedes ajustar la unidad para visualizar el aporte en la escala más útil para tu análisis."
                        )
                        resumen_escenario += f"\n\nAporte de {nut}:\n{df_aporte.to_csv(index=False)}"

                st.markdown("---")
                st.markdown("### Análisis automático con IA")
                prompt_esc2 = st.text_area("Consulta para IA sobre el aporte por ingrediente a nutrientes", value="Describe el aporte de nutrientes por ingrediente y posibles ajustes a la dieta.")
                if st.button("Analizar con IA - Aporte Nutrientes"):
                    respuesta_ia = analizar_escenario_con_ia(prompt_esc2, resumen_escenario)
                    st.markdown(f"**IA:** {respuesta_ia}")

            with subtab3:
                unit_options = {
                    'kg': ['kg', 'ton'],
                    'g': ['g', '100g', 'kg', 'ton'],
                    'kcal': ['kcal', '1000kcal'],
                    '%': ['%', '100 unidades'],
                    'unidad': ['unidad', '100 unidades', '1000 unidades', 'kg', 'ton'],
                }
                shadow_tab = st.tabs([nut for nut in nutrientes_seleccionados])
                for idx, nut in enumerate(nutrientes_seleccionados):
                    with shadow_tab[idx]:
                        unit = unidades_dict.get(nut, "unidad")
                        manual_unit = unit_selector(
                            f"Unidad para {nut}",
                            unit_options.get(unit, ["unidad", "100 unidades", "1000 unidades", "kg", "ton"]),
                            unit_options.get(unit, ["unidad"])[0],
                            key=f"unit_selector_{nut}_shadow_tab1"
                        )
                        factor, label = get_unit_factor(unit, manual_unit)
                        precios_unit = []
                        contenidos = []
                        precios_ing = []
                        for i, ing in enumerate(ingredientes_seleccionados):
                            row = df_formula[df_formula["Ingrediente"] == ing].iloc[0]
                            contenido = fmt2(pd.to_numeric(row.get(nut, 0), errors="coerce"))
                            precio = fmt2(row.get("precio", np.nan))
                            if pd.notnull(contenido) and contenido > 0 and pd.notnull(precio):
                                precios_unit.append(fmt2(precio / contenido * factor))
                            else:
                                precios_unit.append(np.nan)
                            contenidos.append(contenido)
                            precios_ing.append(precio)
                        df_shadow = pd.DataFrame({
                            "Ingrediente": ingredientes_seleccionados,
                            f"Precio por {manual_unit}": precios_unit,
                            f"Contenido de {nut} por kg": contenidos,
                            "Precio ingrediente (USD/kg)": precios_ing,
                        })
                        min_idx = df_shadow[f"Precio por {manual_unit}"].idxmin()
                        df_shadow["Es el más barato"] = ["✅" if i == min_idx else "" for i in df_shadow.index]
                        bar_colors = ['green' if i == min_idx else 'royalblue' for i in range(len(df_shadow))]
                        fig_shadow = go.Figure()
                        fig_shadow.add_trace(go.Bar(
                            x=df_shadow["Ingrediente"],
                            y=df_shadow[f"Precio por {manual_unit}"],
                            marker_color=bar_colors,
                            text=[f"{fmt2(v)}" for v in df_shadow[f"Precio por {manual_unit}"]],
                            textposition='auto',
                            customdata=np.stack([df_shadow["Es el más barato"]], axis=-1),
                            hovertemplate=f'%{{x}}<br>Precio sombra: %{{y:.2f}} {label}<br>%{{customdata[0]}}<extra></extra>',
                        ))
                        fig_shadow.update_layout(
                            xaxis_title="Ingrediente",
                            yaxis_title=label,
                            title=f"Precio sombra y costo por ingrediente para {nut}",
                        )
                        st.plotly_chart(fig_shadow, use_container_width=True)
                        st.dataframe(fmt2_df(df_shadow), use_container_width=True)
                        st.markdown(
                            f"**El precio sombra de {nut} es el menor costo posible para obtener una unidad de este nutriente usando el ingrediente más barato en la fórmula.**\n\n"
                            f"- Puedes ajustar la unidad para mejorar la visualización.\n"
                            f"- El ingrediente marcado con ✅ aporta el precio sombra."
                        )
                        resumen_escenario += f"\n\nShadow price {nut}:\n{df_shadow.to_csv(index=False)}"

                st.markdown("---")
                st.markdown("### Análisis automático con IA")
                prompt_esc3 = st.text_area("Consulta para IA sobre el precio sombra", value="Interpreta el precio sombra de los nutrientes y su impacto en el costo de la dieta.")
                if st.button("Analizar con IA - Shadow Price"):
                    respuesta_ia = analizar_escenario_con_ia(prompt_esc3, resumen_escenario)
                    st.markdown(f"**IA:** {respuesta_ia}")

            st.markdown("---")
            escenarios = cargar_escenarios()
            nombre_escenario = st.text_input("Nombre para guardar este escenario", value="Escenario " + str(len(escenarios)+1), key="nombre_escenario")
            if st.button("Guardar escenario"):
                escenario = {
                    "nombre": nombre_escenario,
                    "ingredientes": ingredientes_seleccionados,
                    "nutrientes": nutrientes_seleccionados,
                    "data_formula": data_formula,
                    "tabla": fmt2_df(tabla).to_dict(),
                    "unidades_dict": unidades_dict,
                    "costo_total": fmt2(float(tabla["Costo proporcional (USD/kg)"].iloc[-1]) * 1000),
                }
                escenarios.append(escenario)
                guardar_escenarios(escenarios)
                st.success(f"Escenario '{nombre_escenario}' guardado exitosamente.")
        else:
            st.info("Selecciona ingredientes y nutrientes para comenzar el análisis y visualización.")

with tab2:
    escenarios = cargar_escenarios()
    comparador_escenarios(escenarios)
