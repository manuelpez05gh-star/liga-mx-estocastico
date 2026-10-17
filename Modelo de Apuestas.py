import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Liga MX - Análisis Estocástico", layout="wide")

# ==========================================
# 1. CARGA DE DATOS LOCALES (CSV)
# ==========================================
equipos_liga_mx = [
    "Guadalajara", "Toluca", "América", "Cruz Azul", 
    "Querétaro", "León", "Tijuana", "Puebla", 
    "Atlas", "Pachuca", "UNAM", "Monterrey", 
    "Necaxa", "UANL", "Atlante", "Atlético San Luis", 
    "Santos Laguna", "FC Juárez"
]

@st.cache_data
def obtener_datos_historicos_locales(equipo_a, equipo_b):
    """
    Carga los datos reales directamente desde el archivo local 'liga_mx.csv'.
    """
    try:
        df_liga = pd.read_csv("liga_mx.csv")
        
        # Filtrar datos de ambos clubes
        datos_a = df_liga[df_liga['Squad'].str.contains(equipo_a, case=False, na=False)]
        datos_b = df_liga[df_liga['Squad'].str.contains(equipo_b, case=False, na=False)]
        
        if datos_a.empty or datos_b.empty:
            st.error("Error: No se encontró uno de los equipos en el archivo CSV.")
            return 1.2, 1.2, 0, 0
            
        partidos_a = float(datos_a['MP'].values[0])
        partidos_b = float(datos_b['MP'].values[0])
        
        # Intensidad de goles por partido (Lambda Poisson)
        lambda_a = float(datos_a['GF'].values[0]) / partidos_a if partidos_a > 0 else 1.0
        lambda_b = float(datos_b['GF'].values[0]) / partidos_b if partidos_b > 0 else 1.0
        
        victorias_a = int(datos_a['W'].values[0])
        victorias_b = int(datos_b['W'].values[0])
        
        return lambda_a, lambda_b, victorias_a, victorias_b

    except FileNotFoundError:
        st.error("No se encontró el archivo 'liga_mx.csv'. Asegúrate de guardarlo en la misma carpeta que este script.")
        return 1.2, 1.2, 0, 0
    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo: {e}")
        return 1.2, 1.2, 0, 0

# ==========================================
# 2. INTERFAZ: SECCIÓN DE ANÁLISIS HISTÓRICO Y ESPERADO
# ==========================================
st.title("⚽ Liga MX: Modelo Estocástico de Predicción")
st.markdown("Basado en Procesos de Poisson y Simulación de Montecarlo leyendo datos históricos reales.")

st.header("1. Selección de Equipos y Valor Esperado")
col1, col2 = st.columns(2)

with col1:
    equipo_local = st.selectbox("Equipo Local", equipos_liga_mx, index=0)
with col2:
    equipo_visita = st.selectbox("Equipo Visitante", equipos_liga_mx, index=1)

if equipo_local == equipo_visita:
    st.error("Por favor, selecciona dos equipos distintos.")
    st.stop()

# Cargar datos (sin tiros de esquina)
l_local, l_visita, vic_loc, vic_vis = obtener_datos_historicos_locales(equipo_local, equipo_visita)
lambda_goles_totales = l_local + l_visita

# Simulación de Montecarlo para el resultado (10,000 iteraciones)
n_simulaciones = 10000
sim_goles_local = np.random.poisson(l_local, n_simulaciones)
sim_goles_visita = np.random.poisson(l_visita, n_simulaciones)

victorias_local = np.sum(sim_goles_local > sim_goles_visita)
empates = np.sum(sim_goles_local == sim_goles_visita)
victorias_visita = np.sum(sim_goles_local < sim_goles_visita)

prob_loc = victorias_local / n_simulaciones
prob_emp = empates / n_simulaciones
prob_vis = victorias_visita / n_simulaciones

# Métricas Históricas
st.subheader("Historial Reciente y Estadísticas Base (Torneo Actual)")
c1, c2 = st.columns(2)
c1.metric(f"Victorias Totales {equipo_local}", vic_loc)
c2.metric(f"Victorias Totales {equipo_visita}", vic_vis)

# Resultados Estocásticos
st.subheader("Resultados Más Probables (Simulación de Montecarlo)")
st.markdown(f"Valor Esperado de Goles ($\\lambda$): **{equipo_local} ({l_local:.2f}) vs {equipo_visita} ({l_visita:.2f})** | Total esperado: **{lambda_goles_totales:.2f} goles**")

col_res1, col_res2, col_res3 = st.columns(3)
col_res1.metric(f"Gana {equipo_local}", f"{prob_loc * 100:.1f}%")
col_res2.metric("Empate", f"{prob_emp * 100:.1f}%")
col_res3.metric(f"Gana {equipo_visita}", f"{prob_vis * 100:.1f}%")

st.divider()

# ==========================================
# 3. INTERFAZ: VISUALIZACIONES ESTADÍSTICAS
# ==========================================
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    # Gráfico de distribución de goles (Total)
    st.subheader("Distribución de Goles Totales")
    x_goles = np.arange(0, 8)
    y_prob_goles = poisson.pmf(x_goles, lambda_goles_totales)

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    ax_bar.bar(x_goles, y_prob_goles, color='#1f77b4', alpha=0.8)
    ax_bar.set_xlabel("Número de Goles Totales")
    ax_bar.set_ylabel("Probabilidad")
    ax_bar.set_title("Distribución de Poisson (Total del Partido)")
    st.pyplot(fig_bar)

with col_viz2:
    # Mapa de Calor de Resultados Exactos
    st.subheader("Probabilidad de Marcador Exacto")
    max_goles = 5
    matriz_prob = np.zeros((max_goles + 1, max_goles + 1))

    for x in range(max_goles + 1):
        for y in range(max_goles + 1):
            matriz_prob[x, y] = poisson.pmf(x, l_local) * poisson.pmf(y, l_visita)

    fig_heat, ax_heat = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        matriz_prob, 
        annot=True, 
        fmt=".1%", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Probabilidad'}, 
        ax=ax_heat
    )
    ax_heat.set_ylabel(f"Goles {equipo_local}", fontsize=10)
    ax_heat.set_xlabel(f"Goles {equipo_visita}", fontsize=10)
    ax_heat.set_title("Matriz Conjunta de Resultados", fontsize=12)
    st.pyplot(fig_heat)

# Marcador más probable (Matriz)
idx_max = np.unravel_index(np.argmax(matriz_prob, axis=None), matriz_prob.shape)
prob_max = np.max(matriz_prob)
st.success(f"📌 El marcador exacto con mayor probabilidad matemática es **{equipo_local} {idx_max[0]} - {idx_max[1]} {equipo_visita}** con un **{(prob_max * 100):.2f}%**.")

st.divider()

# ==========================================
# 4. INTERFAZ: CALCULADORA MANUAL DE PROBABILIDADES
# ==========================================
st.header("2. Calculadora de Escenarios Manual")
st.markdown("Ingresa un escenario específico para calcular su probabilidad teórica conjunta.")

col_man1, col_man2 = st.columns(2)

with col_man1:
    resultado_deseado = st.selectbox(
        "Resultado del Partido", 
        ["Gana Local", "Empate", "Gana Visita", "Local o Empate", "Visita o Empate"]
    )

with col_man2:
    goles_deseados = st.number_input("Goles Totales Exactos", min_value=0, max_value=15, value=2)

# 1. Probabilidad del resultado
if resultado_deseado == "Gana Local":
    p_resultado = prob_loc
elif resultado_deseado == "Empate":
    p_resultado = prob_emp
elif resultado_deseado == "Gana Visita":
    p_resultado = prob_vis
elif resultado_deseado == "Local o Empate":
    p_resultado = prob_loc + prob_emp
elif resultado_deseado == "Visita o Empate":
    p_resultado = prob_vis + prob_emp

# 2. Probabilidad de goles exactos
p_goles = poisson.pmf(goles_deseados, lambda_goles_totales)

# Probabilidad Conjunta (Resultado y Goles)
prob_conjunta = p_resultado * p_goles

st.info(f"**Probabilidad del escenario seleccionado:** {(prob_conjunta * 100):.2f}%")

with st.expander("Desglose Matemático del Cálculo"):
    st.markdown(f"""
    * **Probabilidad de Resultado ({resultado_deseado}):** {p_resultado*100:.2f}%
    * **Probabilidad de Goles Totales ({goles_deseados}):** {p_goles*100:.2f}%
    
    *Cálculo conjunto:* Multiplicación de las probabilidades de ambos eventos.
    """)
