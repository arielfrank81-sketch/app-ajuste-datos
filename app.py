import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import random

# Configuración de la página
st.set_page_config(page_title="FitDistPro - Equipo [TU_NOMBRE]", layout="wide")

# ============================================================================
# 📚 BASE DE DATOS EDUCATIVA (Fórmulas y Descripciones)
# ============================================================================
DISTRIBUCIONES_INFO = {
    "Bernoulli": {
        "nombre": "Distribución de Bernoulli",
        "descripcion": "Modela un experimento con solo DOS resultados posibles: éxito (1) o fracaso (0). Es la base de todas las distribuciones discretas.",
        "uso": "Lanzar una moneda, aprobar/reprobar, comprar/no comprar",
        "formula": r"P(X = k) = \begin{cases} p & \text{si } k = 1 \\ 1-p & \text{si } k = 0 \end{cases}",
        "parametros": {"p": "Probabilidad de éxito (0 ≤ p ≤ 1)"},
        "media": "p",
        "varianza": "p(1-p)"
    },
    "Binomial": {
        "nombre": "Distribución Binomial",
        "descripcion": "Cuenta el número de éxitos en 'n' ensayos independientes, donde cada ensayo tiene la misma probabilidad 'p' de éxito.",
        "uso": "Número de caras en 10 lanzamientos, productos defectuosos en un lote",
        "formula": r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}",
        "parametros": {"n": "Número de ensayos", "p": "Probabilidad de éxito"},
        "media": "np",
        "varianza": "np(1-p)"
    },
    "Poisson": {
        "nombre": "Distribución de Poisson",
        "descripcion": "Modela el número de eventos que ocurren en un intervalo fijo de tiempo o espacio, cuando estos eventos suceden con una tasa constante.",
        "uso": "Llamadas telefónicas por hora, accidentes por día, clientes por minuto",
        "formula": r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}",
        "parametros": {"λ (lambda)": "Tasa promedio de eventos"},
        "media": "λ",
        "varianza": "λ"
    },
    "Geométrica": {
        "nombre": "Distribución Geométrica",
        "descripcion": "Representa el número de ensayos necesarios para obtener el PRIMER éxito en una secuencia de ensayos de Bernoulli independientes.",
        "uso": "Lanzamientos hasta sacar el primer 6, intentos hasta primera venta",
        "formula": r"P(X = k) = (1-p)^{k-1} p",
        "parametros": {"p": "Probabilidad de éxito en cada ensayo"},
        "media": "1/p",
        "varianza": "(1-p)/p²"
    },
    "Binomial Negativa": {
        "nombre": "Distribución Binomial Negativa",
        "descripcion": "Generaliza la geométrica: cuenta el número de fracasos antes de obtener 'r' éxitos en ensayos independientes.",
        "uso": "Fallas antes de 3 éxitos, pacientes antes de 5 curaciones",
        "formula": r"P(X = k) = \binom{k+r-1}{k} p^r (1-p)^k",
        "parametros": {"r": "Número de éxitos deseados", "p": "Probabilidad de éxito"},
        "media": "r(1-p)/p",
        "varianza": "r(1-p)/p²"
    },
    "Normal": {
        "nombre": "Distribución Normal (Gaussiana)",
        "descripcion": "La distribución más importante en estadística. Describe fenómenos naturales y sociales donde los datos se agrupan simétricamente alrededor de una media.",
        "uso": "Alturas, pesos, calificaciones, errores de medición",
        "formula": r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        "parametros": {"μ (mu)": "Media (centro)", "σ (sigma)": "Desviación estándar (dispersión)"},
        "media": "μ",
        "varianza": "σ²"
    },
    "Weibull": {
        "nombre": "Distribución de Weibull",
        "descripcion": "Modela tiempos de falla y vida útil de componentes. Muy flexible: puede representar tasas de falla crecientes, decrecientes o constantes.",
        "uso": "Duración de baterías, tiempo hasta falla de máquinas, supervivencia",
        "formula": r"f(x) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}",
        "parametros": {"k": "Parámetro de forma", "λ (lambda)": "Parámetro de escala"},
        "media": "λ Γ(1 + 1/k)",
        "varianza": "λ² [Γ(1 + 2/k) - (Γ(1 + 1/k))²]"
    },
    "Exponencial": {
        "nombre": "Distribución Exponencial",
        "descripcion": "Modela el tiempo entre eventos en un proceso de Poisson. Tiene la propiedad de 'sin memoria': el pasado no afecta el futuro.",
        "uso": "Tiempo entre llamadas, duración de componentes electrónicos",
        "formula": r"f(x) = \lambda e^{-\lambda x}",
        "parametros": {"λ (lambda)": "Tasa de eventos"},
        "media": "1/λ",
        "varianza": "1/λ²"
    },
    "Gamma": {
        "nombre": "Distribución Gamma",
        "descripcion": "Generaliza la exponencial. Modela el tiempo hasta que ocurren 'α' eventos en un proceso de Poisson.",
        "uso": "Tiempos de espera, montos de seguros, precipitación acumulada",
        "formula": r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}",
        "parametros": {"α (alpha)": "Parámetro de forma", "β (beta)": "Parámetro de tasa"},
        "media": "α/β",
        "varianza": "α/β²"
    },
    "Lognormal": {
        "nombre": "Distribución Log-Normal",
        "descripcion": "Una variable es log-normal si el logaritmo de la variable sigue una distribución normal. Siempre positiva, con cola derecha larga.",
        "uso": "Ingresos, precios de acciones, tamaños de partículas",
        "formula": r"f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}",
        "parametros": {"μ": "Media del logaritmo", "σ": "Desviación del logaritmo"},
        "media": "e^(μ + σ²/2)",
        "varianza": "(e^(σ²) - 1) e^(2μ + σ²)"
    }
}

# ============================================================================
#  SECCIÓN 1: GENERADOR DE DATOS ALEATORIOS (SIMPLIFICADO)
# ============================================================================
st.title(" FitDistPro - Ajuste de Distribuciones de Probabilidad")
st.markdown("---")

with st.expander("🎲 Generador de Datos de Prueba", expanded=True):
    st.write("Genera datos aleatorios y deja que la app detecte automáticamente qué distribución son.")
    
    col_g1, col_g2 = st.columns([1, 2])
    with col_g1:
        n_generate = st.number_input("Cantidad de datos a generar", min_value=20, max_value=5000, value=100)
    
    with col_g2:
        if st.button("🎲 Generar y Analizar Automáticamente", type="primary"):
            # La app elige una distribución secreta para generar los datos
            opciones = [
                {"name": "Normal", "func": lambda n: stats.norm.rvs(loc=50, scale=10, size=n)},
                {"name": "Poisson", "func": lambda n: stats.poisson.rvs(mu=5, size=n)},
                {"name": "Exponencial", "func": lambda n: stats.expon.rvs(scale=10, size=n)},
                {"name": "Weibull", "func": lambda n: stats.weibull_min.rvs(2, scale=15, size=n)},
                {"name": "Binomial", "func": lambda n: stats.binom.rvs(n=10, p=0.5, size=n)}
            ]
            
            eleccion = random.choice(opciones)
            datos_generados = eleccion["func"](n_generate)
            
            st.session_state['datos_generados'] = datos_generados
            st.session_state['secreto_dist'] = eleccion["name"]
            st.success(f"✅ Se generaron {n_generate} datos. ¡Ahora analicemos!")
            st.info(f"🤫 Pista para ti: La app usó una distribución **{eleccion['name']}** para crear estos datos. ¿Podrá detectarla?")

# ============================================================================
# 🔹 SECCIÓN 2: CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

st.markdown("---")
st.subheader("📥 Fuente de Datos")

# Prioridad: 1. Generados, 2. Archivo, 3. Pegados, 4. Demo
if 'datos_generados' in st.session_state:
    data = st.session_state['datos_generados']
    fuente_datos = "🎲 Datos Generados Aleatoriamente"
else:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        upload = st.file_uploader("Subir CSV/Excel", type=["csv", "xlsx", "xls"])
    with col_up2:
        paste_data = st.text_area("O pegar datos (separados por comas)", height=80)

    if upload is not None:
        try:
            if upload.name.endswith(".csv"): df = pd.read_csv(upload)
            else: df = pd.read_excel(upload)
            data = df.iloc[:, 0].dropna().values
            fuente_datos = " Archivo Subido"
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")
            st.stop()
    elif paste_data:
        try:
            raw_list = [x.strip() for x in paste_data.replace("\n", ",").split(",") if x.strip()]
            data = np.array([float(x.replace(",", ".")) for x in raw_list])
            fuente_datos = "✍️ Datos Pegados"
        except:
            st.error("Formato inválido. Usa números separados por comas.")
            st.stop()
    else:
        # Datos de demostración por defecto si no hay nada
        data = np.random.normal(loc=50, scale=10, size=200)
        fuente_datos = "📊 Datos de Demostración (Normal)"

if len(data) > 0:
    st.caption(f"**Fuente**: {fuente_datos} | **Registros**: {len(data)}")
    
    # Detección automática
    is_discrete = np.all(np.mod(data, 1) == 0) and len(data) > 10
    tipo = "Discreto (conteos)" if is_discrete else "Continuo (mediciones)"
    st.info(f"🔍 Tipo detectado: {tipo}")
else:
    st.warning("No hay datos disponibles")
    st.stop()

# ============================================================================
# 🔹 SECCIÓN 3: FUNCIÓN DE AJUSTE (CORREGIDA Y ROBUSTA)
# ============================================================================

def fit_distribution(dist_name, data):
    try:
        dist_map = {
            "Normal": stats.norm, "Lognormal": stats.lognorm, "Weibull": stats.weibull_min,
            "Gamma": stats.gamma, "Exponencial": stats.expon, "Beta": stats.beta,
            "Logística": stats.logistic, "Gumbel": stats.gumbel_r, "Pareto": stats.pareto,
            "Poisson": stats.poisson, "Binomial": stats.binom, "Binomial Negativa": stats.nbinom,
            "Geométrica": stats.geom, "Bernoulli": stats.bernoulli
        }
        
        if dist_name not in dist_map: return None
        dist_obj = dist_map[dist_name]
        n = len(data)
        
        # Ajuste de parámetros específico para evitar errores
        if dist_name == "Bernoulli":
            if not np.all(np.isin(data, [0, 1])): return None
            params = (data.mean(), 0)
        elif dist_name == "Binomial":
            n_trials = max(int(data.max()), 10)
            p = min(data.mean() / n_trials, 0.99)
            params = (n_trials, p, 0)
        elif dist_name == "Binomial Negativa":
            mean_val, var_val = data.mean(), data.var()
            if var_val <= mean_val: return None # Requiere sobredispersión
            p = mean_val / var_val
            r = max((mean_val ** 2) / (var_val - mean_val), 0.5)
            params = (r, min(max(p, 0.01), 0.99), 0)
        elif dist_name == "Geométrica":
            p = 1.0 / data.mean() if data.mean() > 0 else 0.5
            params = (min(max(p, 0.01), 0.99), 0)
        elif dist_name == "Poisson":
            params = (data.mean(), 0)
        else:
            # Continuas
            if dist_name in ["Beta", "Pareto"]:
                params = dist_obj.fit(data, floc=data.min(), scale=max(data.max()-data.min(), 1))
            elif dist_name in ["Lognormal", "Weibull", "Gamma", "Exponencial"]:
                params = dist_obj.fit(data, floc=0)
            else:
                params = dist_obj.fit(data)
        
        dist = dist_obj(*params)
        k = len(params) if hasattr(params, '__len__') else 1
        
        # Métricas
        if dist_name in ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"]:
            loglik = float(np.sum(dist.logpmf(data)))
            unique_vals = np.unique(data)
            observed = np.array([np.sum(data == x) for x in unique_vals])
            expected = np.maximum(np.array([dist.pmf(x) * n for x in unique_vals]), 1e-8)
            chi2, p_chi2 = stats.chisquare(observed, expected)
            ks_stat, ks_p = 0, 0
        else:
            loglik = float(np.sum(dist.logpdf(data)))
            ks_stat, ks_p = stats.kstest(data, dist.cdf)
            counts, edges = np.histogram(data, bins='auto')
            expected = np.maximum(n * np.diff(dist.cdf(edges)), 1e-8)
            chi2, p_chi2 = stats.chisquare(counts, expected)
        
        aic = float(2*k - 2*loglik)
        bic = float(k*np.log(n) - 2*loglik)
        
        return {
            "Distribución": dist_name,
            "Parámetros": tuple(round(float(p), 4) for p in params),
            "Log-Likelihood": round(loglik, 2),
            "AIC": round(aic, 2),
            "BIC": round(bic, 2),
            "K-S p-valor": round(float(ks_p), 4),
            "Chi² p-valor": round(float(p_chi2), 4),
            "dist": dist
        }
    except:
        return None

# Selección de candidatos según tipo de dato
if is_discrete:
    candidates = ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"]
else:
    candidates = ["Normal", "Lognormal", "Weibull", "Gamma", "Exponencial", "Beta", "Logística", "Gumbel", "Pareto"]

results = [fit_distribution(d, data) for d in candidates]
results = [r for r in results if r is not None]
results.sort(key=lambda x: x['AIC'])

if len(results) == 0:
    st.error("No se pudo ajustar ninguna distribución a estos datos.")
    st.stop()

# Crear DataFrame de resultados
df_res = pd.DataFrame(results).drop(columns=["dist"])
df_res.index = df_res.index + 1

if is_discrete:
    df_res["Prueba"] = "Chi-Cuadrado"
    df_res["p-valor"] = df_res["Chi² p-valor"]
    df_res.drop(columns=["K-S p-valor", "Chi² p-valor"], inplace=True)
else:
    df_res["Prueba"] = "Kolmogorov-Smirnov"
    df_res["p-valor"] = df_res["K-S p-valor"]
    df_res.drop(columns=["K-S p-valor", "Chi² p-valor"], inplace=True)

cols_order = ["Distribución", "Parámetros", "Log-Likelihood", "AIC", "BIC", "Prueba", "p-valor"]
df_res = df_res[cols_order]

# ============================================================================
# 🔹 SECCIÓN 4: VISUALIZACIÓN (TABLA Y GRÁFICA)
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Ranking de Ajuste (menor AIC = mejor)")
    st.dataframe(df_res.round(4), use_container_width=True)

with col2:
    st.subheader("📊 Histograma + Ajuste (Top 1)")
    best = results[0]
    dist = best["dist"]
    dist_name = best["Distribución"]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Datos", opacity=0.6, histnorm='probability density'))
    
    if is_discrete:
        x_vals = np.arange(int(data.min()), int(data.max()) + 1)
        y_vals = dist.pmf(x_vals)
        fig.add_trace(go.Bar(x=x_vals, y=y_vals, name=f"Top 1: {dist_name}", opacity=0.8))
    else:
        x_vals = np.linspace(data.min(), data.max(), 200)
        y_vals = dist.pdf(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=f"Top 1: {dist_name}", line=dict(color='red', width=3)))
        
    fig.update_layout(barmode='overlay', template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 🔹 SECCIÓN 5: INFORMACIÓN EDUCATIVA (FÓRMULAS Y DESCRIPCIÓN)
# ============================================================================

st.markdown("---")
st.subheader(f"📚 Información Detallada: {dist_name}")

if dist_name in DISTRIBUCIONES_INFO:
    info = DISTRIBUCIONES_INFO[dist_name]
    
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown(f"### {info['nombre']}")
        st.info(f"📖 **Descripción**: {info['descripcion']}")
        st.success(f"💡 **Casos de uso comunes**: {info['uso']}")
        
        st.markdown("### 📐 Fórmula Matemática")
        st.latex(info['formula'])
        
        st.markdown("### 🔧 Parámetros Clave")
        for param, desc in info['parametros'].items():
            st.write(f"- **{param}**: {desc}")
    
    with col_info2:
        st.markdown("###  Estadísticos Teóricos")
        st.write(f"**Media Esperada**: {info['media']}")
        st.write(f"**Varianza Teórica**: {info['varianza']}")
        
        st.markdown("### 🔢 Parámetros Estimados")
        st.json(dict(zip(["Param "+str(i+1) for i in range(len(best['Parámetros']))], best['Parámetros'])))

# Mostrar la "Pista Secreta" si fue generado
if 'secreto_dist' in st.session_state:
    st.divider()
    secreto = st.session_state['secreto_dist']
    if secreto == dist_name:
        st.balloons()
        st.success(f"🎉 **¡EXCELENTE!** La app detectó correctamente que los datos fueron generados con una distribución **{secreto}**.")
    else:
        st.warning(f"🤔 Interesante... La app generó datos usando **{secreto}**, pero el ajuste estadístico sugirió que **{dist_name}** era una mejor explicación matemática para esta muestra específica.")

# ============================================================================
# 🔹 SECCIÓN 6: REPORTE FINAL
# ============================================================================

st.markdown("---")
col_rep1, col_rep2 = st.columns(2)

with col_rep1:
    st.subheader("📝 Conclusión Estadística")
    top = df_res.iloc[0]
    test = top["Prueba"]
    p = top["p-valor"]
    msg_ajuste = "✅ Ajuste ACEPTADO" if p > 0.05 else "❌ Ajuste RECHAZADO"
    
    st.success(f"**Distribución Ganadora**: `{top['Distribución']}`\n\n"
               f"**AIC**: {top['AIC']:.2f} | **BIC**: {top['BIC']:.2f}\n\n"
               f"**{test}**: p-valor = {p:.4f} → {msg_ajuste}")

with col_rep2:
    if st.button(" Descargar Resultados (CSV)", type="secondary"):
        csv = df_res.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"ajuste_{dist_name.lower()}.csv",
            mime="text/csv"
        )
