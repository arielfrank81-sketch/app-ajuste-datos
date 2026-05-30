import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(page_title="FitDistPro - Equipo [TU_NOMBRE]", layout="wide")

# 📚 BASE DE DATOS DE DISTRIBUCIONES (Fórmulas y descripciones)
DISTRIBUCIONES_INFO = {
    "Bernoulli": {
        "nombre": "Distribución de Bernoulli",
        "descripcion": "Modela un experimento con solo DOS resultados posibles: éxito (1) o fracaso (0). Es la base de todas las distribuciones discretas.",
        "uso": "Lanzar una moneda, aprobar/reprobar, comprar/no comprar",
        "formula": r"""
        P(X = k) = \begin{cases}
        p & \text{si } k = 1 \text{ (éxito)} \\
        1-p & \text{si } k = 0 \text{ (fracaso)}
        \end{cases}
        """,
        "parametros": {"p": "Probabilidad de éxito (0 ≤ p ≤ 1)"},
        "media": "p",
        "varianza": "p(1-p)"
    },
    "Binomial": {
        "nombre": "Distribución Binomial",
        "descripcion": "Cuenta el número de éxitos en 'n' ensayos independientes, donde cada ensayo tiene la misma probabilidad 'p' de éxito.",
        "uso": "Número de caras en 10 lanzamientos, productos defectuosos en un lote",
        "formula": r"""
        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
        """,
        "parametros": {"n": "Número de ensayos", "p": "Probabilidad de éxito"},
        "media": "np",
        "varianza": "np(1-p)"
    },
    "Poisson": {
        "nombre": "Distribución de Poisson",
        "descripcion": "Modela el número de eventos que ocurren en un intervalo fijo de tiempo o espacio, cuando estos eventos suceden con una tasa constante y independientemente.",
        "uso": "Llamadas telefónicas por hora, accidentes por día, clientes por minuto",
        "formula": r"""
        P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
        """,
        "parametros": {"λ (lambda)": "Tasa promedio de eventos"},
        "media": "λ",
        "varianza": "λ"
    },
    "Geométrica": {
        "nombre": "Distribución Geométrica",
        "descripcion": "Representa el número de ensayos necesarios para obtener el PRIMER éxito en una secuencia de ensayos de Bernoulli independientes.",
        "uso": "Lanzamientos hasta sacar el primer 6, intentos hasta primera venta",
        "formula": r"""
        P(X = k) = (1-p)^{k-1} p
        """,
        "parametros": {"p": "Probabilidad de éxito en cada ensayo"},
        "media": "1/p",
        "varianza": "(1-p)/p²"
    },
    "Binomial Negativa": {
        "nombre": "Distribución Binomial Negativa",
        "descripcion": "Generaliza la geométrica: cuenta el número de fracasos antes de obtener 'r' éxitos en ensayos independientes.",
        "uso": "Fallas antes de 3 éxitos, pacientes antes de 5 curaciones",
        "formula": r"""
        P(X = k) = \binom{k+r-1}{k} p^r (1-p)^k
        """,
        "parametros": {"r": "Número de éxitos deseados", "p": "Probabilidad de éxito"},
        "media": "r(1-p)/p",
        "varianza": "r(1-p)/p²"
    },
    "Normal": {
        "nombre": "Distribución Normal (Gaussiana)",
        "descripcion": "La distribución más importante en estadística. Describe fenómenos naturales y sociales donde los datos se agrupan simétricamente alrededor de una media.",
        "uso": "Alturas, pesos, calificaciones, errores de medición",
        "formula": r"""
        f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
        """,
        "parametros": {"μ (mu)": "Media (centro)", "σ (sigma)": "Desviación estándar (dispersión)"},
        "media": "μ",
        "varianza": "σ²"
    },
    "Weibull": {
        "nombre": "Distribución de Weibull",
        "descripcion": "Modela tiempos de falla y vida útil de componentes. Muy flexible: puede representar tasas de falla crecientes, decrecientes o constantes.",
        "uso": "Duración de baterías, tiempo hasta falla de máquinas, supervivencia",
        "formula": r"""
        f(x) = \frac{k}{\lambda} \left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}
        """,
        "parametros": {"k": "Parámetro de forma", "λ (lambda)": "Parámetro de escala"},
        "media": "λ Γ(1 + 1/k)",
        "varianza": "λ² [Γ(1 + 2/k) - (Γ(1 + 1/k))²]"
    },
    "Exponencial": {
        "nombre": "Distribución Exponencial",
        "descripcion": "Modela el tiempo entre eventos en un proceso de Poisson. Tiene la propiedad de 'sin memoria': el pasado no afecta el futuro.",
        "uso": "Tiempo entre llamadas, duración de componentes electrónicos",
        "formula": r"""
        f(x) = \lambda e^{-\lambda x}
        """,
        "parametros": {"λ (lambda)": "Tasa de eventos"},
        "media": "1/λ",
        "varianza": "1/λ²"
    },
    "Gamma": {
        "nombre": "Distribución Gamma",
        "descripcion": "Generaliza la exponencial. Modela el tiempo hasta que ocurren 'α' eventos en un proceso de Poisson.",
        "uso": "Tiempos de espera, montos de seguros, precipitación acumulada",
        "formula": r"""
        f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}
        """,
        "parametros": {"α (alpha)": "Parámetro de forma", "β (beta)": "Parámetro de tasa"},
        "media": "α/β",
        "varianza": "α/β²"
    },
    "Lognormal": {
        "nombre": "Distribución Log-Normal",
        "descripcion": "Una variable es log-normal si el logaritmo de la variable sigue una distribución normal. Siempre positiva, con cola derecha larga.",
        "uso": "Ingresos, precios de acciones, tamaños de partículas",
        "formula": r"""
        f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}
        """,
        "parametros": {"μ": "Media del logaritmo", "σ": "Desviación del logaritmo"},
        "media": "e^(μ + σ²/2)",
        "varianza": "(e^(σ²) - 1) e^(2μ + σ²)"
    }
}

# 🎲 GENERADOR DE DATOS ALEATORIOS
def generar_datos_aleatorios(distribucion, n, **params):
    """Genera datos aleatorios según la distribución seleccionada"""
    try:
        if distribucion == "Bernoulli":
            return np.random.bernoulli(params['p'], n)
        elif distribucion == "Binomial":
            return np.random.binomial(params['n'], params['p'], n)
        elif distribucion == "Poisson":
            return np.random.poisson(params['lambda'], n)
        elif distribucion == "Geométrica":
            return np.random.geometric(params['p'], n)
        elif distribucion == "Binomial Negativa":
            return np.random.negative_binomial(params['r'], params['p'], n)
        elif distribucion == "Normal":
            return np.random.normal(params['mu'], params['sigma'], n)
        elif distribucion == "Weibull":
            return np.random.weibull(params['k'], n) * params['lambda']
        elif distribucion == "Exponencial":
            return np.random.exponential(1/params['lambda'], n)
        elif distribucion == "Gamma":
            return np.random.gamma(params['alpha'], 1/params['beta'], n)
        elif distribucion == "Lognormal":
            return np.random.lognormal(params['mu'], params['sigma'], n)
    except:
        return None
    return None

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("📊 FitDistPro - Ajuste de Distribuciones de Probabilidad")
st.markdown("---")

# 🔹 SECCIÓN 1: GENERADOR DE DATOS ALEATORIOS
with st.expander("🎲 Generador de Datos Aleatorios", expanded=False):
    col_gen1, col_gen2 = st.columns(2)
    
    with col_gen1:
        tipo_gen = st.selectbox("Tipo de dato", ["Discreto", "Continuo"])
        
        if tipo_gen == "Discreto":
            dist_gen = st.selectbox("Distribución", 
                ["Bernoulli", "Binomial", "Poisson", "Geométrica", "Binomial Negativa"])
        else:
            dist_gen = st.selectbox("Distribución",
                ["Normal", "Weibull", "Exponencial", "Gamma", "Lognormal"])
        
        n_datos = st.number_input("Cantidad de datos a generar", 
            min_value=10, max_value=10000, value=100, step=10)
    
    with col_gen2:
        # Parámetros dinámicos según distribución
        st.markdown("**Parámetros de la distribución:**")
        
        params_gen = {}
        if dist_gen == "Bernoulli":
            params_gen['p'] = st.slider("p (prob. éxito)", 0.1, 0.9, 0.5)
        elif dist_gen == "Binomial":
            params_gen['n'] = st.number_input("n (ensayos)", 1, 100, 10)
            params_gen['p'] = st.slider("p (prob. éxito)", 0.1, 0.9, 0.5)
        elif dist_gen == "Poisson":
            params_gen['lambda'] = st.number_input("λ (tasa)", 0.5, 20.0, 3.0)
        elif dist_gen == "Geométrica":
            params_gen['p'] = st.slider("p (prob. éxito)", 0.1, 0.9, 0.5)
        elif dist_gen == "Binomial Negativa":
            params_gen['r'] = st.number_input("r (éxitos)", 1, 20, 3)
            params_gen['p'] = st.slider("p (prob. éxito)", 0.1, 0.9, 0.3)
        elif dist_gen == "Normal":
            params_gen['mu'] = st.number_input("μ (media)", 0.0, 100.0, 50.0)
            params_gen['sigma'] = st.number_input("σ (desv. estándar)", 1.0, 20.0, 10.0)
        elif dist_gen == "Weibull":
            params_gen['k'] = st.number_input("k (forma)", 0.5, 5.0, 2.0)
            params_gen['lambda'] = st.number_input("λ (escala)", 1.0, 50.0, 10.0)
        elif dist_gen == "Exponencial":
            params_gen['lambda'] = st.number_input("λ (tasa)", 0.1, 10.0, 1.0)
        elif dist_gen == "Gamma":
            params_gen['alpha'] = st.number_input("α (forma)", 0.5, 10.0, 2.0)
            params_gen['beta'] = st.number_input("β (tasa)", 0.1, 5.0, 1.0)
        elif dist_gen == "Lognormal":
            params_gen['mu'] = st.number_input("μ (media log)", 0.0, 5.0, 2.0)
            params_gen['sigma'] = st.number_input("σ (desv log)", 0.1, 2.0, 0.5)
    
    if st.button("🎲 Generar Datos", type="primary"):
        datos_generados = generar_datos_aleatorios(dist_gen, n_datos, **params_gen)
        if datos_generados is not None:
            st.session_state['datos_generados'] = datos_generados
            st.success(f"✅ Se generaron {n_datos} datos de la distribución {dist_gen}")
            st.write("Primeros 10 datos:", datos_generados[:10])
        else:
            st.error("❌ Error al generar datos")

# ============================================================================
# 🔹 SECCIÓN 2: CARGA DE DATOS
# ============================================================================

st.markdown("---")
st.subheader("📥 Carga de Datos")

col1, col2 = st.columns(2)

with col1:
    upload = st.file_uploader("Subir archivo CSV o Excel", type=["csv", "xlsx", "xls"])

with col2:
    paste_data = st.text_area("O pegar datos (separados por comas o saltos de línea)", 
        height=100, placeholder="Ej: 3, 5, 2, 4, 3, 6...")

# Procesar datos
if 'datos_generados' in st.session_state:
    data = st.session_state['datos_generados']
    fuente_datos = "🎲 Datos generados aleatoriamente"
elif upload is not None:
    try:
        if upload.name.endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
        data = df.iloc[:, 0].dropna().values
        fuente_datos = "📁 Archivo subido"
    except Exception as e:
        st.error(f"Error al leer archivo: {e}")
        st.stop()
elif paste_data:
    try:
        raw_data = paste_data.replace("\n", ",").replace(";", ",").strip()
        raw_list = [x.strip() for x in raw_data.split(",") if x.strip()]
        clean_data = []
        for x in raw_list:
            try:
                clean_data.append(float(x.replace(",", ".")))
            except:
                pass
        data = np.array(clean_data)
        fuente_datos = "✍️ Datos pegados manualmente"
    except:
        st.error("Error al procesar datos. Verifica el formato.")
        st.stop()
else:
    # Datos de demostración
    data = np.random.normal(loc=50, scale=10, size=200)
    fuente_datos = "📊 Datos de demostración (Normal)"

if len(data) > 0:
    st.info(f"**Fuente**: {fuente_datos} | **Registros válidos**: {len(data)}")
    
    # Detección de tipo
    is_discrete = np.all(np.mod(data, 1) == 0) and len(data) > 10
    tipo = "Discreto (conteos)" if is_discrete else "Continuo (mediciones)"
    st.caption(f"🔍 Tipo detectado: {tipo}")
else:
    st.warning("No hay datos disponibles")
    st.stop()

# ============================================================================
# 🔹 SECCIÓN 3: AJUSTE DE DISTRIBUCIONES
# ============================================================================

st.markdown("---")
st.subheader("📈 Ajuste de Distribuciones")

# Función de ajuste (simplificada para brevedad - usa la que ya tenías)
def fit_distribution(dist_name, data):
    try:
        # Mapeo de distribuciones
        dist_map = {
            "Normal": stats.norm,
            "Lognormal": stats.lognorm,
            "Weibull": stats.weibull_min,
            "Gamma": stats.gamma,
            "Exponencial": stats.expon,
            "Beta": stats.beta,
            "Logística": stats.logistic,
            "Gumbel": stats.gumbel_r,
            "Pareto": stats.pareto,
            "Poisson": stats.poisson,
            "Binomial": stats.binom,
            "Binomial Negativa": stats.nbinom,
            "Geométrica": stats.geom,
            "Bernoulli": stats.bernoulli
        }
        
        if dist_name not in dist_map:
            return None
            
        dist_obj = dist_map[dist_name]
        n = len(data)
        
        # Ajuste de parámetros
        if dist_name == "Bernoulli":
            if not np.all(np.isin(data, [0, 1])):
                return None
            p = data.mean()
            params = (p, 0)
        elif dist_name == "Binomial":
            n_trials = max(int(data.max()), 10)
            p = min(data.mean() / n_trials, 0.99)
            params = (n_trials, p, 0)
        elif dist_name == "Binomial Negativa":
            mean_val = data.mean()
            var_val = data.var()
            if var_val <= mean_val:
                return None
            p = mean_val / var_val
            r = (mean_val ** 2) / (var_val - mean_val)
            r = max(r, 0.5)
            p = min(max(p, 0.01), 0.99)
            params = (r, p, 0)
        elif dist_name == "Geométrica":
            mean_val = data.mean()
            p = 1.0 / mean_val if mean_val > 0 else 0.5
            p = min(max(p, 0.01), 0.99)
            params = (p, 0)
        elif dist_name == "Poisson":
            mu = data.mean()
            params = (mu, 0)
        else:
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
            expected = np.array([dist.pmf(x) * n for x in unique_vals])
            expected = np.maximum(expected, 1e-8)
            chi2, p_chi2 = stats.chisquare(observed, expected)
            ks_stat, ks_p = 0, 0
        else:
            loglik = float(np.sum(dist.logpdf(data)))
            ks_stat, ks_p = stats.kstest(data, dist.cdf)
            counts, edges = np.histogram(data, bins='auto')
            expected = n * np.diff(dist.cdf(edges))
            expected = np.maximum(expected, 1e-8)
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

# Selección de distribuciones
if is_discrete:
    candidates = ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"]
else:
    candidates = ["Normal", "Lognormal", "Weibull", "Gamma", "Exponencial", "Beta", "Logística", "Gumbel", "Pareto"]

results = [fit_distribution(d, data) for d in candidates]
results = [r for r in results if r is not None]
results.sort(key=lambda x: x['AIC'])

if len(results) == 0:
    st.error("No se pudo ajustar ninguna distribución")
    st.stop()

# Crear DataFrame
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
# 🔹 SECCIÓN 4: VISUALIZACIÓN
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
# 🔹 SECCIÓN 5: INFORMACIÓN EDUCATIVA DE LA DISTRIBUCIÓN GANADORA
# ============================================================================

st.markdown("---")
st.subheader(f"📚 Información: {dist_name}")

if dist_name in DISTRIBUCIONES_INFO:
    info = DISTRIBUCIONES_INFO[dist_name]
    
    # Columnas para organizar la información
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown(f"### {info['nombre']}")
        st.info(f"📖 **Descripción**: {info['descripcion']}")
        st.success(f"💡 **Casos de uso**: {info['uso']}")
        
        st.markdown("### 📐 Fórmula Matemática")
        st.latex(info['formula'])
        
        # Parámetros
        st.markdown("### 🔧 Parámetros")
        for param, desc in info['parametros'].items():
            st.write(f"- **{param}**: {desc}")
    
    with col_info2:
        st.markdown("### 📊 Estadísticos Teóricos")
        st.write(f"**Media**: {info['media']}")
        st.write(f"**Varianza**: {info['varianza']}")
        
        # Parámetros estimados
        st.markdown("### 🔢 Parámetros Estimados")
        st.json(dict(zip(["p" + str(i) for i in range(len(best['Parámetros']))], best['Parámetros'])))

# ============================================================================
# 🔹 SECCIÓN 6: REPORTE Y EXPORTACIÓN
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
    if st.button("📥 Descargar Resultados (CSV)", type="secondary"):
        csv = df_res.to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name=f"ajuste_{dist_name.lower()}.csv",
            mime="text/csv"
        )
