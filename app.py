import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(page_title="FitDistPro - Equipo [TU_NOMBRE]", layout="wide")

# ============================================================================
# 📚 BASE DE DATOS EDUCATIVA
# ============================================================================
DISTRIBUCIONES_INFO = {
    "Bernoulli": {"nombre": "Distribución de Bernoulli", "descripcion": "Modela experimentos con solo DOS resultados: éxito (1) o fracaso (0).", "uso": "Lanzar moneda, aprobar/reprobar, comprar/no comprar.", "formula": r"P(X=k) = p^k (1-p)^{1-k}", "parametros": {"p": "Probabilidad de éxito"}},
    "Binomial": {"nombre": "Distribución Binomial", "descripcion": "Cuenta éxitos en 'n' ensayos independientes con misma probabilidad 'p'.", "uso": "Defectuosos en un lote, caras en lanzamientos.", "formula": r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}", "parametros": {"n": "Ensayos", "p": "Prob. éxito"}},
    "Poisson": {"nombre": "Distribución de Poisson", "descripcion": "Eventos en intervalo fijo con tasa constante y independiente.", "uso": "Llamadas/hora, accidentes/día, clientes/minuto.", "formula": r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}", "parametros": {"λ": "Tasa promedio"}},
    "Geométrica": {"nombre": "Distribución Geométrica", "descripcion": "Ensayos necesarios para obtener el PRIMER éxito.", "uso": "Lanzamientos hasta primer 6, intentos hasta primera venta.", "formula": r"P(X=k) = (1-p)^{k-1} p", "parametros": {"p": "Prob. éxito"}},
    "Binomial Negativa": {"nombre": "Distribución Binomial Negativa", "descripcion": "Fracasos antes de obtener 'r' éxitos en ensayos independientes.", "uso": "Fallas antes de 3 éxitos, pacientes antes de 5 curaciones.", "formula": r"P(X=k) = \binom{k+r-1}{k} p^r (1-p)^k", "parametros": {"r": "Éxitos deseados", "p": "Prob. éxito"}},
    "Normal": {"nombre": "Distribución Normal (Gaussiana)", "descripcion": "Datos simétricos alrededor de una media. La más importante en estadística.", "uso": "Alturas, pesos, calificaciones, errores.", "formula": r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}", "parametros": {"μ": "Media", "σ": "Desv. Estándar"}},
    "Weibull": {"nombre": "Distribución de Weibull", "descripcion": "Tiempos de falla y vida útil. Flexible para tasas crecientes/decrecientes.", "uso": "Duración baterías, falla de máquinas, supervivencia.", "formula": r"f(x) = \frac{k}{\lambda} (\frac{x}{\lambda})^{k-1} e^{-(x/\lambda)^k}", "parametros": {"k": "Forma", "λ": "Escala"}},
    "Exponencial": {"nombre": "Distribución Exponencial", "descripcion": "Tiempo entre eventos en proceso de Poisson. Propiedad 'sin memoria'.", "uso": "Tiempo entre llamadas, duración electrónicos.", "formula": r"f(x) = \lambda e^{-\lambda x}", "parametros": {"λ": "Tasa"}},
    "Gamma": {"nombre": "Distribución Gamma", "descripcion": "Tiempo hasta que ocurren 'α' eventos en proceso de Poisson.", "uso": "Tiempos de espera, seguros, precipitación.", "formula": r"f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}", "parametros": {"α": "Forma", "β": "Tasa"}},
    "Lognormal": {"nombre": "Distribución Log-Normal", "descripcion": "Logaritmo de la variable es normal. Siempre positiva, cola derecha larga.", "uso": "Ingresos, precios acciones, tamaños partículas.", "formula": r"f(x) = \frac{1}{x\sigma\sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}", "parametros": {"μ": "Media log", "σ": "Desv log"}}
}

st.title(" FitDistPro - Ajuste de Distribuciones de Probabilidad")

# ============================================================================
# 1️⃣ GENERADOR DE DATOS ALEATORIOS (OPCIONAL)
# ============================================================================
with st.expander("🎲 Generar Datos Aleatorios de Prueba (10-100)", expanded=False):
    col_g1, col_g2, col_g3 = st.columns([1, 1, 2])
    with col_g1:
        n_gen = st.number_input("Cantidad", min_value=10, max_value=100, value=50)
    with col_g2:
        tipo_gen = st.selectbox("Tipo", ["Discreto (Poisson)", "Continuo (Normal)"])
    with col_g3:
        if st.button("🎲 Generar y Analizar", type="primary"):
            if tipo_gen.startswith("Discreto"):
                st.session_state['data_source'] = np.random.poisson(lam=5, size=n_gen)
            else:
                st.session_state['data_source'] = np.random.normal(loc=50, scale=10, size=n_gen)
            st.session_state['source_type'] = "generados"
            st.success(f"✅ {n_gen} datos generados. El análisis se ejecutará abajo.")

# ============================================================================
# 2️⃣ CARGA MANUAL DE DATOS (COMO LO TENÍAS)
# ============================================================================
st.markdown("---")
st.subheader("📥 Cargar Mis Propios Datos")
col_up1, col_up2 = st.columns(2)
with col_up1:
    upload = st.file_uploader("CSV o Excel", type=["csv", "xlsx", "xls"])
with col_up2:
    paste_data = st.text_area("O pegar datos (separados por comas o saltos)", height=80)

# ============================================================================
# 3️⃣ PROCESAMIENTO UNIFICADO
# ============================================================================
data = None
if 'source_type' in st.session_state and st.session_state['source_type'] == "generados":
    data = st.session_state['data_source']
    st.info("🔄 Analizando datos generados automáticamente...")
elif upload is not None:
    try:
        df = pd.read_csv(upload) if upload.name.endswith(".csv") else pd.read_excel(upload)
        data = df.iloc[:, 0].dropna().values
        st.session_state['source_type'] = "manual"
    except Exception as e:
        st.error(f"Error al leer archivo: {e}")
        st.stop()
elif paste_data:
    try:
        raw = [x.strip() for x in paste_data.replace("\n", ",").split(",") if x.strip()]
        data = np.array([float(x.replace(",", ".")) for x in raw])
        st.session_state['source_type'] = "manual"
    except:
        st.error("Formato inválido. Usa números separados por comas.")
        st.stop()
else:
    st.write("⬆️ Usa el generador de arriba o carga/pega tus datos para comenzar.")
    st.stop()

# Limpieza y detección
data = data[~np.isnan(data)]
if len(data) < 5: st.error("Muy pocos datos válidos."); st.stop()

is_discrete = bool(np.all(np.mod(data, 1) == 0) and len(data) > 10)
tipo = "Discreto (conteos)" if is_discrete else "Continuo (mediciones)"
st.info(f" Tipo detectado: {tipo} | Registros: {len(data)}")

# Mostrar datos crudos si fueron generados
if st.session_state['source_type'] == "generados":
    with st.expander("👁️ Ver datos generados"):
        st.write(data)

# ============================================================================
# 4️⃣ FUNCIÓN DE AJUSTE (ROBUSTA)
# ============================================================================
def fit_distribution(dist_name, data):
    try:
        dist_map = {"Normal": stats.norm, "Lognormal": stats.lognorm, "Weibull": stats.weibull_min,
                    "Gamma": stats.gamma, "Exponencial": stats.expon, "Beta": stats.beta,
                    "Logística": stats.logistic, "Gumbel": stats.gumbel_r, "Pareto": stats.pareto,
                    "Poisson": stats.poisson, "Binomial": stats.binom, "Binomial Negativa": stats.nbinom,
                    "Geométrica": stats.geom, "Bernoulli": stats.bernoulli}
        if dist_name not in dist_map: return None
        dist_obj = dist_map[dist_name]
        n = len(data)
        
        if dist_name == "Bernoulli":
            if not np.all(np.isin(data, [0, 1])): return None
            params = (data.mean(), 0)
        elif dist_name == "Binomial":
            n_trials = max(int(data.max()), 10)
            p = min(data.mean() / n_trials, 0.99)
            params = (n_trials, p, 0)
        elif dist_name == "Binomial Negativa":
            mu, var = data.mean(), data.var()
            if var <= mu: return None
            p = mu / var
            r = max((mu**2)/(var-mu), 0.5)
            params = (r, min(max(p, 0.01), 0.99), 0)
        elif dist_name == "Geométrica":
            p = 1.0/data.mean() if data.mean()>0 else 0.5
            params = (min(max(p, 0.01), 0.99), 0)
        elif dist_name == "Poisson":
            params = (data.mean(), 0)
        else:
            if dist_name in ["Beta", "Pareto"]: params = dist_obj.fit(data, floc=data.min(), scale=max(data.max()-data.min(), 1))
            elif dist_name in ["Lognormal", "Weibull", "Gamma", "Exponencial"]: params = dist_obj.fit(data, floc=0)
            else: params = dist_obj.fit(data)
            
        dist = dist_obj(*params)
        k = len(params) if hasattr(params, '__len__') else 1
        
        if dist_name in ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"]:
            loglik = float(np.sum(dist.logpmf(data)))
            uvals = np.unique(data)
            obs = np.array([np.sum(data==x) for x in uvals])
            exp = np.maximum(np.array([dist.pmf(x)*n for x in uvals]), 1e-8)
            chi2, p_chi2 = stats.chisquare(obs, exp)
            ks_p = 0
        else:
            loglik = float(np.sum(dist.logpdf(data)))
            ks_stat, ks_p = stats.kstest(data, dist.cdf)
            counts, edges = np.histogram(data, bins='auto')
            exp = np.maximum(n*np.diff(dist.cdf(edges)), 1e-8)
            chi2, p_chi2 = stats.chisquare(counts, exp)
            
        aic, bic = 2*k-2*loglik, k*np.log(n)-2*loglik
        return {"Distribución": dist_name, "Parámetros": tuple(round(float(p),4) for p in params),
                "Log-Likelihood": round(loglik,2), "AIC": round(aic,2), "BIC": round(bic,2),
                "K-S p-valor": round(float(ks_p),4), "Chi² p-valor": round(float(p_chi2),4), "dist": dist}
    except: return None

candidates = ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"] if is_discrete else \
             ["Normal", "Lognormal", "Weibull", "Gamma", "Exponencial", "Beta", "Logística", "Gumbel", "Pareto"]
results = [r for r in [fit_distribution(d, data) for d in candidates] if r]
results.sort(key=lambda x: x['AIC'])
if not results: st.error("No se ajustó ninguna distribución."); st.stop()

df_res = pd.DataFrame(results).drop(columns=["dist"])
df_res.index += 1
if is_discrete:
    df_res["Prueba"] = "Chi-Cuadrado"; df_res["p-valor"] = df_res["Chi² p-valor"]; df_res.drop(columns=["K-S p-valor","Chi² p-valor"], inplace=True)
else:
    df_res["Prueba"] = "Kolmogorov-Smirnov"; df_res["p-valor"] = df_res["K-S p-valor"]; df_res.drop(columns=["K-S p-valor","Chi² p-valor"], inplace=True)

# ============================================================================
# 5️⃣ VISUALIZACIÓN
# ============================================================================
col1, col2 = st.columns(2)
with col1:
    st.subheader(" Ranking (menor AIC = mejor)")
    st.dataframe(df_res.round(4), use_container_width=True)

with col2:
    st.subheader("📊 Histograma + Ajuste (Top 1)")
    best = results[0]; dist = best["dist"]; dname = best["Distribución"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Datos", opacity=0.6, histnorm='probability density'))
    if is_discrete:
        xv = np.arange(int(data.min()), int(data.max())+1)
        fig.add_trace(go.Bar(x=xv, y=dist.pmf(xv), name=f"Top 1: {dname}", opacity=0.8))
    else:
        xv = np.linspace(data.min(), data.max(), 200)
        fig.add_trace(go.Scatter(x=xv, y=dist.pdf(xv), name=f"Top 1: {dname}", line=dict(color='red', width=3)))
    fig.update_layout(barmode='overlay', template="plotly_white", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 6️⃣ FICHA EDUCATIVA (FÓRMULA, DESCRIPCIÓN, USO)
# ============================================================================
st.markdown("---")
st.subheader(f"📚 Distribución Ganadora: {dname}")
if dname in DISTRIBUCIONES_INFO:
    info = DISTRIBUCIONES_INFO[dname]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"### {info['nombre']}")
        st.info(f"📖 **Descripción**: {info['descripcion']}")
        st.success(f"💡 **Casos de uso**: {info['uso']}")
        st.markdown("### 📐 Fórmula Matemática")
        st.latex(info['formula'])
    with col_b:
        st.markdown("### 🔧 Parámetros Clave")
        for p, d in info['parametros'].items(): st.write(f"- **{p}**: {d}")
        st.markdown("### 🔢 Estimados en tus datos")
        st.json(dict(zip(["P"+str(i) for i in range(len(best['Parámetros']))], best['Parámetros'])))

# ============================================================================
# 7️ REPORTE Y EXPORTACIÓN
# ============================================================================
st.markdown("---")
col_r1, col_r2 = st.columns(2)
with col_r1:
    top = df_res.iloc[0]; test = top["Prueba"]; p = top["p-valor"]
    st.success(f"**Conclusión**: `{top['Distribución']}` es la más adecuada.\n**{test}**: p-valor = {p:.4f} {'✅ ACEPTADO' if p>0.05 else '❌ RECHAZADO'}")
with col_r2:
    if st.button("📥 Descargar CSV"):
        st.download_button("Descargar", data=df_res.to_csv(index=False), file_name="ajuste.csv", mime="text/csv")
