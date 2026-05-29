import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import io, base64

st.set_page_config(page_title="FitDistPro", layout="wide")
st.title("📊 FitDistPro - Ajuste de Distribuciones de Probabilidad")

# 1️⃣ CARGA DE DATOS
st.sidebar.header(" Carga de Datos")
upload = st.sidebar.file_uploader("CSV o Excel", type=["csv", "xlsx", "xls"])
paste_data = st.sidebar.text_area("O pega datos separados por comas o saltos de línea")

if upload is not None:
    if upload.name.endswith(".csv"):
        df = pd.read_csv(upload)
    else:
        df = pd.read_excel(upload)
    data = df.iloc[:, 0].dropna().values
elif paste_data:
    data = np.array([float(x.replace(",", ".")) for x in paste_data.split() if x])
else:
    data = np.random.normal(loc=50, scale=10, size=200)  # Demo
    st.warning("Usando datos de demostración. Sube tus propios datos.")

st.sidebar.metric("Registros válidos", len(data))

# Detección automático
is_discrete = np.all(np.mod(data, 1) == 0) and len(data) > 10
tipo = "Discreto (conteos)" if is_discrete else "Continuo (mediciones)"
st.sidebar.info(f"🔍 Tipo detectado: {tipo}")

# 2️⃣ DEFINICIÓN DE DISTRIBUCIONES
def fit_distribution(dist_name, data):
    try:
        # Mapeo de distribuciones
        dist_map = {
            # Continuas
            "Normal": stats.norm,
            "Lognormal": stats.lognorm,
            "Weibull": stats.weibull_min,
            "Gamma": stats.gamma,
            "Exponencial": stats.expon,
            "Beta": stats.beta,
            "Logística": stats.logistic,
            "Gumbel": stats.gumbel_r,
            "Pareto": stats.pareto,
            # Discretas
            "Poisson": stats.poisson,
            "Binomial": stats.binom,
            "Binomial Negativa": stats.nbinom,
            "Geométrica": stats.geom,
            "Bernoulli": stats.bernoulli,
            "Hipergeométrica": stats.hypergeom
        }
        
        if dist_name not in dist_map:
            return None
            
        dist_obj = dist_map[dist_name]
        n = len(data)
        
        # === AJUSTE DE PARÁMETROS ===
        if dist_name == "Bernoulli":
            # Bernoulli: solo éxitos/fracasos (0 o 1)
            if not np.all(np.isin(data, [0, 1])):
                return None  # Solo funciona con 0s y 1s
            p = data.mean()
            params = (p, 0)  # p, loc
            
        elif dist_name == "Binomial":
            # Binomial: n ensayos, p probabilidad
            n_trials = max(int(data.max()), 10)  # Estimamos n
            p = min(data.mean() / n_trials, 0.99)
            params = (n_trials, p, 0)  # n, p, loc
            
        elif dist_name == "Binomial Negativa":
            # Binomial negativa: r éxitos, p probabilidad
            params = stats.nbinom.fit(data, floc=0)
            
        elif dist_name == "Geométrica":
            # Geométrica: intentos hasta primer éxito
            params = stats.geom.fit(data, floc=0)
            
        elif dist_name == "Hipergeométrica":
            # Hipergeométrica: M población total, n éxitos en población, N muestras
            # Estimación simplificada
            M = max(int(data.max() * 2), int(data.mean() * 10), 100)  # Población total
            n = max(int(data.mean() * 2), 10)  # Éxitos en población
            N = max(int(n * 1.5), 5)  # Tamaño de muestra
            params = (M, n, N, 0)  # M, n, N, loc
            
        elif dist_name == "Poisson":
            # Poisson: lambda = media
            mu = data.mean()
            params = (mu, 0)  # mu, loc
            
        else:
            # Distribuciones continuas
            if dist_name in ["Beta", "Pareto"]:
                params = dist_obj.fit(data, floc=data.min(), scale=max(data.max()-data.min(), 1))
            elif dist_name in ["Lognormal", "Weibull", "Gamma", "Exponencial"]:
                params = dist_obj.fit(data, floc=0)
            else:
                params = dist_obj.fit(data)
        
        # Crear objeto de distribución
        dist = dist_obj(*params)
        k = len(params) if hasattr(params, '__len__') else 1
        
        # === CÁLCULO DE MÉTRICAS ===
        if dist_name in ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli", "Hipergeométrica"]:
            # Distribuciones discretas
            try:
                loglik = float(np.sum(dist.logpmf(data)))
            except:
                return None
                
            # Chi² para discretas
            unique_vals = np.unique(data)
            observed = np.array([np.sum(data == x) for x in unique_vals])
            try:
                expected = np.array([dist.pmf(x) * n for x in unique_vals])
                expected = np.maximum(expected, 1e-8)
                chi2, p_chi2 = stats.chisquare(observed, expected)
            except:
                chi2, p_chi2 = 0, 1.0
                
            ks_stat, ks_p = 0, 0  # KS no aplica bien a discretas
            
        else:
            # Distribuciones continuas
            try:
                loglik = float(np.sum(dist.logpdf(data)))
            except:
                return None
                
            try:
                ks_stat, ks_p = stats.kstest(data, dist.cdf)
            except:
                ks_stat, ks_p = 0, 0
                
            # Chi² aproximado
            try:
                counts, edges = np.histogram(data, bins='auto')
                expected = n * np.diff(dist.cdf(edges))
                expected = np.maximum(expected, 1e-8)
                chi2, p_chi2 = stats.chisquare(counts, expected)
            except:
                chi2, p_chi2 = 0, 1.0
        
        # Calcular AIC y BIC
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
    except Exception as e:
        print(f"Error en {dist_name}: {str(e)}")
        return None

# Selección automática
# Selección automática
if is_discrete:
    # Solo distribuciones discretas
    candidates = ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli", "Hipergeométrica"]
else:
    # Solo distribuciones continuas
    candidates = ["Normal", "Lognormal", "Weibull", "Gamma", "Exponencial", "Beta", "Logística", "Gumbel", "Pareto"]

results = [fit_distribution(d, data) for d in candidates]
results = [r for r in results if r is not None]

# Ranking por AIC
df_res = pd.DataFrame(results).drop(columns=["dist"]).sort_values("AIC")
df_res.index = range(1, len(df_res)+1)
df_res.rename(columns={"Distribución": "Rank"}, inplace=False)

# 3️⃣ VISUALIZACIÓN Y RESULTADOS
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Ranking de Ajuste (menor AIC = mejor)")
    st.dataframe(df_res.round(4), use_container_width=True)

with col2:
    st.subheader(" Histograma + Densidad (Top 1)")
    best = results[0]["dist"]
    x = np.linspace(data.min(), data.max(), 200)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Datos", opacity=0.6, histnorm='probability density'))
    fig.add_trace(go.Scatter(x=x, y=best.pdf(x) if not is_discrete else best.pmf(x), name=f"Mejor ajuste: {results[0]['Distribución']}", line=dict(color='red', width=3)))
    fig.update_layout(barmode='overlay', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# 4️⃣ REPORTE EN LENGUAJE NATURAL
st.markdown("---")
st.subheader(" Reporte Automático")
top = df_res.iloc[0]
st.success(f" **Conclusión**: La distribución `{top['Distribución']}` es la más adecuada para tus datos. "
           f"AIC={top['AIC']:.2f}, BIC={top['BIC']:.2f}. "
           f"{'El p-valor de Chi² ('+str(round(top['Chi² p-valor'],3))+') > 0.05 indica un buen ajuste.' if is_discrete else 'El p-valor de K-S ('+str(round(top['K-S p-valor'],3))+') > 0.05 indica un buen ajuste.'}")

# 5️⃣ EXPORTACIÓN
if st.button(" Descargar Reporte CSV"):
    csv = df_res.drop(columns=["Parámetros"]).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resultados_ajuste.csv">Descargar CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
