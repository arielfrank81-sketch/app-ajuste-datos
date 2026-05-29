import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import io, base64

st.set_page_config(page_title="FitDistPro", layout="wide")
st.title("📊 FitDistPro - Ajuste de Distribuciones de Probabilidad")

# 1️⃣ CARGA DE DATOS
st.sidebar.header("📥 Carga de Datos")
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
        if dist_name == "Normal":
            params = stats.norm.fit(data)
            dist = stats.norm(*params)
        elif dist_name == "Lognormal":
            params = stats.lognorm.fit(data, floc=0)
            dist = stats.lognorm(*params)
        elif dist_name == "Weibull":
            params = stats.weibull_min.fit(data, floc=0)
            dist = stats.weibull_min(*params)
        elif dist_name == "Gamma":
            params = stats.gamma.fit(data, floc=0)
            dist = stats.gamma(*params)
        elif dist_name == "Exponencial":
            params = stats.expon.fit(data, floc=0)
            dist = stats.expon(*params)
        elif dist_name == "Beta":
            params = stats.beta.fit(data, floc=data.min(), scale=data.max()-data.min())
            dist = stats.beta(*params)
        elif dist_name == "Logística":
            params = stats.logistic.fit(data)
            dist = stats.logistic(*params)
        elif dist_name == "Gumbel":
            params = stats.gumbel_r.fit(data)
            dist = stats.gumbel_r(*params)
        elif dist_name == "Pareto":
            params = stats.pareto.fit(data, floc=data.min())
            dist = stats.pareto(*params)
        elif is_discrete:
            if dist_name == "Poisson":
                params = (data.mean(),)
                dist = stats.poisson(*params, loc=0)
            elif dist_name == "Binomial":
                n = max(data.max(), 10)
                p = data.mean() / n
                params = (n, p)
                dist = stats.binom(*params, loc=0)
            elif dist_name == "Binomial Negativa":
                params = stats.nbinom.fit(data, floc=0)
                dist = stats.nbinom(*params)
            elif dist_name == "Geométrica":
                params = stats.geom.fit(data, floc=0)
                dist = stats.geom(*params)
            elif dist_name == "Bernoulli":
                p = data.mean()
                params = (p,)
                dist = stats.bernoulli(*params, loc=0)
            else: return None
        else:
            return None

        # Métricas
        k = len(params)
        if is_discrete:
            loglik = np.sum(dist.logpmf(data))
            # Chi2 para discretas
            observed = np.array([np.sum(data==x) for x in np.unique(data)])
            expected = np.array([dist.pmf(x)*len(data) for x in np.unique(data)])
            chi2, p_chi2 = stats.chisquare(observed, expected)
            ks_stat, ks_p = 0, 0  # KS no aplica bien a discretas
        else:
            loglik = np.sum(dist.logpdf(data))
            ks_stat, ks_p = stats.kstest(data, dist.cdf)
            # Chi2 aproximado para continuas
            hist, edges = np.histogram(data, bins='auto')
            expected = len(data) * np.diff(dist.cdf(edges))
            chi2, p_chi2 = stats.chisquare(hist, expected+1e-8)

        aic = 2*k - 2*loglik
        bic = k*np.log(len(data)) - 2*loglik

        return {
            "Distribución": dist_name,
            "Parámetros": params,
            "Log-Likelihood": loglik,
            "AIC": aic,
            "BIC": bic,
            "K-S p-valor": ks_p,
            "Chi² p-valor": p_chi2,
            "dist": dist
        }
    except:
        return None

# Selección automática
candidates = ["Normal", "Lognormal", "Weibull", "Gamma", "Exponencial", "Logística", "Gumbel"]
if not is_discrete:
    candidates += ["Beta", "Pareto"]
else:
    candidates += ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"]

results = [fit_distribution(d, data) for d in candidates]
results = [r for r in results if r is not None]

# Ranking por AIC
df_res = pd.DataFrame(results).drop(columns=["dist"]).sort_values("AIC")
df_res.index = range(1, len(df_res)+1)
df_res.rename(columns={"Distribución": "Rank"}, inplace=False)

# 3️⃣ VISUALIZACIÓN Y RESULTADOS
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏆 Ranking de Ajuste (menor AIC = mejor)")
    st.dataframe(df_res.round(4), use_container_width=True)

with col2:
    st.subheader("📈 Histograma + Densidad (Top 1)")
    best = results[0]["dist"]
    x = np.linspace(data.min(), data.max(), 200)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Datos", opacity=0.6, histnorm='probability density'))
    fig.add_trace(go.Scatter(x=x, y=best.pdf(x) if not is_discrete else best.pmf(x), name=f"Mejor ajuste: {results[0]['Distribución']}", line=dict(color='red', width=3)))
    fig.update_layout(barmode='overlay', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# 4️⃣ REPORTE EN LENGUAJE NATURAL
st.markdown("---")
st.subheader("📝 Reporte Automático")
top = df_res.iloc[0]
st.success(f"✅ **Conclusión**: La distribución `{top['Distribución']}` es la más adecuada para tus datos. "
           f"AIC={top['AIC']:.2f}, BIC={top['BIC']:.2f}. "
           f"{'El p-valor de Chi² ('+str(round(top['Chi² p-valor'],3))+') > 0.05 indica un buen ajuste.' if is_discrete else 'El p-valor de K-S ('+str(round(top['K-S p-valor'],3))+') > 0.05 indica un buen ajuste.'}")

# 5️⃣ EXPORTACIÓN
if st.button("📥 Descargar Reporte CSV"):
    csv = df_res.drop(columns=["Parámetros"]).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resultados_ajuste.csv">Descargar CSV</a>'
    st.markdown(href, unsafe_allow_html=True)