import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import io, base64

st.set_page_config(page_title="FitDistPro", layout="wide")
st.title(" FitDistPro - Ajuste de Distribuciones de Probabilidad")

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
    # Limpiar y procesar datos pegados
    raw_data = paste_data.replace("\n", ",").replace(";", ",").replace("  ", " ").strip()
    raw_list = [x.strip() for x in raw_data.split(",") if x.strip()]
    
    # Convertir a float con manejo de errores
    clean_data = []
    for x in raw_list:
        try:
            clean_data.append(float(x.replace(",", ".")))
        except ValueError:
            pass  # Ignorar valores no numéricos
    
    data = np.array(clean_data)
    
    if len(data) == 0:
        st.error("❌ No se encontraron datos numéricos válidos. Verifica el formato.")
        st.stop()
else:
    data = np.random.normal(loc=50, scale=10, size=200)  # Demo
    st.warning("Usando datos de demostración. Sube tus propios datos.")

st.sidebar.metric("Registros válidos", len(data))

# Detección automático
is_discrete = np.all(np.mod(data, 1) == 0) and len(data) > 10
tipo = "Discreto (conteos)" if is_discrete else "Continuo (mediciones)"
st.sidebar.info(f" Tipo detectado: {tipo}")

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
            try:
                mu = data.mean()
                var = data.var()
                # NB solo aplica si hay sobredispersión (var > media)
                if var <= mu:
                    return None
                
                # Estimación por momentos (fórmula exacta para nbinom)
                p = mu / var
                r = mu * p / (1 - p)
                
                # Límites de seguridad numérica
                r = max(r, 0.5)
                p = min(max(p, 0.01), 0.99)
                
                # Validar que scipy acepta los parámetros antes de continuar
                test_dist = stats.nbinom(r, p, loc=0)
                if np.any(np.isnan(test_dist.pmf(data[:5]))):
                    return None
                    
                params = (r, p, 0)
            except Exception as e:
                # Para depuración: print(e)  # Se verá en los Logs de Streamlit
                return None
            
        elif dist_name == "Geométrica":
            # Método manual más estable: p = 1/mean
            try:
                mean_val = data.mean()
                p = 1.0 / mean_val if mean_val > 0 else 0.5
                p = min(max(p, 0.01), 0.99)  # Limitar a [0.01, 0.99]
                params = (p, 0)  # p, loc
            except:
                return None
            
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

# Selección automática ESTRICTA
if is_discrete:
    # Solo distribuciones discretas del profesor
    candidates = ["Poisson", "Binomial", "Binomial Negativa", "Geométrica", "Bernoulli"]
else:
    # Solo distribuciones continuas
    candidates = ["Normal", "Lognormal", "Weibull", "Gamma", "Exponencial", "Beta", "Logística", "Gumbel", "Pareto"]

results = [fit_distribution(d, data) for d in candidates]
results = [r for r in results if r is not None]

# Ordenar por AIC (menor = mejor)
results.sort(key=lambda x: x['AIC'])

# Crear DataFrame
df_res = pd.DataFrame(results).drop(columns=["dist"])
df_res.index = df_res.index + 1  # Ranking 1, 2, 3...

# Adaptar columnas de pruebas según tipo de dato
if is_discrete:
    df_res["Prueba Estadística"] = "Chi-Cuadrado"
    df_res["p-valor"] = df_res["Chi² p-valor"]
    df_res.drop(columns=["K-S p-valor", "Chi² p-valor"], inplace=True)
else:
    df_res["Prueba Estadística"] = "Kolmogorov-Smirnov"
    df_res["p-valor"] = df_res["K-S p-valor"]
    df_res.drop(columns=["K-S p-valor", "Chi² p-valor"], inplace=True)

# Orden final de columnas
cols_order = ["Distribución", "Parámetros", "Log-Likelihood", "AIC", "BIC", "Prueba Estadística", "p-valor"]
df_res = df_res[cols_order]

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Ranking de Ajuste (menor AIC = mejor)")
    st.dataframe(df_res.round(4).style.set_properties(**{'max-width': '120px', 'text-align': 'center'}), use_container_width=True)

with col2:
    st.subheader("📊 Histograma + Ajuste (Top 1)")
    if len(results) > 0:
        best = results[0]
        dist = best["dist"]
        dist_name = best["Distribución"]
        
        fig = go.Figure()
        # Histograma de datos reales
        fig.add_trace(go.Histogram(x=data, nbinsx=30, name="Datos", opacity=0.6, histnorm='probability density'))
        
        if is_discrete:
            # Distribuciones discretas: usar PMF (barras)
            x_vals = np.arange(int(data.min()), int(data.max()) + 1)
            y_vals = dist.pmf(x_vals)
            fig.add_trace(go.Bar(x=x_vals, y=y_vals, name=f"Top 1: {dist_name}", opacity=0.8))
        else:
            # Distribuciones continuas: usar PDF (línea)
            x_vals = np.linspace(data.min(), data.max(), 200)
            y_vals = dist.pdf(x_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, name=f"Top 1: {dist_name}", line=dict(color='red', width=3)))
            
        fig.update_layout(barmode='overlay', template="plotly_white", xaxis_title="Valor", yaxis_title="Densidad / Probabilidad")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(" No se pudo ajustar ninguna distribución con estos datos.")

# REPORTE AUTOMÁTICO
st.markdown("---")
st.subheader("Conclusión Estadística")
if len(results) > 0:
    top = df_res.iloc[0]
    test = top["Prueba Estadística"]
    p = top["p-valor"]
    msg_ajuste = " Ajuste ACEPTADO" if p > 0.05 else " Ajuste RECHAZADO"
    st.success(f"**Distribución Ganadora**: `{top['Distribución']}` | AIC: {top['AIC']:.2f} | BIC: {top['BIC']:.2f}\n"
               f"**{test}**: p-valor = {p:.4f} → {msg_ajuste}\n"
               f"💡 *Parámetros estimados: {top['Parámetros']}*")
