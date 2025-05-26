import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st


from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from io import BytesIO

# Configuracao inicial do Streamlit
st.set_page_config(
    page_title='K-means e Segmentacao',
    layout="wide",
    initial_sidebar_state='expanded'
)

@st.cache_data

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    return output.getvalue()

st.title("Clusterização com K-means")

st.markdown("""
Este app aplica o algoritmo de clusterização K-means após pré-processamento e padronização dos dados.

Faça upload de um arquivo CSV com suas variáveis numéricas.
""")

st.sidebar.header("📁 Suba seu arquivo")
file = st.sidebar.file_uploader("Arquivo CSV", type=['csv'])

if file is not None:
    df = pd.read_csv(file)
    st.subheader("Visualização Inicial dos Dados")
    st.dataframe(df.head())

    # Verificação de dados
    st.write("Tipos de dados:")
    st.write(df.dtypes)

    st.write("Dados faltantes:")
    st.write(df.isna().sum())

    # Selecionar colunas numéricas
    colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
    st.write("Colunas numéricas detectadas:", colunas_numericas)

    if len(colunas_numericas) < 2:
        st.warning("São necessárias pelo menos duas colunas numéricas para aplicar K-means.")
    else:
        X = df[colunas_numericas].dropna()

        # Padronização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Método do cotovelo
        st.subheader("Método do Cotovelo")
        SQD = []
        K = range(1, 11)
        for k in K:
            km = KMeans(n_clusters=k, n_init=10, algorithm="lloyd", random_state=42)
            km.fit(X_scaled)
            SQD.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(K, SQD, 'bo-')
        ax.set_xlabel('Número de Clusters (k)')
        ax.set_ylabel('Soma dos Quadrados das Distâncias (SQD)')
        ax.set_title('Método do Cotovelo')
        st.pyplot(fig)

        # Selecionar K ideal
        k_user = st.slider("Selecione o número de clusters (k)", 2, 10, 3)

        # KMeans final
        km_final = KMeans(n_clusters=k_user, n_init=10, algorithm="lloyd", random_state=42)
        labels = km_final.fit_predict(X_scaled)

        # Score silhueta
        sil_score = silhouette_score(X_scaled, labels)
        st.write(f"Score da silhueta para k={k_user}: {sil_score:.4f}")

        # Anexar rótulos ao dataframe original
        df_resultado = df.copy()
        df_resultado['Cluster'] = labels

        st.subheader("Resultado da Clusterização")
        st.dataframe(df_resultado.head())

        
        # PCA para visualização
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = labels

        st.subheader("Visualização PCA dos Clusters")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax2)
        ax2.set_title('Clusters visualizados em 2D via PCA')
        st.pyplot(fig2)

        # Download resultado
        df_xlsx = to_excel(df_resultado)
        st.download_button(label="📥 Baixar Resultado", data=df_xlsx, file_name='cluster_resultado.xlsx')
