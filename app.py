import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import nltk
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import plotly.io as pio

# Baixar stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Caminho para a imagem
image_path = 'https://raw.githubusercontent.com/lcbueno/streamlit/main/yamaha.png'

# Exibir a imagem na barra lateral
st.sidebar.image(image_path, use_column_width=True)

# Estilo da barra lateral (mantendo a cor azul nos botões)
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #262730;
            padding: 10px;
        }
        .sidebar .sidebar-content h2 {
            color: white;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .stButton > button {
            font-size: 18px;
            color: white;
            background-color: #1F77B4;
            border: none;
            padding: 10px 20px;
            margin-bottom: 10px;
            width: 100%;
            text-align: left;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #0073e6;
        }
        .stButton > button:focus {
            background-color: #005bb5;
        }
        .stContainer > div {
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar para seleção da página principal
st.sidebar.title("Analytical Dashboard")

# Novo botão "NLP" adicionado acima do botão "Overview Data"
if st.sidebar.button("NLP"):
    st.session_state['page'] = 'NLP'

if st.sidebar.button("Overview Data"):
    st.session_state['page'] = 'Overview'
if st.sidebar.button("Regional Sales"):
    st.session_state['page'] = 'Regional Sales'
if st.sidebar.button("Vehicle Sales"):
    st.session_state['page'] = 'Vendas Carros'
if st.sidebar.button("Customer Profile"):
    st.session_state['page'] = 'Perfil do Cliente'

# Botões de upload para dois arquivos CSV diferentes
uploaded_file_1 = st.sidebar.file_uploader("Choose first CSV file", type="csv")
uploaded_file_2 = st.sidebar.file_uploader("Choose second CSV file", type="csv")

# Inicializar o estado da sessão para a página principal
if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview Data'

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    # Carregar os datasets com a codificação correta
    df1 = pd.read_csv(uploaded_file_1, encoding='ISO-8859-1')
    df2 = pd.read_csv(uploaded_file_2, encoding='ISO-8859-1')

    # Verificar se a coluna 'Date' existe no primeiro dataframe
    if 'Date' in df1.columns:
        # Converter a coluna "Date" para datetime sem exibir a mensagem de aviso
        df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True, errors='coerce')
        # Remover qualquer linha com datas inválidas (NaT)
        df1 = df1.dropna(subset=['Date'])
    else:
        st.error("A coluna 'Date' não foi encontrada no primeiro arquivo CSV. Verifique o arquivo e tente novamente.")

    # Verificar se a coluna 'sentiment score' existe no segundo dataframe
    if 'sentiment score' in df2.columns:
        # Extrair os componentes do sentimento de forma correta
        df_sentiment_scores = pd.json_normalize(df2['sentiment score'].apply(eval))
        df2['sentiment_pos'] = df_sentiment_scores['pos']
        df2['sentiment_neg'] = df_sentiment_scores['neg']
        df2['sentiment_neu'] = df_sentiment_scores['neu']
    else:
        st.error("A coluna 'sentiment score' não foi encontrada no segundo arquivo CSV. Verifique o arquivo e tente novamente.")

    # Aplicar filtros (sem mostrar no layout) no primeiro dataset
    regions = list(df1['Dealer_Region'].unique()) if 'Dealer_Region' in df1.columns else []
    min_date = df1['Date'].min().date() if 'Date' in df1.columns else None
    max_date = df1['Date'].max().date() if 'Date' in df1.columns else None
    selected_region = regions if regions else []  # Verifica se há regiões disponíveis
    selected_dates = [min_date, max_date] if min_date and max_date else []  # Aplica automaticamente o intervalo completo

    # Converter selected_dates para datetime64 se houver datas válidas
    if selected_dates and len(selected_dates) == 2:
        selected_dates = pd.to_datetime(selected_dates)

    # Verifica se há regiões e datas selecionadas para filtrar
    if selected_region and len(selected_dates) == 2:
        filtered_df1 = df1[(df1['Dealer_Region'].isin(selected_region)) & 
                           (df1['Date'].between(selected_dates[0], selected_dates[1]))]
    else:
        filtered_df1 = df1

    # Página: NLP
    if st.session_state['page'] == "NLP":
        st.title('Dashboard Yamaha - NLP')

        # Inicializar o estado da sessão para os gráficos se ainda não foi definido
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Sentiment Analysis'

        # Botões no topo para escolher o gráfico
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sentiment Analysis"):
                st.session_state['chart_type'] = "Sentiment Analysis"
        with col2:
            if st.button("Salient Terms"):
                st.session_state['chart_type'] = "Salient Terms"

        # Exibir o gráfico com base na escolha do botão
        if st.session_state['chart_type'] == 'Sentiment Analysis':
            # Calcular a média dos sentimentos por marca
            brand_sentiment = df2.groupby('brand_name').agg({
                'sentiment_pos': 'mean',
                'sentiment_neg': 'mean',
                'sentiment_neu': 'mean'
            }).reset_index()

            # Transformar os dados para um formato longo para facilitar a plotagem
            brand_sentiment_melted = brand_sentiment.melt(id_vars='brand_name', 
                                                          value_vars=['sentiment_pos', 'sentiment_neg', 'sentiment_neu'],
                                                          var_name='Sentimento', value_name='Média')

            # Mapeamento de nomes mais legíveis
            brand_sentiment_melted['Sentimento'] = brand_sentiment_melted['Sentimento'].map({
                'sentiment_pos': 'Positivo',
                'sentiment_neg': 'Negativo',
                'sentiment_neu': 'Neutro'
            })

            # Criar gráfico interativo usando Plotly
            fig = px.bar(brand_sentiment_melted, 
                         x='brand_name', 
                         y='Média', 
                         color='Sentimento', 
                         barmode='group',
                         labels={'brand_name': 'Marca', 'Média': 'Sentimento Médio'},
                         title='Comparação de Sentimentos por Marca')

            # Exibir o gráfico interativo
            st.plotly_chart(fig)

        elif st.session_state['chart_type'] == 'Salient Terms':
            # Garantir que a coluna 'review' tenha valores válidos e tratar NaNs
            df2['review'] = df2['review'].astype(str).fillna('')

            # Pré-processamento de texto
            df2['processed_review'] = df2['review'].str.lower().str.split().apply(
                lambda x: [word for word in x if word not in stop_words]
            )

            # Criar dicionário e corpus para LDA
            dictionary = corpora.Dictionary(df2['processed_review'])
            corpus = [dictionary.doc2bow(text) for text in df2['processed_review']]

            # Rodar o modelo LDA
            lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

            # Preparar visualização interativa com pyLDAvis
            lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

            # Exibir visualização interativa
            st.write("Visualização Interativa LDA:")
            pyLDAvis.display(lda_vis)

# Código restante para outras páginas...
else:
    st.warning("Por favor, carregue os dois arquivos CSV para visualizar os dados.")
