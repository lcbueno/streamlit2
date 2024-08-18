import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import bigrams
from nltk.corpus import stopwords
from collections import Counter
import nltk

# Baixar recursos necessários do nltk
nltk.download('stopwords')

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
if st.sidebar.button("Overview Data"):
    st.session_state['page'] = 'Overview'
if st.sidebar.button("Regional Sales"):
    st.session_state['page'] = 'Regional Sales'
if st.sidebar.button("Vehicle Sales"):
    st.session_state['page'] = 'Vendas Carros'
if st.sidebar.button("Customer Profile"):
    st.session_state['page'] = 'Perfil do Cliente'
if st.sidebar.button("NLP"):
    st.session_state['page'] = 'NLP'  # Adiciona a funcionalidade do botão NLP

# Botão de upload do arquivo CSV abaixo dos botões de seleção de página
uploaded_files = st.sidebar.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

# Inicializar o estado da sessão para a página principal
if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview Data'

# Variáveis para armazenar os DataFrames processados
df_sales = None
df_nlp = None

# Processar os arquivos CSV carregados
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        # Verifica se o arquivo contém a coluna 'Date' para o dataset de vendas
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            df_sales = df
        # Verificar se o arquivo carregado é o dataset de NLP
        elif 'review' in df.columns:
            df_nlp = df
        else:
            st.warning(f"O arquivo {uploaded_file.name} não contém as colunas necessárias para análise e será ignorado.")

# Tratamento do dataset de vendas
if df_sales is not None and st.session_state['page'] != "NLP":
    # Aplicar filtros (sem mostrar no layout)
    regions = df_sales['Dealer_Region'].unique()
    min_date = df_sales['Date'].min().date()
    max_date = df_sales['Date'].max().date()
    selected_region = regions  # Aplica automaticamente todas as regiões
    selected_dates = [min_date, max_date]  # Aplica automaticamente o intervalo completo

    selected_dates = pd.to_datetime(selected_dates)

    filtered_df = df_sales[(df_sales['Dealer_Region'].isin(selected_region)) & 
                           (df_sales['Date'].between(selected_dates[0], selected_dates[1]))]

    # Página: Visão Geral Dados
    if st.session_state['page'] == "Overview":
        st.title('Dashboard Yamaha - Overview Data')

        # Inicializar o estado da sessão para os gráficos se ainda não foi definido
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Overview'

        # Botões no topo para escolher o gráfico
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Overview"):
                st.session_state['chart_type'] = "Overview"
        with col2:
            if st.button("Unique Values"):
                st.session_state['chart_type'] = "Unique Values"
        with col3:
            if st.button("Download Dataset"):
                st.session_state['chart_type'] = "Download Dataset"

        # Exibir o gráfico com base na escolha do botão
        if st.session_state['chart_type'] == 'Overview':
            st.write("DataFrame Visualization:")
            st.dataframe(filtered_df, width=1500, height=600)

        elif st.session_state['chart_type'] == 'Unique Values':
            unique_counts = filtered_df.nunique()
            st.write("Count unique values ​​per column:")
            st.write(unique_counts)

        elif st.session_state['chart_type'] == 'Download Dataset':
            st.download_button(
                label="Download Full Dataset",
                data=filtered_df.to_csv(index=False),
                file_name='dataset_completo.csv',
                mime='text/csv',
            )

    # Página: Vendas Regionais
    elif st.session_state['page'] == "Regional Sales":
        st.title('Dashboard Yamaha - Regional Sales')

        # Inicializar o estado da sessão para os gráficos se ainda não foi definido
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Distribuição de Vendas por Região'

        # Botões no topo para escolher o gráfico
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("Sales by Region"):
                st.session_state['chart_type'] = "Distribuição de Vendas por Região"
        with col2:
            if st.button("Sales Evolution Over Time"):
                st.session_state['chart_type'] = "Evolução de Vendas"
        with col3:
            if st.button("Sales Evolution by Region"):
                st.session_state['chart_type'] = "Evolução de Vendas por Região"
        with col4:
            if st.button("Region x Vehicle Model"):
                st.session_state['chart_type'] = "Séries Temporais por Região e Modelo"
        with col5:
            if st.button("Product Mix Heatmap"):
                st.session_state['chart_type'] = "Heatmap do Mix de Produtos"

        # Exibir o gráfico com base na escolha do botão
        if st.session_state['chart_type'] == 'Distribuição de Vendas por Região':
            sales_by_region = filtered_df['Dealer_Region'].value_counts().reset_index()
            sales_by_region.columns = ['Dealer_Region', 'count']
            fig1 = px.pie(sales_by_region, names='Dealer_Region', values='count', title='Sales by Region')
            st.plotly_chart(fig1)

        elif st.session_state['chart_type'] == 'Evolução de Vendas':
            sales_over_time = filtered_df.groupby('Date').size().reset_index(name='Counts')
            fig4 = px.line(sales_over_time, x='Date', y='Counts', title='Sales Evolution Over Time')
            st.plotly_chart(fig4)

        elif st.session_state['chart_type'] == 'Evolução de Vendas por Região':
            sales_over_time_region = filtered_df.groupby([filtered_df['Date'].dt.to_period('M'), 'Dealer_Region']).size().unstack().fillna(0).reset_index()
            sales_over_time_region['Date'] = sales_over_time_region['Date'].astype(str)

            fig9 = px.line(sales_over_time_region, 
                           x='Date', 
                           y=sales_over_time_region.columns[1:], 
                           title='Evolution of Sales Over Time by Region',
                           labels={'value': 'Number of Sales', 'Date': 'Month'},
                           color_discrete_sequence=px.colors.qualitative.Set1)

            st.plotly_chart(fig9)

        elif st.session_state['chart_type'] == 'Séries Temporais por Região e Modelo':
            selected_region_time_series = st.selectbox('Select Region', regions)
            selected_model_time_series = st.selectbox('Select Vehicle Model', filtered_df['Model'].unique())

            def plot_sales(region, model):
                sales_time = filtered_df[(filtered_df['Dealer_Region'] == region) & (filtered_df['Model'] == model)].groupby(filtered_df['Date'].dt.to_period('M')).size()
                plt.figure(figsize=(14, 8))
                sales_time.plot(kind='line', marker='o', color='#FF7F0E', linewidth=2, markersize=6)
                plt.title(f'Monthly Sales - Region: {region} | Model: {model}')
                plt.xlabel('Month')
                plt.ylabel('Number of Sales')
                plt.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(plt)

            plot_sales(selected_region_time_series, selected_model_time_series)

        elif st.session_state['chart_type'] == 'Heatmap do Mix de Produtos':
            product_mix = filtered_df.groupby(['Dealer_Region', 'Model']).size().unstack().fillna(0)
            fig12 = px.imshow(product_mix, labels=dict(x="Model", y="Region", color="Count"), title="Product Mix Heatmap")
            st.plotly_chart(fig12)

# Página: Vendas Carros
elif df_sales is not None and st.session_state['page'] == "Vendas Carros":
    st.title('Dashboard Yamaha - Vehicle Sales')

    # Inicializar o estado da sessão para os gráficos se ainda não foi definido
    if 'chart_type' not in st.session_state:
        st.session_state['chart_type'] = 'Vendas por Modelo'

    # Botões no topo para escolher o gráfico
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Sales by Model"):
            st.session_state['chart_type'] = "Vendas por Modelo"
    with col2:
        if st.button("Sales by Region and Model"):
            st.session_state['chart_type'] = "Vendas por Região e Modelo"
    with col3:
        if st.button("Sales Distribution"):
            st.session_state['chart_type'] = "Distribuição de Vendas"
    with col4:
        if st.button("Sales Heatmap"):
            st.session_state['chart_type'] = "Heatmap de Vendas"
    with col5:
        if st.button("Sales Trend Analysis"):
            st.session_state['chart_type'] = "Análise de Tendência de Vendas"

    # Exibir o gráfico com base na escolha do botão
    if st.session_state['chart_type'] == 'Vendas por Modelo':
        sales_by_model = filtered_df['Model'].value_counts().reset_index()
        sales_by_model.columns = ['Model', 'Count']
        fig1 = px.bar(sales_by_model, x='Model', y='Count', title='Sales by Vehicle Model')
        st.plotly_chart(fig1)

    elif st.session_state['chart_type'] == 'Vendas por Região e Modelo':
        sales_by_region_model = filtered_df.groupby(['Dealer_Region', 'Model']).size().unstack().fillna(0)
        fig4 = px.imshow(sales_by_region_model, labels=dict(x="Model", y="Region", color="Count"), title='Sales by Region and Model')
        st.plotly_chart(fig4)

    elif st.session_state['chart_type'] == 'Distribuição de Vendas':
        sales_distribution = filtered_df['Dealer_Region'].value_counts().reset_index()
        sales_distribution.columns = ['Region', 'Count']
        fig8 = px.pie(sales_distribution, names='Region', values='Count', title='Sales Distribution by Region')
        st.plotly_chart(fig8)

    elif st.session_state['chart_type'] == 'Heatmap de Vendas':
        sales_heatmap = filtered_df.groupby(['Date', 'Model']).size().unstack().fillna(0)
        fig12 = px.imshow(sales_heatmap, labels=dict(x="Model", y="Date", color="Count"), title='Sales Heatmap')
        st.plotly_chart(fig12)

    elif st.session_state['chart_type'] == 'Análise de Tendência de Vendas':
        sales_trend = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).size()
        plt.figure(figsize=(14, 8))
        sales_trend.plot(kind='line', marker='o', color='#FF7F0E', linewidth=2, markersize=6)
        plt.title('Sales Trend Analysis')
        plt.xlabel('Month')
        plt.ylabel('Number of Sales')
        plt.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(plt)

# Página: Perfil do Cliente
elif st.session_state['page'] == "Perfil do Cliente":
    st.title('Customer Profile Dashboard')

    # Inicializar o estado da sessão para os gráficos se ainda não foi definido
    if 'chart_type' not in st.session_state:
        st.session_state['chart_type'] = 'Customer Demographics'

    # Botões no topo para escolher o gráfico
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Customer Demographics"):
            st.session_state['chart_type'] = "Customer Demographics"
    with col2:
        if st.button("Sales Analysis"):
            st.session_state['chart_type'] = "Sales Analysis"
    with col3:
        if st.button("Download Dataset"):
            st.session_state['chart_type'] = "Download Dataset"

    # Exibir o gráfico com base na escolha do botão
    if st.session_state['chart_type'] == 'Customer Demographics':
        demographics = df_sales.groupby('Customer_Segment').size().reset_index(name='Counts')
        fig2 = px.bar(demographics, x='Customer_Segment', y='Counts', title='Customer Demographics')
        st.plotly_chart(fig2)

    elif st.session_state['chart_type'] == 'Sales Analysis':
        sales_analysis = df_sales.groupby(['Customer_Segment', 'Model']).size().unstack().fillna(0)
        fig3 = px.imshow(sales_analysis, labels=dict(x="Model", y="Customer Segment", color="Count"), title='Sales Analysis by Customer Segment and Model')
        st.plotly_chart(fig3)

    elif st.session_state['chart_type'] == 'Download Dataset':
        st.download_button(
            label="Download Customer Dataset",
            data=df_sales.to_csv(index=False),
            file_name='customer_data.csv',
            mime='text/csv',
        )

# Página: NLP
elif df_nlp is not None and st.session_state['page'] == "NLP":
    st.title('NLP Dashboard')

    # Inicializar o estado da sessão para os gráficos se ainda não foi definido
    if 'nlp_chart_type' not in st.session_state:
        st.session_state['nlp_chart_type'] = 'Sentiment Analysis'

    # Botões no topo para escolher o gráfico
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Sentiment Analysis"):
            st.session_state['nlp_chart_type'] = "Sentiment Analysis"
    with col2:
        if st.button("Word Cloud"):
            st.session_state['nlp_chart_type'] = "Word Cloud"
    with col3:
        if st.button("Bigramas"):
            st.session_state['nlp_chart_type'] = "Bigramas"

    # Função para gerar a nuvem de palavras
    def generate_word_cloud(text):
        stop_words = set(stopwords.words('portuguese'))
        wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    # Função para gerar a análise de sentimentos
    def analyze_sentiments(text):
        from textblob import TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment.polarity, sentiment.subjectivity

    # Função para gerar bigramas
    def generate_bigrams(text):
        stop_words = set(stopwords.words('portuguese'))
        tokens = [word for word in text.lower().split() if word not in stop_words]
        bigram_list = list(bigrams(tokens))
        bigram_freq = Counter(bigram_list)
        return bigram_freq

    # Exibir o gráfico com base na escolha do botão
    if st.session_state['nlp_chart_type'] == 'Sentiment Analysis':
        sentiment_polarity, sentiment_subjectivity = analyze_sentiments(' '.join(df_nlp['review'].dropna()))
        st.write(f"Sentiment Polarity: {sentiment_polarity}")
        st.write(f"Sentiment Subjectivity: {sentiment_subjectivity}")

    elif st.session_state['nlp_chart_type'] == 'Word Cloud':
        generate_word_cloud(' '.join(df_nlp['review'].dropna()))

    elif st.session_state['nlp_chart_type'] == 'Bigramas':
        bigram_freq = generate_bigrams(' '.join(df_nlp['review'].dropna()))
        bigrams_df = pd.DataFrame(bigram_freq.items(), columns=['Bigram', 'Frequency'])
        fig = px.bar(bigrams_df, x='Bigram', y='Frequency', title='Bigram Frequency')
        st.plotly_chart(fig)
