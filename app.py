import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk import bigrams
from collections import Counter

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
                plt.title(f'Monthly Sales - Region: {region}, Model: {model}')
                plt.xlabel('Month')
                plt.ylabel('Number of Sales')
                plt.grid(True)
                plt.xticks(rotation=45)
                st.pyplot(plt)

            plot_sales(selected_region_time_series, selected_model_time_series)

        elif st.session_state['chart_type'] == 'Heatmap do Mix de Produtos':
            product_sales = filtered_df.groupby(['Dealer_Region', 'Product']).size().unstack().fillna(0)
            fig8 = px.imshow(product_sales, 
                             title='Product Mix Heatmap',
                             labels={'x': 'Product', 'y': 'Region'},
                             color_continuous_scale='Viridis')
            st.plotly_chart(fig8)

# Página: Vendas de Carros
elif st.session_state['page'] == "Vendas Carros":
    st.title('Dashboard Yamaha - Vendas de Carros')

    # Inicializar o estado da sessão para os gráficos se ainda não foi definido
    if 'chart_type' not in st.session_state:
        st.session_state['chart_type'] = 'Car Sales Analysis'

    # Botões no topo para escolher o gráfico
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sales Overview"):
            st.session_state['chart_type'] = "Car Sales Analysis"
    with col2:
        if st.button("Vehicle Sales by Region"):
            st.session_state['chart_type'] = "Sales by Region"

    # Exibir o gráfico com base na escolha do botão
    if st.session_state['chart_type'] == 'Car Sales Analysis':
        # Inserir o código para análise de vendas de carros aqui
        st.write("Car Sales Analysis")

    elif st.session_state['chart_type'] == 'Sales by Region':
        # Inserir o código para análise de vendas de carros por região aqui
        st.write("Vehicle Sales by Region")

# Página: Perfil do Cliente
elif st.session_state['page'] == "Perfil do Cliente":
    st.title('Dashboard Yamaha - Perfil do Cliente')

    # Inicializar o estado da sessão para os gráficos se ainda não foi definido
    if 'chart_type' not in st.session_state:
        st.session_state['chart_type'] = 'Customer Profile'

    # Botões no topo para escolher o gráfico
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Customer Overview"):
            st.session_state['chart_type'] = "Customer Profile"
    with col2:
        if st.button("Customer Demographics"):
            st.session_state['chart_type'] = "Customer Demographics"

    # Exibir o gráfico com base na escolha do botão
    if st.session_state['chart_type'] == 'Customer Profile':
        # Inserir o código para análise do perfil do cliente aqui
        st.write("Customer Profile")

    elif st.session_state['chart_type'] == 'Customer Demographics':
        # Inserir o código para análise das características do cliente aqui
        st.write("Customer Demographics")

# Página: NLP
elif st.session_state['page'] == "NLP":
    st.title('Análise de NLP')

    if df_nlp is not None:
        # Botões para funcionalidades de NLP
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Análise de Sentimentos"):
                st.session_state['nlp_chart_type'] = 'Sentiment Analysis'
        with col2:
            if st.button("Word Cloud"):
                st.session_state['nlp_chart_type'] = 'Word Cloud'
        with col3:
            if st.button("Bigramas"):
                st.session_state['nlp_chart_type'] = 'Bigramas'

        # Exibir o gráfico com base na escolha do botão
        if 'nlp_chart_type' not in st.session_state:
            st.session_state['nlp_chart_type'] = 'Sentiment Analysis'

        if st.session_state['nlp_chart_type'] == 'Sentiment Analysis':
            st.write("Análise de Sentimentos")
            # Código para análise de sentimentos
            st.write("Placeholder for Sentiment Analysis")

        elif st.session_state['nlp_chart_type'] == 'Word Cloud':
            st.write("Word Cloud")
            # Gerar a nuvem de palavras
            text = ' '.join(df_nlp['review'].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        elif st.session_state['nlp_chart_type'] == 'Bigramas':
            st.write("Bigramas")
            # Gerar e exibir bigramas
            text = ' '.join(df_nlp['review'].astype(str))
            tokens = text.split()
            bigram_list = list(bigrams(tokens))
            bigram_freq = Counter(bigram_list)
            bigram_df = pd.DataFrame(bigram_freq.items(), columns=['Bigram', 'Frequency']).sort_values(by='Frequency', ascending=False)
            st.write("Bigramas Frequência:")
            st.dataframe(bigram_df)

    else:
        st.warning("Nenhum dataset NLP carregado. Por favor, faça o upload do arquivo correto.")
