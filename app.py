import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
        col1 = st.columns(1)
        with col1[0]:
            if st.button("Sentiment Analysis"):
                st.session_state['chart_type'] = "Sentiment Analysis"

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

    # Página: Visão Geral Dados
    elif st.session_state['page'] == "Overview":
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
            st.dataframe(df1, width=1500, height=600)

        elif st.session_state['chart_type'] == 'Unique Values':
            unique_counts = df1.nunique()
            st.write("Count unique values ​​per column:")
            st.write(unique_counts)

        elif st.session_state['chart_type'] == 'Download Dataset':
            st.download_button(
                label="Download Full Dataset",
                data=df1.to_csv(index=False),
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
            sales_by_region = filtered_df1['Dealer_Region'].value_counts().reset_index()
            sales_by_region.columns = ['Dealer_Region', 'count']
            fig1 = px.pie(sales_by_region, names='Dealer_Region', values='count', title='Sales by Region')
            st.plotly_chart(fig1)

        elif st.session_state['chart_type'] == 'Evolução de Vendas':
            sales_over_time = filtered_df1.groupby('Date').size().reset_index(name='Counts')
            fig4 = px.line(sales_over_time, x='Date', y='Counts', title='Sales Evolution Over Time')
            st.plotly_chart(fig4)

        elif st.session_state['chart_type'] == 'Evolução de Vendas por Região':
            sales_over_time_region = df1.groupby([df1['Date'].dt.to_period('M'), 'Dealer_Region']).size().unstack().fillna(0).reset_index()
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
            selected_model_time_series = st.selectbox('Select Vehicle Model', df1['Model'].unique())

            def plot_sales(region, model):
                sales_time = df1[(df1['Dealer_Region'] == region) & (df1['Model'] == model)].groupby(df1['Date'].dt.to_period('M')).size()
                plt.figure(figsize=(14, 8))
                sales_time.plot(kind='line', marker='o', color='#FF7F0E', linewidth=2, markersize=6)
                plt.title(f'Monthly Sales - Region: {region}, Model: {model}', fontsize=16)
                plt.xlabel('Mês', fontsize=14)
                plt.ylabel('Number of Sales', fontsize=14)
                plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.gca().spines['top'].set_color('none')
                plt.gca().spines['right'].set_color('none')
                plt.gca().set_facecolor('white')
                plt.gca().xaxis.label.set_color('black')
                plt.gca().yaxis.label.set_color('black')
                plt.gca().title.set_color('black')
                plt.gca().tick_params(axis='x', colors='black')
                plt.gca().tick_params(axis='y', colors='black')
                st.pyplot(plt)

            plot_sales(selected_region_time_series, selected_model_time_series)

        elif st.session_state['chart_type'] == 'Heatmap do Mix de Produtos':
            mix_product_region = df1.groupby(['Dealer_Region', 'Body Style']).size().unstack().fillna(0)
            plt.figure(figsize=(12, 8))
            sns.heatmap(mix_product_region, annot=True, cmap='coolwarm', fmt='g')

            # Assegurando que as legendas e rótulos sejam visíveis
            plt.title('Product Mix by Region (Body Style)', fontsize=16)
            plt.xlabel('Body Style', fontsize=14)
            plt.ylabel('Reseller Region', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            st.pyplot(plt)

    # Página: Vendas Carros
    elif st.session_state['page'] == "Vendas Carros":
        st.title('Dashboard Yamaha - Vehicle Sales')

        # Inicializar o estado da sessão para os gráficos se ainda não foi definido
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Receita Média por Tipo de Carro'

        # Botões no topo para escolher o gráfico
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Average Revenue by Car Type"):
                st.session_state['chart_type'] = "Receita Média por Tipo de Carro"
        with col2:
            if st.button("Top 10 Companies by Revenue"):
                st.session_state['chart_type'] = "Top 10 Empresas por Receita"
        with col3:
            if st.button("Transmission Distribution by Engine"):
                st.session_state['chart_type'] = "Distribuição de Transmissão por Motor"

        # Exibir o gráfico com base na escolha do botão
        if st.session_state['chart_type'] == 'Receita Média por Tipo de Carro':
            avg_price_by_body = filtered_df1.groupby('Body Style')['Price ($)'].mean().reset_index()
            fig2 = px.bar(avg_price_by_body, x='Body Style', y='Price ($)', title='Average Revenue by Car Type')
            st.plotly_chart(fig2)

        elif st.session_state['chart_type'] == 'Top 10 Empresas por Receita':
            top_companies = filtered_df1.groupby('Company')['Price ($)'].sum().reset_index().sort_values(by='Price ($)', ascending=False).head(10)
            fig5 = px.bar(top_companies, x='Company', y='Price ($)', title='Top 10 Companies by Revenue')
            st.plotly_chart(fig5)

        elif st.session_state['chart_type'] == 'Distribuição de Transmissão por Motor':
            transmission_distribution = filtered_df1.groupby(['Engine', 'Transmission']).size().reset_index(name='Counts')
            fig6 = px.bar(transmission_distribution, x='Engine', y='Counts', color='Transmission', barmode='group', title='Transmission Distribution by Engine')
            st.plotly_chart(fig6)

    # Página: Perfil do Cliente
    elif st.session_state['page'] == "Perfil do Cliente":
        st.title('Dashboard Yamaha - Customer Profile')

        # Inicializar o estado da sessão para os gráficos se ainda não foi definido
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Distribuição de Gênero por Região'

        # Botões no topo para escolher o gráfico
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Gender Distribution by Region"):
                st.session_state['chart_type'] = "Distribuição de Gênero por Região"
        with col2:
            if st.button("Top 10 Models by Gender"):
                st.session_state['chart_type'] = "Top 10 Modelos por Gênero"

        # Exibir o gráfico com base na escolha do botão
        if st.session_state['chart_type'] == 'Distribuição de Gênero por Região':
            gender_distribution = filtered_df1.groupby(['Dealer_Region', 'Gender']).size().reset_index(name='Counts')
            fig3 = px.bar(gender_distribution, x='Dealer_Region', y='Counts', color='Gender', barmode='group', title='Gender Distribution by Region')
            st.plotly_chart(fig3)

        elif st.session_state['chart_type'] == 'Top 10 Modelos por Gênero':
            top_10_male_models = filtered_df1[filtered_df1['Gender'] == 'Male']['Model'].value_counts().head(10)
            top_10_female_models = filtered_df1[filtered_df1['Gender'] == 'Female']['Model'].value_counts().head(10)

            top_10_models_df = pd.DataFrame({
                'Male': top_10_male_models,
                'Female': top_10_female_models
            }).fillna(0)

            top_10_models_df_sorted = top_10_models_df.sort_values(by=['Male', 'Female'], ascending=False)

            fig7 = px.bar(top_10_models_df_sorted, 
                          x=top_10_models_df_sorted.index, 
                          y=['Male', 'Female'], 
                          title='Top 10 Models by Gender',
                          labels={'value': 'Number of Sales', 'index': 'Models'},
                          barmode='group')

            st.plotly_chart(fig7)
else:
    st.warning("Por favor, carregue os dois arquivos CSV para visualizar os dados.")
