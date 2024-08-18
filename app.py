import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
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

# Novo botão para a página NLP
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

# Botão de upload de múltiplos arquivos CSV
uploaded_files = st.sidebar.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

# Inicializar o estado da sessão para a página principal
if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview Data'

if uploaded_files:
    # Carregar e concatenar os datasets com a codificação correta
    dfs = []
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    # Converter a coluna "Date" para datetime sem exibir a mensagem de aviso
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')

    # Remover qualquer linha com datas inválidas (NaT)
    combined_df = combined_df.dropna(subset=['Date'])

    # Aplicar filtros (sem mostrar no layout)
    regions = combined_df['Dealer_Region'].unique()
    min_date = combined_df['Date'].min().date()
    max_date = combined_df['Date'].max().date()
    selected_region = regions  # Aplica automaticamente todas as regiões
    selected_dates = [min_date, max_date]  # Aplica automaticamente o intervalo completo

    # Converter selected_dates para datetime64
    selected_dates = pd.to_datetime(selected_dates)

    # Filtrando o DataFrame para todas as páginas
    filtered_df = combined_df[(combined_df['Dealer_Region'].isin(selected_region)) & 
                              (combined_df['Date'].between(selected_dates[0], selected_dates[1]))]

    # Página: Visão Geral Dados
    if st.session_state['page'] == "Overview":
        st.title('Dashboard Yamaha - Overview Data')
        # [seu código de visualização para Overview Data]

    # Página: NLP (Nova página)
    elif st.session_state['page'] == "NLP":
        st.title('Dashboard Yamaha - NLP Analysis')
        st.write("Esta é a página para análise de NLP.")
        # Adicione seu código para análise de NLP aqui

    # Página: Vendas Regionais
    elif st.session_state['page'] == "Regional Sales":
        st.title('Dashboard Yamaha - Regional Sales')
        # [seu código de visualização para Regional Sales]

    # Página: Vendas Carros
    elif st.session_state['page'] == "Vendas Carros":
        st.title('Dashboard Yamaha - Vehicle Sales')
        # [seu código de visualização para Vendas Carros]

    # Página: Perfil do Cliente
    elif st.session_state['page'] == "Perfil do Cliente":
        st.title('Dashboard Yamaha - Customer Profile')
        # [seu código de visualização para Perfil do Cliente]
else:
    st.warning("Por favor, carregue um ou mais arquivos CSV para visualizar os dados.")
