import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objs as go
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.express as px

# Path to the image
image_path = 'https://raw.githubusercontent.com/lcbueno/streamlit/main/yamaha.png'

# Display the image in the sidebar
st.sidebar.image(image_path, use_column_width=True)

# Sidebar style (keeping the blue color on buttons)
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


# New button "Following"
if st.sidebar.button("Following"):
    st.session_state['page'] = 'Following'

# Página: Following
if st.session_state['page'] == "Following":
    st.title('Dashboard Yamaha - Following')

    # Initialize the session state for the 'Following' page buttons if not yet defined
    if 'following_chart_type' not in st.session_state:
        st.session_state['following_chart_type'] = 'Leads'

    # Top button for the "Following" page
    col1 = st.columns(1)
    with col1[0]:
        if st.button("Leads"):
            st.session_state['following_chart_type'] = "Leads"

    # Display the chart or data based on the button choice
    if st.session_state['following_chart_type'] == 'Leads':
        st.write("Leads data and analysis will be displayed here.")

# Existing buttons
if st.sidebar.button("Overview Data"):
    st.session_state['page'] = 'Overview'
if st.sidebar.button("Regional Sales"):
    st.session_state['page'] = 'Regional Sales'
if st.sidebar.button("Vehicle Sales"):
    st.session_state['page'] = 'Vehicle Sales'
if st.sidebar.button("Customer Profile"):
    st.session_state['page'] = 'Customer Profile'
if st.sidebar.button("NLP"):
    st.session_state['page'] = 'NLP'  # Adds the NLP button functionality

# CSV file upload button below the page selection buttons
uploaded_files = st.sidebar.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

# Initialize the session state for the main page
if 'page' not in st.session_state:
    st.session_state['page'] = 'Overview Data'

# The rest of your code remains intact...


# Variables to store processed DataFrames
df_sales = None
df_nlp = None

# Process the uploaded CSV files
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        # Check if the file contains the 'Date' column for the sales dataset
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date'])
            df_sales = df
        # Check if the uploaded file is the NLP dataset
        elif 'review' in df.columns:
            df_nlp = df
        # Check if the file is the american_names_with_random_download_versions_scheduled.csv dataset
        elif 'timestamp' in df.columns and 'download' in df.columns:
            df_american_names = df
        else:
            st.warning(f"The file {uploaded_file.name} does not contain the necessary columns for analysis and will be ignored.")

# Sales dataset processing
if df_sales is not None and st.session_state['page'] != "NLP":
    # Apply filters (without showing in the layout)
    regions = df_sales['Dealer_Region'].unique()
    min_date = df_sales['Date'].min().date()
    max_date = df_sales['Date'].max().date()
    selected_region = regions  # Automatically apply all regions
    selected_dates = [min_date, max_date]  # Automatically apply the full range

    selected_dates = pd.to_datetime(selected_dates)

    filtered_df = df_sales[(df_sales['Dealer_Region'].isin(selected_region)) & 
                           (df_sales['Date'].between(selected_dates[0], selected_dates[1]))]

    # Página: Overview Data
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
            st.write("Sales DataFrame Visualization:")
            st.dataframe(filtered_df, width=1500, height=600)  # Exibe o DataFrame de vendas
    
            if df_nlp is not None:
                st.write("NLP DataFrame Visualization:")
                st.dataframe(df_nlp, width=1500, height=600)  # Exibe o DataFrame de NLP
    
        elif st.session_state['chart_type'] == 'Unique Values':
            unique_counts = filtered_df.nunique()
            st.write("Count unique values per column:")
            st.write(unique_counts)
    
        elif st.session_state['chart_type'] == 'Download Dataset':
            st.download_button(
                label="Download Full Dataset",
                data=filtered_df.to_csv(index=False),
                file_name='full_dataset.csv',
                mime='text/csv',
            )
    

    # Page: Regional Sales
    elif st.session_state['page'] == "Regional Sales":
        st.title('Dashboard Yamaha - Regional Sales')

        # Initialize the session state for charts if not yet defined
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Sales Distribution by Region'

        # First row of buttons for the "Regional Sales" page
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sales by Region"):
                st.session_state['chart_type'] = "Sales Distribution by Region"
        with col2:
            if st.button("Sales Evolution Over Time"):
                st.session_state['chart_type'] = "Sales Evolution"
        
        # Second row of buttons for the "Regional Sales" page
        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button("Sales Evolution by Region"):
                st.session_state['chart_type'] = "Sales Evolution by Region"
        with col4:
            if st.button("Region x Vehicle Model"):
                st.session_state['chart_type'] = "Time Series by Region and Model"
        with col5:
            if st.button("Product Mix Heatmap"):
                st.session_state['chart_type'] = "Product Mix Heatmap"


        # Display the chart based on the button choice
        if st.session_state['chart_type'] == 'Sales Distribution by Region':
            sales_by_region = filtered_df['Dealer_Region'].value_counts().reset_index()
            sales_by_region.columns = ['Dealer_Region', 'count']
            fig1 = px.pie(sales_by_region, names='Dealer_Region', values='count', title='Sales by Region')
            st.plotly_chart(fig1)

        elif st.session_state['chart_type'] == 'Sales Evolution':
            sales_over_time = filtered_df.groupby('Date').size().reset_index(name='Counts')
            fig4 = px.line(sales_over_time, x='Date', y='Counts', title='Sales Evolution Over Time')
            st.plotly_chart(fig4)

        elif st.session_state['chart_type'] == 'Sales Evolution by Region':
            sales_over_time_region = filtered_df.groupby([filtered_df['Date'].dt.to_period('M'), 'Dealer_Region']).size().unstack().fillna(0).reset_index()
            sales_over_time_region['Date'] = sales_over_time_region['Date'].astype(str)

            fig9 = px.line(sales_over_time_region, 
                           x='Date', 
                           y=sales_over_time_region.columns[1:], 
                           title='Evolution of Sales Over Time by Region',
                           labels={'value': 'Number of Sales', 'Date': 'Month'},
                           color_discrete_sequence=px.colors.qualitative.Set1)

            st.plotly_chart(fig9)

        elif st.session_state['chart_type'] == 'Time Series by Region and Model':
            selected_region_time_series = st.selectbox('Select Region', regions)
            selected_model_time_series = st.selectbox('Select Vehicle Model', filtered_df['Model'].unique())

            def plot_sales(region, model):
                sales_time = filtered_df[(filtered_df['Dealer_Region'] == region) & (filtered_df['Model'] == model)].groupby(filtered_df['Date'].dt.to_period('M')).size()
                plt.figure(figsize=(14, 8))
                sales_time.plot(kind='line', marker='o', color='#FF7F0E', linewidth=2, markersize=6)
                plt.title(f'Monthly Sales - Region: {region}, Model: {model}', fontsize=16)
                plt.xlabel('Month', fontsize=14)
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

        elif st.session_state['chart_type'] == 'Product Mix Heatmap':
            mix_product_region = filtered_df.groupby(['Dealer_Region', 'Body Style']).size().unstack().fillna(0)
            plt.figure(figsize=(12, 8))
            sns.heatmap(mix_product_region, annot=True, cmap='coolwarm', fmt='g')

            # Ensure that legends and labels are visible
            plt.title('Product Mix by Region (Body Style)', fontsize=16)
            plt.xlabel('Body Style', fontsize=14)
            plt.ylabel('Reseller Region', fontsize=14)
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

    # Page: Vehicle Sales
    elif st.session_state['page'] == "Vehicle Sales":
        st.title('Dashboard Yamaha - Vehicle Sales')

        # Initialize the session state for charts if not yet defined
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Average Revenue by Car Type'

        # First row of buttons for the "Vehicle Sales" page
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Average Revenue by Car Type"):
                st.session_state['chart_type'] = "Average Revenue by Car Type"
        with col2:
            if st.button("Top 10 Companies by Revenue"):
                st.session_state['chart_type'] = "Top 10 Companies by Revenue"
        
        # Second row of buttons for the "Vehicle Sales" page
        col3 = st.columns(1)
        with col3[0]:
            if st.button("Transmission Distribution by Engine"):
                st.session_state['chart_type'] = "Transmission Distribution by Engine"


        # Display the chart based on the button choice
        if st.session_state['chart_type'] == 'Average Revenue by Car Type':
            avg_price_by_body = filtered_df.groupby('Body Style')['Price ($)'].mean().reset_index()
            fig2 = px.bar(avg_price_by_body, x='Body Style', y='Price ($)', title='Average Revenue by Car Type')
            st.plotly_chart(fig2)

        elif st.session_state['chart_type'] == 'Top 10 Companies by Revenue':
            top_companies = filtered_df.groupby('Company')['Price ($)'].sum().reset_index().sort_values(by='Price ($)', ascending=False).head(10)
            fig5 = px.bar(top_companies, x='Company', y='Price ($)', title='Top 10 Companies by Revenue')
            st.plotly_chart(fig5)

        elif st.session_state['chart_type'] == 'Transmission Distribution by Engine':
            transmission_distribution = filtered_df.groupby(['Engine', 'Transmission']).size().reset_index(name='Counts')
            fig6 = px.bar(transmission_distribution, x='Engine', y='Counts', color='Transmission', barmode='group', title='Transmission Distribution by Engine')
            st.plotly_chart(fig6)

    # Page: Customer Profile
    elif st.session_state['page'] == "Customer Profile":
        st.title('Dashboard Yamaha - Customer Profile')

        # Initialize the session state for charts if not yet defined
        if 'chart_type' not in st.session_state:
            st.session_state['chart_type'] = 'Gender Distribution by Region'

        # First row of buttons for the "Customer Profile" page
        col1 = st.columns(1)
        with col1[0]:
            if st.button("Gender Distribution by Region"):
                st.session_state['chart_type'] = "Gender Distribution by Region"
        
        # Second row of buttons for the "Customer Profile" page
        col2 = st.columns(1)
        with col2[0]:
            if st.button("Top 10 Models by Gender"):
                st.session_state['chart_type'] = "Top 10 Models by Gender"

        # Display the chart based on the button choice
        if st.session_state['chart_type'] == 'Gender Distribution by Region':
            gender_distribution = filtered_df.groupby(['Dealer_Region', 'Gender']).size().reset_index(name='Counts')
            fig3 = px.bar(gender_distribution, x='Dealer_Region', y='Counts', color='Gender', barmode='group', title='Gender Distribution by Region')
            st.plotly_chart(fig3)

        elif st.session_state['chart_type'] == 'Top 10 Models by Gender':
            top_10_male_models = filtered_df[filtered_df['Gender'] == 'Male']['Model'].value_counts().head(10)
            top_10_female_models = filtered_df[filtered_df['Gender'] == 'Female']['Model'].value_counts().head(10)

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

# NLP dataset processing
if df_nlp is not None and st.session_state['page'] == "NLP":
    st.title('Dashboard Yamaha - NLP Analysis')

    # Custom color palette
    colorscale = [
        [0.0, "rgb(0, 0, 139)"],   # Navy (equivalent to dark purple)
        [0.2, "rgb(75, 0, 130)"],  # Indigo
        [0.4, "rgb(138, 43, 226)"], # BlueViolet
        [0.6, "rgb(255, 0, 255)"],  # Magenta
        [0.8, "rgb(255, 165, 0)"],  # Orange
        [1.0, "rgb(255, 255, 0)"],  # Yellow
    ]

    # First row of buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Sentiment Analysis"):
            st.session_state['chart_type'] = "Sentiment Analysis"
    with col2:
        if st.button("Word Cloud"):
            st.session_state['chart_type'] = "Word Cloud"
    with col3:
        if st.button("Top word frequency"):
            st.session_state['chart_type'] = "Top word frequency"
    
    # Second row of buttons
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("Bigrams"):
            st.session_state['chart_type'] = "Bigrams"
    with col5:
        if st.button("Trigrams"):
            st.session_state['chart_type'] = "Trigrams"
    with col6:
        if st.button("Top Words Sentiment Analysis"):
            st.session_state['chart_type'] = "Top Words Sentiment Analysis"

        
    if 'chart_type' in st.session_state and st.session_state['chart_type'] == "Top Words Sentiment Analysis":
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Ensure all reviews are strings and handle NaNs
        df_nlp['review'] = df_nlp['review'].astype(str).fillna('')
        
        # Apply sentiment analysis
        sia = SentimentIntensityAnalyzer()
        df_nlp['sentiment_vader'] = df_nlp['review'].apply(lambda x: sia.polarity_scores(x)['compound'])
        df_nlp['sentiment_category'] = df_nlp['sentiment_vader'].apply(lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral'))
        
        # Filter for the categories of interest: Negative and Neutral (renamed as Positive)
        df_negative = df_nlp[df_nlp['sentiment_category'] == 'Negative']
        df_neutral = df_nlp[df_nlp['sentiment_category'] == 'Neutral']  # Will be treated as Positive
        
        # Word count for each category
        negative_words = df_negative['review'].str.split(expand=True).unstack().value_counts()
        neutral_words = df_neutral['review'].str.split(expand=True).unstack().value_counts()
        
        # Create bar chart for negative words
        data_negative = [go.Bar(
                    x = negative_words.index.values[:30],
                    y = negative_words.values[:30],
                    marker=dict(colorscale='Jet',
                                color=negative_words.values[:30]),
                    text=''
            )]
        
        layout_negative = go.Layout(
            title='Top 30 Word Frequencies in Negative Reviews',
            xaxis=dict(tickangle=-45)  # Adjusts the rotation of the x-axis labels
        )
            
        fig_negative = go.Figure(data=data_negative, layout=layout_negative)
        st.plotly_chart(fig_negative)
        
        # Create bar chart for neutral words (renamed as positive)
        data_neutral = [go.Bar(
                    x = neutral_words.index.values[:30],
                    y = neutral_words.values[:30],
                    marker=dict(colorscale='Jet',
                                color=neutral_words.values[:30]),
                    text=''
            )]
        
        layout_neutral = go.Layout(
            title='Top 30 Word Frequencies in Positive Reviews',
            xaxis=dict(tickangle=-45)  # Adjusts the rotation of the x-axis labels
        )
        
        fig_neutral = go.Figure(data=data_neutral, layout=layout_neutral)
        st.plotly_chart(fig_neutral)




    if 'chart_type' in st.session_state and st.session_state['chart_type'] == "Sentiment Analysis":
        # Extract sentiment components correctly
        df_sentiment_scores = pd.json_normalize(df_nlp['sentiment score'].apply(eval))
        df_nlp['sentiment_pos'] = df_sentiment_scores['pos']
        df_nlp['sentiment_neg'] = df_sentiment_scores['neg']
        df_nlp['sentiment_neu'] = df_sentiment_scores['neu']

        # Calculate the average sentiment by brand
        brand_sentiment = df_nlp.groupby('brand_name').agg({
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'sentiment_neu': 'mean'
        }).reset_index()

        # Transform data to long format for easier plotting
        brand_sentiment_melted = brand_sentiment.melt(id_vars='brand_name', 
                                                      value_vars=['sentiment_pos', 'sentiment_neg', 'sentiment_neu'],
                                                      var_name='Sentiment', value_name='Average')

        # Mapping to more readable names
        brand_sentiment_melted['Sentiment'] = brand_sentiment_melted['Sentiment'].map({
            'sentiment_pos': 'Positive',
            'sentiment_neg': 'Negative',
            'sentiment_neu': 'Neutral'
        })

        # Create interactive chart using Plotly
        fig = px.bar(brand_sentiment_melted, 
                     x='brand_name', 
                     y='Average', 
                     color='Sentiment', 
                     barmode='group',
                     labels={'brand_name': 'Brand', 'Average': 'Average Sentiment'},
                     title='Sentiment Comparison by Brand',
                     color_continuous_scale=colorscale)  # Applying the color palette

        # Display the interactive chart
        st.plotly_chart(fig)
    
    elif 'chart_type' in st.session_state and st.session_state['chart_type'] == "Word Cloud":
        # Check and clean missing data in the 'review' column
        df_nlp['review'] = df_nlp['review'].fillna("")

        # Generate a word cloud for the reviews in the 'review' column of the dataset detailed_car_5_brands.csv
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df_nlp['review']))

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Most Frequent Words')
        plt.axis('off')
        st.pyplot(plt)
    
    elif 'chart_type' in st.session_state and st.session_state['chart_type'] == "Top word frequency":
        # Word count in the NLP dataset
        word_counts = df_nlp['review'].str.split(expand=True).unstack().value_counts().reset_index()
        word_counts.columns = ['Word', 'Count']

        # Select the top 10 most frequent words
        top_20_words = word_counts.head(20)

        # Create interactive bar chart
        fig = px.bar(top_20_words, x='Word', y='Count', title='Top 20 Words', labels={'Word': 'Word', 'Count': 'Count'},
                     color='Count',
                     color_continuous_scale=colorscale)  # Applying the color palette
        st.plotly_chart(fig)
    
    elif 'chart_type' in st.session_state and st.session_state['chart_type'] == "Bigrams":
        # Function to generate bigrams
        def generate_bigrams(text):
            words = text.split()
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            return bigrams

        # Ensure the 'review' column does not contain null values and all values are strings
        df_nlp['review'] = df_nlp['review'].fillna("").astype(str)

        # Generate bigrams for all reviews
        df_nlp['bigrams'] = df_nlp['review'].apply(generate_bigrams)
        
        # Unify bigrams into a single list for counting
        all_bigrams = [bigram for sublist in df_nlp['bigrams'] for bigram in sublist]
        
        # Count bigram frequencies
        bigram_counts = pd.Series(all_bigrams).value_counts().reset_index()
        bigram_counts.columns = ['Bigram', 'Count']

        # Select the top 20 most frequent bigrams
        top_20_bigrams = bigram_counts.head(20)

        # Create interactive bigram chart
        bigram_strs = top_20_bigrams['Bigram'].apply(lambda x: ' '.join(x))
        fig = px.bar(top_20_bigrams, x=bigram_strs, y='Count', title='Top 20 Bigrams', labels={'x': 'Bigram', 'Count': 'Count'},
                     color='Count',
                     color_continuous_scale=colorscale)  # Applying the color palette
        st.plotly_chart(fig)

    elif 'chart_type' in st.session_state and st.session_state['chart_type'] == "Trigrams":
        # Function to generate trigrams
        def generate_trigrams(text):
            words = text.split()
            trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
            return trigrams
    
        # Ensure the 'review' column does not contain null values and all values are strings
        df_nlp['review'] = df_nlp['review'].fillna("").astype(str)
    
        # Generate trigrams for all reviews
        df_nlp['trigrams'] = df_nlp['review'].apply(generate_trigrams)
        
        # Unify trigrams into a single list for counting
        all_trigrams = [trigram for sublist in df_nlp['trigrams'] for trigram in sublist]
        
        # Count trigram frequencies
        trigram_counts = pd.Series(all_trigrams).value_counts().reset_index()
        trigram_counts.columns = ['Trigram', 'Count']
    
        # Select the top 20 most frequent trigrams
        top_20_trigrams = trigram_counts.head(20)
    
        # Create interactive trigram chart with the custom color palette
        trigram_strs = top_20_trigrams['Trigram'].apply(lambda x: ' '.join(x))
        fig = px.bar(top_20_trigrams, 
                     x=trigram_strs, 
                     y='Count', 
                     title='Top 20 Trigrams', 
                     labels={'x': 'Trigram', 'Count': 'Count'},
                     color='Count',
                     color_continuous_scale=colorscale)  # Applying the color palette
        st.plotly_chart(fig)
