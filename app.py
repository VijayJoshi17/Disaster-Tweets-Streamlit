import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud
from PIL import Image

# Load the dataset
TRAIN_DATA_URL = "./train.csv"

@st.cache_data(persist=True)
def load_train_data():
    train_df = pd.read_csv(TRAIN_DATA_URL)
    return train_df

df = load_train_data()

# Fill NaN values in 'location' column with most occuring values
specific_locations = ['USA', 'New York', 'London', 'Canada', 'Nigeria', 'UK', 'Los Angeles', 'CA', 'India', 'Washington, DC', 'Mumbai', 'Kenya', 'Australia', 'California']
df['location'].fillna(pd.Series(np.random.choice(specific_locations, size=len(df.index))), inplace=True)

# Set title
st.title('Disaster Tweet Analysis')
st.write('This is a project about Disaster Tweets and classfying them into disaster related or not.')
st.write("Download the dataset from [here](https://www.kaggle.com/competitions/nlp-getting-started/data?select=train.csv)")

image_path = ".images\yosh-ginsu-qexZLgMcbPc-unsplash.jpg"
try:
    with Image.open(image_path) as img:
        st.image(img, use_column_width=True)
except FileNotFoundError:
    st.error("Image not found. Please provide the correct path.")


# Data exploration

st.sidebar.title('Options')
# Sidebar for data exploration options
st.sidebar.title('Data Exploration')
explore_head = st.sidebar.checkbox('Head')
explore_info = st.sidebar.checkbox('Info')
explore_describe = st.sidebar.checkbox('Describe')
explore_shape = st.sidebar.checkbox('Shape')
explore_columns = st.sidebar.checkbox('Columns')
explore_value_counts = st.sidebar.checkbox('Column Value Counts')
explore_correlation_matrix = st.sidebar.checkbox('Correlation Matrix')
explore_covariance_matrix = st.sidebar.checkbox('Covariance Matrix')
explore_pivot_table = st.sidebar.checkbox('Pivot Table')

# Display selected data exploration options in main interface
if explore_head:
    st.subheader('Head:')
    st.write(df.head())

if explore_info:
    st.subheader('Info:')
    st.write(df.dtypes)

if explore_describe:
    st.subheader('Description:')
    st.write(df.describe())

if explore_shape:
    st.subheader('Shape:')
    st.write(df.shape)

if explore_columns:
    st.subheader('Columns:')
    st.write(df.columns)

if explore_value_counts:
    selected_column = st.sidebar.selectbox('Select a column:', df.columns)
    st.subheader('Column Value Counts:')
    value_counts_df = df[selected_column].value_counts().reset_index()
    value_counts_df.columns = ['Value', 'Count']
    st.write(value_counts_df.style.format({'Value': '{:}', 'Count': '{:,.0f}'}).set_table_styles([{'selector': 'th','props': [('background-color', 'skyblue')]}]), max_width=12)

if explore_correlation_matrix:
    st.subheader('Correlation Matrix:')
    numeric_columns = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numeric_columns].corr()
    st.write(correlation_matrix)

if explore_covariance_matrix:
    st.subheader('Covariance Matrix:')
    numeric_columns = df.select_dtypes(include=np.number).columns
    covariance_matrix = df[numeric_columns].cov()
    st.write(covariance_matrix)

if explore_pivot_table:
    st.subheader('Pivot Table:')
    index_column = st.sidebar.selectbox('Select an index column:', df.columns)
    value_column = st.sidebar.selectbox('Select a value column:', df.columns)
    columns = st.sidebar.multiselect('Select columns for pivot table:', df.columns)
    if columns:
        pivot_table = pd.pivot_table(df, values=value_column, index=index_column, columns=columns, aggfunc=np.sum)
        st.write(pivot_table)




# Train Model 
st.sidebar.title('Train Model')

# Train Model Options
LR = st.sidebar.checkbox('Logistic Regression')
MNB = st.sidebar.checkbox('Multinomial Naive Bayes')
svc_checkbox = st.sidebar.checkbox('Support Vector Machine')
RF = st.sidebar.checkbox('Random Forest')
GB = st.sidebar.checkbox('Gradient Boosting')

def train_and_evaluate_model(model_name, model, X_train_vec, y_train):
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_train_vec)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    return {'Classifier': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

def preprocess_data(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(df['text'])
    y_train = df['target']
    return X_train_vec, y_train

results = []

if LR:
    model_LR = LogisticRegression()
    X_train_vec, y_train = preprocess_data(df)
    results.append(train_and_evaluate_model('Logistic Regression', model_LR, X_train_vec, y_train))

if MNB:
    model_MNB = MultinomialNB()
    X_train_vec, y_train = preprocess_data(df)
    results.append(
        train_and_evaluate_model('Multinomial Naive Bayes', model_MNB, X_train_vec, y_train))

if svc_checkbox:
    model_SVC = SVC()
    X_train_vec, y_train = preprocess_data(df)
    results.append(train_and_evaluate_model('Support Vector Machine', model_SVC, X_train_vec, y_train))

if RF:
    model_RF = RandomForestClassifier()
    X_train_vec, y_train = preprocess_data(df)
    results.append(train_and_evaluate_model('Random Forest', model_RF, X_train_vec, y_train))

if GB:
    model_GB = GradientBoostingClassifier()
    X_train_vec, y_train = preprocess_data(df)
    results.append(
        train_and_evaluate_model('Gradient Boosting', model_GB, X_train_vec, y_train))

if results:
    st.title('Model Results')
    results_df = pd.DataFrame(results)
    st.write(results_df)

    all_metrics = pd.concat([results_df.drop('Classifier', axis=1)], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = all_metrics.plot(kind='bar', ax=ax, legend=True)

    ax.set_xticks(range(len(results_df)))
    ax.set_xticklabels(results_df['Classifier'], rotation=0, ha='center')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(all_metrics.columns))

    ax.set_ylabel('Score')
    ax.set_xlabel('Classifier')
    ax.set_title('Comparison of Model Metrics')

    ax.set_ylim(0, 1)

    ax.grid(axis='y')

    plt.tight_layout()
    st.pyplot(fig)

# Word Cloud
st.sidebar.title('Word Cloud')
if st.sidebar.button('Generate Word Cloud'):
    all_tweets = ' '.join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(all_tweets)
    st.title('Word Cloud of Tweets')
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)



classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}


# Define the TF-IDF vectorizer object
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec, y_train = preprocess_data(df)

# Prediction
st.sidebar.title('Prediction')


selected_model = st.sidebar.radio('Select a model for prediction:', list(classifiers.keys()))


predic_tweet = st.sidebar.checkbox('click for prediction ')

if predic_tweet:
    tweet = st.text_input('Enter a tweet:')
    X_train_vec = vectorizer.fit_transform(df['text'])

    if st.button('Predict'):
        if tweet:
            X_test_vec = vectorizer.transform([tweet])
            model = classifiers[selected_model]
            model.fit(X_train_vec, y_train)
            prediction = model.predict(X_test_vec)
            if prediction[0] == 1:
                st.error('This tweet is about a disaster!')
            else:
                st.success('This tweet is not about a disaster.')
        else:
            st.warning("Please enter a tweet.")






# Plots

st.sidebar.title('Plots')

plot_tweet_length_histogram = st.sidebar.checkbox('Tweet Length Distribution')
plot_word_cloud = st.sidebar.checkbox('Word Cloud of Tweets')
plot_target_distribution = st.sidebar.checkbox('Target Distribution')
plot_10_countries = st.sidebar.checkbox('Plot Top Ten Countries')
plot_keyword_distribution = st.sidebar.checkbox('Keyword Distribution')
plot_country_bar = st.sidebar.checkbox('Disaster Tweets Distribution by Location')


# Plot histogram of tweet lengths
if plot_tweet_length_histogram:
    st.write('### Tweet Length Distribution')
    fig = px.histogram(df, x=df['text'].apply(len), nbins=30)
    fig.update_layout(xaxis_title='Tweet Length', yaxis_title='Frequency')
    st.plotly_chart(fig)

# Plot word cloud
if plot_word_cloud:
    st.write('### Word Cloud of Tweets')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text']))
    fig = px.imshow(wordcloud)
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig)

# Plot bar chart of target distribution
if plot_target_distribution:
    st.write('### Target Distribution')
    fig = px.histogram(df, x='target', labels={'target': 'Target'}, color='target')
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Not Disaster', 'Disaster']))
    st.plotly_chart(fig)

# Plot bar chart of keyword distribution
if plot_keyword_distribution:
    st.write('### Keyword Distribution')
    top_keywords = df['keyword'].value_counts().head(10)
    fig = px.bar(top_keywords, y=top_keywords.values, x=top_keywords.index)
    fig.update_layout(xaxis_title='Keyword', yaxis_title='Count')
    st.plotly_chart(fig)

if plot_10_countries:
    st.title('Top Ten Countries with the Most Tweets')
    df['country'] = df['location'].str.split(',').str[-1].str.strip()
    df['country'].replace('', np.nan, inplace=True)
    df.dropna(subset=['country'], inplace=True)
    country_counts = df['country'].value_counts().head(10)
    
    disaster_tweets = df[df['target'] == 1]
    non_disaster_tweets = df[df['target'] == 0]

    disaster_country_counts = disaster_tweets['country'].value_counts().head(10)
    non_disaster_country_counts = non_disaster_tweets['country'].value_counts().head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=country_counts.index, y=country_counts.values, name='Total Tweets'))
    fig.add_trace(go.Bar(x=disaster_country_counts.index, y=disaster_country_counts.values, name='Disaster Tweets'))
    fig.add_trace(go.Bar(x=non_disaster_country_counts.index, y=non_disaster_country_counts.values, name='Non-Disaster Tweets'))
    fig.update_layout(barmode='group', xaxis_title='Country', yaxis_title='Number of Tweets', title='Top Ten Countries with the Most Tweets')
    st.plotly_chart(fig)

if plot_country_bar:
    st.title('Disaster Tweets Distribution by Location')
    disaster_tweets = df[df['target'] == 1]
    country_distribution = disaster_tweets['location'].value_counts().head(10)
    fig = px.bar(country_distribution, x=country_distribution.index, y=country_distribution.values)
    fig.update_layout(xaxis_title='Country', yaxis_title='Number of Tweets')
    st.plotly_chart(fig)


