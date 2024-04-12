# Disaster-Tweets-Streamlit

This project contains a Streamlit app which includes visualizations and predictions on the Disaster Tweets Dataset.

## Demo
This the working demo of the project:

https://user-images.githubusercontent.com/96180147/e05f765e-d709-4a1b-86cc-2ac6d16e3066

## Description

The Streamlit app provides various functionalities:

1. Data Exploration:
    - Head
    - Info
    - Describe
    - Shape
    - Columns
    - Column Value Counts
    - Correlation Matrix
    - Covariance Matrix
    - Pivot Table

2. Train Model:
    - Logistic Regression
    - Multinomial Naive Bayes
    - Support Vector Machine
    - Random Forest
    - Gradient Boosting

3. Model Results:
    - Displays the results of various classifiers with metrics such as Accuracy, Precision, Recall, and F1 Score.

4. Word Cloud:
    - Generates a word cloud of the tweets.

5. Prediction:
    - Allows users to input a tweet and predicts whether it's about a disaster or not using selected classifiers.

6. Plots:
    - Tweet Length Distribution
    - Word Cloud of Tweets
    - Target Distribution
    - Top Ten Countries with the Most Tweets
    - Keyword Distribution
    - Disaster Tweets Distribution by Location

## Usage

To run the Streamlit app, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/Disaster-Tweets-Streamlit.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Credits

This project was developed by [Your Name].
