# utils.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def scrape_data(url):
    """
    Scrapes real estate data from a given URL using BeautifulSoup and Requests libraries.
    Returns a BeautifulSoup object containing the scraped data.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup


def extract_features(soup):
    """
    Extracts relevant features from the scraped data and returns a Pandas DataFrame.
    """
    property_features = []
    for prop in soup.find_all('div', class_='property'):
        property_features.append({
            'price': prop.find('span', class_='price').text.strip(),
            'area': prop.find('span', class_='area').text.strip(),
        })

    df = pd.DataFrame(property_features)
    return df


def process_data(df):
    """
    Preprocesses and cleans the data, and performs any necessary feature engineering.
    Returns a processed DataFrame.
    """
    df['price'] = df['price'].str.replace(',', '').astype(float)
    df['area'] = df['area'].str.extract('(\d+)').astype(float)

    return df


def train_model(df):
    """
    Trains a machine learning regression model using the processed data.
    Returns the trained model.
    """
    X = df.drop('price', axis=1)
    y = df['price']

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()
    model.fit(X, y)
    return model


def get_prediction(model, input_features):
    """
    Uses the trained model to generate price predictions for a given set of input features.
    Returns the predicted prices.
    """
    input_df = pd.DataFrame(input_features)
    processed_input = process_data(input_df)

    predictions = model.predict(processed_input)

    return predictions


def main():
    url = 'https://www.example.com/real-estate'
    soup = scrape_data(url)

    df = extract_features(soup)

    processed_df = process_data(df)

    model = train_model(processed_df)

    input_features = {
        'area': 1500,
    }

    predictions = get_prediction(model, input_features)

    print(predictions)


if __name__ == "__main__":
    main()
