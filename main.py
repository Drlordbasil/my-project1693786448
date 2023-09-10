# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify

# Step 1: Web scraping


def scrape_real_estate_prices(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    property_list = []

    # Extract data from the website
    for property_info in soup.find_all('div', class_='property-info'):
        property_dict = {}

        # Extract relevant property features
        property_dict['Bedrooms'] = property_info.find(
            'div', class_='bedrooms').text.strip()
        property_dict['Bathrooms'] = property_info.find(
            'div', class_='bathrooms').text.strip()
        property_dict['Area_Sqft'] = property_info.find(
            'div', class_='area').text.strip()

        # Extract property price
        price = property_info.find('div', class_='price').span.text.strip()
        property_dict['Price'] = int(price.replace('$', '').replace(',', ''))

        property_list.append(property_dict)

    return property_list

# Step 2: Data Processing


def preprocess_data(property_list):
    df = pd.DataFrame(property_list)

    # Convert categorical features to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['Bedrooms', 'Bathrooms'])

    # Split features and target variable
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Step 3: Machine Learning Model


def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae


# Step 4: Application Interface
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()

    # Preprocess input data
    input_df = pd.DataFrame(data, index=[0])
    input_df = pd.get_dummies(input_df)

    # Load the trained model
    model = RandomForestRegressor()
    model.load_model('model.pkl')

    # Make predictions
    predicted_price = model.predict(input_df)

    return jsonify({'predicted_price': predicted_price.tolist()})


if __name__ == '__main__':
    # Step 1: Scrape real estate prices
    property_list = scrape_real_estate_prices(
        'https://www.example.com/real_estate')

    # Step 2: Data processing
    X_train, X_test, y_train, y_test = preprocess_data(property_list)

    # Step 3: Train machine learning model
    model = train_model(X_train, y_train)

    # Step 3: Evaluate model
    mae = evaluate_model(model, X_test, y_test)
    print('Mean Absolute Error:', mae)

    # Step 4: Save the trained model
    model.save_model('model.pkl')

    # Step 5: Run flask application
    app.run(debug=True)
