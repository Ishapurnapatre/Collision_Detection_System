import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def load_data1_from_json(file_path):
    """
    Loads x and y values from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    tuple: x values and y values loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['tw_x_values'], data['tw_y_values']

def prepare_data(x, y):
    """
    Prepares the data for polynomial regression.

    Parameters:
    x (list): The input feature values.
    y (list): The target values.

    Returns:
    numpy.ndarray: Reshaped input feature values.
    numpy.ndarray: Target values.
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    return x, y

def fit_polynomial_regression(x, y, degree):
    """
    Fits a polynomial regression model to the data.

    Parameters:
    x (numpy.ndarray): The input feature values.
    y (numpy.ndarray): The target values.
    degree (int): The degree of the polynomial.

    Returns:
    LinearRegression: The fitted polynomial regression model.
    PolynomialFeatures: The polynomial feature transformer.
    """
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    return model, poly_features

def predict(model, poly_features, x):
    """
    Uses the polynomial regression model to make predictions.

    Parameters:
    model (LinearRegression): The fitted polynomial regression model.
    poly_features (PolynomialFeatures): The polynomial feature transformer.
    x (numpy.ndarray): The input feature values for prediction.

    Returns:
    numpy.ndarray: The predicted values.
    """
    x_poly = poly_features.transform(x)
    return model.predict(x_poly)

def plot_results(x, y, x_fit, y_fit):
    """
    Plots the original data points and the polynomial regression curve.

    Parameters:
    x (numpy.ndarray): The input feature values.
    y (numpy.ndarray): The target values.
    x_fit (numpy.ndarray): The input feature values for the regression curve.
    y_fit (numpy.ndarray): The predicted values for the regression curve.
    """
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x_fit, y_fit, color='red', label='Polynomial regression curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    # Load data from JSON file
    x, y = load_data1_from_json('data.json')

    # Prepare the data
    x, y = prepare_data(x, y)

    # Fit polynomial regression model
    degree = 3  # Degree of the polynomial
    model, poly_features = fit_polynomial_regression(x, y, degree)

    # Predict using the model
    x_fit = np.linspace(min(x), max(x), 100).reshape(100, 1)  # Generate 100 points between min and max of x
    y_fit = predict(model, poly_features, x_fit)

    # Plot the results
    plot_results(x, y, x_fit, y_fit)

    # Predict for a specific x value
    x_new = np.array([[14545]])
    y_new = predict(model, poly_features, x_new)
    print(f'Predicted value for x = 14545: {y_new[0]}')

if __name__ == "__main__":
    main()
