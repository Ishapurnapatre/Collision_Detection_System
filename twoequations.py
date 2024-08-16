import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# def load_data1_from_json(file_path):
    
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data['tw_x_values1'], data['tw_y_values1']

# def load_data2_from_json(file_path):
    
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data['tw_x_values2'], data['tw_y_values2']

def load_data1_from_json(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['lmv_x_values1'], data['lmv_y_values1']

def load_data2_from_json(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['lmv_x_values2'], data['lmv_y_values2']

def prepare_data(x, y):
    
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    return x, y

def fit_polynomial_regression(x, y, degree):
    
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    return model, poly_features

def predict(model, poly_features, x):
    
    x_poly = poly_features.transform(x)
    return model.predict(x_poly)

def plot_results(x1, y1, x_fit1, y_fit1, x2, y2, x_fit2, y_fit2):
    
    plt.scatter(x1, y1, color='blue', label='Data points 1')
    plt.plot(x_fit1, y_fit1, color='red', label='Polynomial regression curve 1')
    plt.scatter(x2, y2, color='green', label='Data points 2')
    plt.plot(x_fit2, y_fit2, color='orange', label='Polynomial regression curve 2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    # Load data from JSON file
    x1, y1 = load_data1_from_json('datasplit.json')
    x2, y2 = load_data2_from_json('datasplit.json')

    # Prepare the data
    x1, y1 = prepare_data(x1, y1)
    x2, y2 = prepare_data(x2, y2)

    # Fit polynomial regression models
    degree = 3  # Degree of the polynomial
    model1, poly_features1 = fit_polynomial_regression(x1, y1, degree)
    model2, poly_features2 = fit_polynomial_regression(x2, y2, degree)

    # Predict using the models
    x_fit1 = np.linspace(min(x1), max(x1), 100).reshape(100, 1)  # Generate 100 points between min and max of x1
    y_fit1 = predict(model1, poly_features1, x_fit1)
    
    x_fit2 = np.linspace(min(x2), max(x2), 100).reshape(100, 1)  # Generate 100 points between min and max of x2
    y_fit2 = predict(model2, poly_features2, x_fit2)

    # Plot the results
    plot_results(x1, y1, x_fit1, y_fit1, x2, y2, x_fit2, y_fit2)

if __name__ == "__main__":
    main()
