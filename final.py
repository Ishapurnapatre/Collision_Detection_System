import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_data_from_json(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['tw_x_values1'], data['tw_y_values1'], data['tw_x_values2'], data['tw_y_values2']

def prepare_data(x, y):
    
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    return x, y

def polynomial_regression(tw_x_values1, tw_x_values2, tw_y_values1, tw_y_values2):

    # Polynomial Regression for case 1 (area < 46200)
    poly_features1 = PolynomialFeatures(degree=3)
    x_poly1 = poly_features1.fit_transform(tw_x_values1)
    model1 = LinearRegression()
    model1.fit(x_poly1, tw_y_values1)

    # Polynomial Regression for case 2 (area >= 46200)
    poly_features2 = PolynomialFeatures(degree=3)
    x_poly2 = poly_features2.fit_transform(tw_x_values2)
    model2 = LinearRegression()
    model2.fit(x_poly2, tw_y_values2)

    return poly_features1, model1, poly_features2, model2

def predict_regression_model(poly_features1, model1, poly_features2, model2, x):

    if(x>46200):
        x_poly1 = poly_features1.transform(x)
        return model1.predict(x_poly1)
    else:
        x_poly2 = poly_features2.transform(x)
        return model2.predict(x_poly2)
    
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
    x1, y1, x2, y2 = load_data_from_json('datasplit.json')

    # Prepare the data
    x1, y1 = prepare_data(x1, y1)
    x2, y2 = prepare_data(x2, y2)

    # Fit polynomial regression models
    model1, poly_features1 = polynomial_regression(x1, y1, x2, y2)

    # Predict using the models
    x_fit1 = np.linspace(min(x1), max(x1), 100).reshape(100, 1)  # Generate 100 points between min and max of x1
    y_fit1 = predict_regression_model(model1, poly_features1, x_fit1)

    # if(x>46200):
    #     x_fit1 = np.linspace(min(x1), max(x1), 100).reshape(100, 1)  # Generate 100 points between min and max of x1
    #     y_fit1 = predict_regression_model(model1, poly_features1, x_fit1)
    # else:
    #     x_fit1 = np.linspace(min(x1), max(x1), 100).reshape(100, 1)  # Generate 100 points between min and max of x1
    #     y_fit1 = predict_regression_model(model1, poly_features1, x_fit1)

    # # Plot the results
    # plot_results(x1, y1, x_fit1, y_fit1, x2, y2, x_fit2, y_fit2)

if __name__ == "__main__":
    main()