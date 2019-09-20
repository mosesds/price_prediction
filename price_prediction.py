#! python3
# "Predicts" stock price using csv datasheet and support vector regression

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

prices = []
dates = []

tesla_file = 'TSLA.csv'

def parse_csv(filename):
    '''
    read in csv data, s
    '''
    with open(filename, 'r') as f_obj:
        csv_file = csv.reader(f_obj)
        for n, row in enumerate(csv_file):
            if n in range(224,244):
                dates.append(int(row[0].split('-')[2]))
                prices.append(float(row[1]))

def plot_regression(dates, prices, x):
    '''
    reshape date list to an 1 dimesional array. create linear, polynomial, 
    and rbf models, train them on the data, then plot them.
    '''
    dates = np.reshape(dates,(len(dates), 1))

    svr_l = SVR(kernel='linear', C=1000)
    svr_p = SVR(kernel='poly', C=1000, degree=2)
    svr_r = SVR(kernel='rbf', C=1000, gamma='auto')
    svr_l.fit(dates, prices)
    svr_p.fit(dates, prices)
    svr_r.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_l.predict(dates), 
             color='green', label='Linear Model')
    plt.plot(dates, svr_p.predict(dates), color='blue', 
             label='Polynomial Model')
    plt.plot(dates, svr_r.predict(dates), color='red', label='RBF Model')
    plt.xlabel('Day (Aug 2019)')
    plt.ylabel('Open Price (Dollars)')
    plt.title('Support Vector Regression for Tesla Stock')
    plt.legend()
    plt.show()
    
    results = [svr_l.predict([[x]])[0], svr_p.predict([[x]])[0], 
               svr_r.predict([[x]])[0]]
    return results

def write_results(results):
    '''
    writes results of prediction to a text file
    '''
    with open('tsla_prediction.txt', 'w') as f_obj:
        if results[0]:
            f_obj.write(f'Linear = {results[0]}\n')
        if results[1]:
            f_obj.write(f'Polynomial = {results[1]}\n')
        if results[2]:
            f_obj.write(f'RBF = {results[2]}\n')

        actual = 224.080002
        f_obj.write(f'The actual value was {actual}.\n')

        closest_func = lambda result,results:min(results,
        										key=lambda x:abs(x-result))
        closest = closest_func(actual, results)
        f_obj.write(f'The closest value was {closest}.\n')

        error = ((abs(closest - actual))/actual) * 100
        f_obj.write(f'The percent error was {round(error, 2)}%.')

parse_csv(tesla_file)

# 33 in this case would be sept 3rd, the next available data set
predicted_price = plot_regression(dates, prices, 33)

write_results(predicted_price)