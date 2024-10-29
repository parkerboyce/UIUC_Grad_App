import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

RISK_FREE_RATE = 0.05
MONTHS_IN_YEAR = 12

class CAPM():
    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
    
    def download_data(self):
        data = {}

        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker['Adj Close']

        return pd.DataFrame(data)
    
    def initialize(self):
        stock_data = self.download_data()
        stock_data = stock_data.resample('ME').last()
        self.data = pd.DataFrame({'stock_adjclose': stock_data[self.stocks[0]], 
                                  'market_adjclose': stock_data[self.stocks[1]]})
        self.data[['stock_adjclose', 'market_adjclose']] = np.log(self.data[['stock_adjclose', 'market_adjclose']] 
                                                                  / self.data[['stock_adjclose', 'market_adjclose']].shift(1))
        self.data = self.data[1:]
        
    def calculate_beta(self):
        covariance_matrix = np.cov(self.data['stock_adjclose'], self.data['market_adjclose'])
        beta = covariance_matrix[0, 1] / covariance_matrix[1,1]
        print(f'Beta from the formula: {beta}')

    def regression(self):
        beta, alpha = np.polyfit(self.data['market_adjclose'], self.data['stock_adjclose'], deg = 1)
        print(f'Beta from the regression: {beta}')
        expected_return = RISK_FREE_RATE + beta * (self.data['market_adjclose'].mean() 
                                                   * MONTHS_IN_YEAR - RISK_FREE_RATE)
        print(f'The Expected Return: {expected_return}')
        self.plot_regression(alpha, beta)
    
    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20,10))
        axis.scatter(self.data["market_adjclose"], self.data['stock_adjclose'], label = 'Data Points')
        axis.plot(self.data['market_adjclose'], beta * self.data['market_adjclose'] + alpha, color = 'red', label = 'CAPM Line')
        plt.title('Capital Asset Pricing Model, finding alpha and beta')
        plt.xlabel('Market Return $R_m$', fontsize = 18)
        plt.ylabel('Stock Return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m \alpha$', fontsize = 18)
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    ''' 
    Use '^GSPC to measure the first stock against the S&P 500
    '''
    capm = CAPM(['NVDA', '^GSPC'], '2010-01-01', '2024-01-01')
    capm.initialize()
    capm.calculate_beta()
    capm.regression()