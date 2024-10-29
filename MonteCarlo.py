import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

NUM_OF_SIMULATIONS = 500

# Download company price data
def stock_monte_carlo(ticker, N=252):
    stock_data = yf.download(ticker, period= '1y')
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    mu = stock_data['Daily Return'].mean()
    sigma = stock_data['Daily Return'].std()
    SO = stock_data['Adj Close'].iloc[-1]

    result = []

# Create 500 possible outcomes 
    for i in range(NUM_OF_SIMULATIONS):
        prices = [SO]
        for point in range(N):
            stock_price = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) 
                                              + sigma * np.random.normal())
            prices.append(stock_price)
        result.append(prices)

    simulation_data = pd.DataFrame(result).T
    simulation_data['mean'] = simulation_data.mean(axis=1)
    
    # Plot the mean values
    plt.plot(simulation_data)
    plt.plot(simulation_data['mean'], color = 'blue', ls = '--')
    plt.title(f'Stock Price Simulation for: ${ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()

# Calculate and print out predicted price 1 year out
    final_price = simulation_data['mean'].tail(1).values[0]
    print(f'Prediction (1yr from now) for future stock price: ${final_price:.2f}')

if __name__ == '__main__':
    ticker_symbol = input("Enter a Stock's Ticker Symbol: ")
    stock_monte_carlo(ticker_symbol)