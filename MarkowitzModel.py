import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Stocks to handle - can change these tickers (or add some or take away) for diff. ports
stocks = ['NVDX', 'JEPQ', 'JEPI', 'MSFT', 'SONY']
# Avg No. of Trading Days
NUM_TRADING_DAYS = 252
# Will Generate Random Portfolios w/ Random Weights
NUM_OF_PORTFOLIOS = 10000

# Define start and end dates for data
start_date = '2010-01-01'
end_date = '2024-01-01'

# Fetch Historical Data
def download_data():
    #Stocks (Key) : Stock Data(2010-2024) (Values)
    stock_data = {}
    for stock in stocks:
        '''Closing Prices'''
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start = start_date, end = end_date)['Close']
    
    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize = (10,5))
    plt.show()

def calculate_return(data):
    #Normalizes data, measures all variables in a comparable metric
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_statistics(returns):
    #after annual returns instead of daily
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weights):
    #After the annual return
    portfolio_return = np.sum(returns.mean()*weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov 
                                                            * NUM_TRADING_DAYS, weights)))
    print(f"Expected Portfolio Mean (Return): {portfolio_return}")
    print(f"Expected Portfolio Volatility (St Dev): {portfolio_volatility}")

def show_portfolios(returns, volatilities):
    plt.figure(figsize = (10,6))
    plt.scatter(volatilities, returns, c = returns/volatilities, marker = 'o')
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Returns")
    plt.colorbar(label = "Sharpe Ratio")
    plt.show()

def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_OF_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))
        
    return np.array(portfolio_weights),np.array(portfolio_means), np.array(portfolio_risks)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() 
                                                            * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, 
                     portfolio_return / portfolio_volatility])

#scipy optimize can grab the minimum sharpe ratio
# the maximum of f(x) is the min of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    #Sum of weights is 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun = min_function_sharpe, x0 = weights[0], args = returns, 
                          method = 'SLSQP', bounds = bounds, constraints=constraints)
    
def print_optimal_portfolio(optimum, returns):
    print("Optimal Portfolio: ", optimum['x'].round(3))
    print("Expected Return, Volatility, and Sharpe Ratio: ", 
          statistics(optimum['x'].round(3), returns))

def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize = (10,6))
    plt.scatter(portfolio_vols, portfolio_rets, c = portfolio_rets / portfolio_vols, marker = 'o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Returns')
    plt.colorbar(label = 'Sharpe Ratio')
    opt_return = np.sum(rets.mean() * opt['x']) * NUM_TRADING_DAYS
    opt_volatility = np.sqrt(np.dot(opt['x'].T, np.dot(rets.cov() * NUM_TRADING_DAYS, opt['x'])))
    
    # Plot the optimal portfolio
    plt.plot(opt_volatility, opt_return, 'g*', markersize=20)
    plt.show()

if __name__ == '__main__':
    portfolio_data = download_data()
    show_data(portfolio_data)
    log_daily_returns = calculate_return(portfolio_data)
    #show_statistics(log_daily_returns)
    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)