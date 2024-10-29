import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
import mplcursors

# Calculate the price of call options using Black-Scholes
''' 
S = Stock Price 
K = Strike Price
T = Time to Maturity 
r = Risk-Free Rate
sigma = Volatility of Underlying
'''
def calculate_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

 # Calculate the price of put options using Black-Scholes 
def calculate_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

# Set up the visual for the option P & L
def plot_pnl(S, K, option_price, option_type):
    stock_price_range = np.linspace(0, 2 * S, 100)
    if option_type == 'Call':
        pnl = np.maximum(stock_price_range - K, 0) - option_price
    else:
        pnl = np.maximum(K - stock_price_range, 0) - option_price

    fig, ax = plt.subplots(figsize=(12,6))
    line, = ax.plot(stock_price_range, pnl, label='Profit/Loss', color='blue')
    ax.axhline(0, color='red', linestyle='--')
    ax.axvline(K, color='green', linestyle='--', label='Strike Price')
    ax.set_title(f'{option_type} Option Profit/Loss')
    ax.set_xlabel('Stock Price at Expiry')
    ax.set_ylabel('Profit/Loss')
    ax.legend()
    ax.grid()

    cursor = mplcursors.cursor(line, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f})"))

    plt.show()

# Set up inputs
def main():
    S = float(input("Enter Stock Price (S): "))
    K = float(input("Enter Strike Price (K): "))
    T = float(input("Enter Time to Maturity (T in years): "))
    r = float(input("Enter Risk-Free Rate (r): "))
    sigma = float(input("Enter Volatility (sigma): "))
    option_type = input("Enter Option Type ('Call' or 'Put'): ").capitalize()

    if option_type == 'Call':
        option_price = calculate_call(S, K, T, r, sigma)
    else:
        option_price = calculate_put(S, K, T, r, sigma)

    print(f"The price of the {option_type} option is: {option_price}")
    plot_pnl(S, K, option_price, option_type)

# Call the function
if __name__ == '__main__':
    '''
    In the graph, along the blue line which represents the P & L of the contract
    the x coordinate represents the possible price of the underlying 
    and the y coordinate the profit/loss
    '''
    main()