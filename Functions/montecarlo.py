import numpy as np
import matplotlib.pyplot as plt

def histogram_mode(values):
    num_bins = int(np.ceil(np.log2(len(values)) + 1))
    hist, bin_edges = np.histogram(values, bins=num_bins)
    max_freq_index = np.argmax(hist)
    mode = (bin_edges[max_freq_index] + bin_edges[max_freq_index + 1]) / 2
    return mode

def MonteCarlo(initial_portfolio_value, years, num_simulations, expected_returns, volatilities, weights, R, plot):
    # Calculate the covariance matrix
    correlation_matrix = np.array([[1.0, R], [R, 1.0]])
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    # Simulate annual returns for each asset using compound interest multiplication
    final_values = []
    all_portfolio_values = np.zeros((num_simulations, years + 1))
    all_portfolio_values[:, 0] = initial_portfolio_value

    for sim in range(num_simulations):
        portfolio_value = initial_portfolio_value
        for year in range(years):
            annual_returns = np.random.multivariate_normal(expected_returns, covariance_matrix)
            portfolio_return = np.dot(weights, annual_returns)
            portfolio_value *= (1 + portfolio_return)
            all_portfolio_values[sim, year + 1] = portfolio_value
        final_values.append(portfolio_value)

    # Calculate statistics
    mode_value = histogram_mode(final_values)
    mean_value = np.mean(final_values)
    median_value = np.median(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)

    # Plot distribution of results
    def PlotReturnHistogram():
        print(f"Mode final portfolio value: £{mode_value:.2f}")
        print(f"Mean final portfolio value: £{mean_value:.2f}")
        print(f"Median final portfolio value: £{median_value:.2f}")
        print(f"5th percentile: £{percentile_5:.2f}")
        print(f"95th percentile: £{percentile_95:.2f}")

        plt.hist(final_values, bins=50, alpha=0.75)
        plt.title(f'Monte-Carlo distribution of portfolio value given £{initial_portfolio_value} initial investment after {years} years')
        plt.xlabel('Log(Portfolio Value)')
        plt.xscale('log')
        plt.ylabel('Frequency')
        plt.show()
        return final_values

    # Plot all simulations (this can be very dense for many simulations)
    def PlotFunnel():
        plt.plot(range(years + 1), all_portfolio_values.T, color='lightgray', alpha=0.1)

        # Plot percentiles
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for p, c in zip(percentiles, colors):
            plt.plot(range(years + 1), np.percentile(all_portfolio_values, p, axis=0), color=c, linewidth=2)

        plt.legend(['5th', '25th', '50th (Median)', '75th', '95th'], loc='upper left')
        plt.title(f'Time-projection given £{initial_portfolio_value} initial investment after {years} years, with FTSE:BTC split = {weights}')
        plt.xlabel('Years')
        plt.ylabel('Portfolio Value')
        plt.yscale('log')
        plt.show()
        return all_portfolio_values

    # Plot the funnel chart
    if plot:
        PlotFunnel()
        PlotReturnHistogram()
    return mode_value, mean_value, median_value, percentile_5, percentile_95

def PlotExpectedReturns(assets, metrics):
    labels = list(metrics.keys())
    values = list(metrics.values())
    plt.plot(labels, values)
    plt.xlabel(f'Proportion of {assets[0]}')
    plt.ylabel('Expected return at given proportion')

# Example usage:
if __name__ == "__main__":
    initial_value = 10000
    num_simulations = 1000 # Assets = FTSE, BTC 
    years = 10

    p = 0
    expected_returns = np.array([0.0418, 0.688])  # Example expected returns for two assets
    volatilities = np.array([0.149, 0.680])  # Example volatilities for two assets
    weights = np.array([p, 1-p])  # Example portfolio weights
    correlation = 0.585  # Example correlation between assets

    results = MonteCarlo(initial_value, years, num_simulations, expected_returns, volatilities, weights, correlation, plot=True)