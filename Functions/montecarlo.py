import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


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
    for _ in range(num_simulations):
        portfolio_value = initial_portfolio_value
        for _ in range(years):
            annual_returns = np.random.multivariate_normal(expected_returns, covariance_matrix)
            portfolio_return = np.dot(weights, annual_returns)
            portfolio_value *= (1 + portfolio_return)
        final_values.append(portfolio_value)

    # Calculate statistics
    mode_value = histogram_mode(final_values)
    mean_value = np.mean(final_values)
    median_value = np.median(final_values)
    percentile_5 = np.percentile(final_values, 5)
    percentile_95 = np.percentile(final_values, 95)

    # Plot results
    def PlotReturnHistogram():
        print(f"Mode final portfolio value: £{mode_value:.2f}")
        print(f"Mean final portfolio value: £{mean_value:.2f}")
        print(f"Median final portfolio value: £{median_value:.2f}")
        print(f"5th percentile: £{percentile_5:.2f}")
        print(f"95th percentile: £{percentile_95:.2f}")

        plt.hist(final_values, bins=50, alpha=0.75)
        plt.title(f'Monte Carlo Simulation of Portfolio Value Given £{initial_portfolio_value} Initial Investment after {years} years')
        plt.xlabel('Portfolio Value')
        plt.ylabel('Frequency')
        plt.show()
    if plot:
        PlotReturnHistogram()
    return mode_value, mean_value, median_value, percentile_5, percentile_95

def PlotExpectedReturns(assets, final_medians):
    labels = list(final_medians.keys())
    values = list(final_medians.values())
    plt.plot(labels, values)
    plt.xlabel(f'Proportion of {assets[0]}')
    plt.ylabel('Expected return at given proportion')
    plt.show()

def main():
    # Simulation parameters
    initial_portfolio_value = 10000  
    years = 10  
    num_simulations = 1000

    # Expected returns & volatilities for each asset
    R = 0
    assets = ['Asset A', 'Asset B']
    expected_returns = np.array([0.15, 0.15])  
    volatilities = np.array([0.05, 0.05])
    print(assets, expected_returns, volatilities, R)

    # Test out different iterations of portfolio weights
    n = 10
    final_medians = {}
    for i in range(n+1):
        weight_a = i/n
        weight_b = 1 - weight_a
        weights = np.array([weight_a, weight_b]) 
        median = MonteCarlo(initial_portfolio_value=initial_portfolio_value,
                            years=years,
                            num_simulations=num_simulations,
                            expected_returns=expected_returns,
                            volatilities=volatilities,
                            weights=weights,
                            R=R,
                            plot=False)
        final_medians[weight_a] = median
        print(weights, median)
    PlotExpectedReturns(assets=assets, final_medians=final_medians)

# Call program for example data
if __name__ == "__main__":
    main()