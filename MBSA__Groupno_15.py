import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Step 1: Define Parameters
true_channel_gain = 0.7  # New value for true channel gain
true_noise_variance = 0.15  # New value for true noise variance
num_samples = 1500  # New number of samples

# Step 2: Generate Simulated Data
rayleigh_data = np.sqrt(true_channel_gain) * np.random.randn(num_samples)  # Rayleigh fading component
awgn_data = np.sqrt(true_noise_variance) * np.random.randn(num_samples)  # AWGN component
combined_data = rayleigh_data + awgn_data  # Combined data

# Step 3: Maximum Likelihood Estimation Functions
def ml_channel_gain_estimator(data):
    return np.sum(np.abs(data) ** 2) / len(data)

def ml_noise_variance_estimator(data, channel_gain_est):
    return np.sum(np.abs(data - np.sqrt(channel_gain_est)) ** 2) / len(data)

# Step 4: Fisher Information and CRB Computation
def compute_fisher_information(true_channel_gain, true_noise_variance):
    fisher_info_channel_gain = 2 / true_noise_variance
    fisher_info_noise_variance = 2 * true_channel_gain ** 2 / true_noise_variance ** 2
    return fisher_info_channel_gain, fisher_info_noise_variance

fisher_info_channel_gain, fisher_info_noise_variance = compute_fisher_information(true_channel_gain, true_noise_variance)
crb_channel_gain = 1 / fisher_info_channel_gain
crb_noise_variance = 1 / fisher_info_noise_variance
print("CRB for channel gain estimation:", crb_channel_gain)
print("CRB for noise variance estimation:", crb_noise_variance)

# Step 5: Beta Distribution Parameters and CRLB for Noise Variance Estimation
shape_a = 3  # New value for shape parameter 'a' of the Beta distribution
shape_b = 4  # New value for shape parameter 'b' of the Beta distribution
crlb_noise_variance_beta = shape_a * shape_b / ((shape_a + shape_b) ** 2 * (shape_a + shape_b + 1))
print("CRLB for noise variance estimation using Beta distribution:", crlb_noise_variance_beta)

# Step 6: ML Estimation
channel_gain_est = ml_channel_gain_estimator(combined_data)
noise_variance_est = ml_noise_variance_estimator(combined_data, channel_gain_est)

# Step 7: Calculate Mean Squared Error (MSE)
mse_channel_gain = np.mean((true_channel_gain - channel_gain_est) ** 2)
mse_noise_variance = np.mean((true_noise_variance - noise_variance_est) ** 2)
print("MSE for channel gain estimation:", mse_channel_gain)
print("MSE for noise variance estimation:", mse_noise_variance)

# Step 8: Comparison with Theoretical Bounds
theoretical_mse_channel_gain = crb_channel_gain / 12
theoretical_mse_noise_variance = crb_noise_variance / 12
print("Theoretical MSE for channel gain estimation:", theoretical_mse_channel_gain)
print("Theoretical MSE for noise variance estimation:", theoretical_mse_noise_variance)

# Step 9: Performance Analysis
def calculate_mse_vs_samples():
    num_runs = 150  # New number of simulation runs
    sample_range = np.arange(200, 2200, 200)  # New sample sizes to evaluate
    mse_channel_gain_values = []
    mse_noise_variance_values = []

    for num_samples_loop in sample_range:
        mse_channel_gain_avg = 0
        mse_noise_variance_avg = 0

        for _ in range(num_runs):
            rayleigh_data_loop = np.sqrt(true_channel_gain) * np.random.randn(num_samples_loop)  # Generate new data
            awgn_data_loop = np.sqrt(true_noise_variance) * np.random.randn(num_samples_loop)
            data_loop = rayleigh_data_loop + awgn_data_loop

            channel_gain_est_run = ml_channel_gain_estimator(data_loop)  # ML estimation for this run
            noise_variance_est_run = ml_noise_variance_estimator(data_loop, channel_gain_est_run)

            mse_channel_gain_run = (true_channel_gain - channel_gain_est_run) ** 2  # Calculate MSE for this run
            mse_noise_variance_run = (true_noise_variance - noise_variance_est_run) ** 2

            mse_channel_gain_avg += mse_channel_gain_run
            mse_noise_variance_avg += mse_noise_variance_run

        mse_channel_gain_avg /= num_runs
        mse_noise_variance_avg /= num_runs

        mse_channel_gain_values.append(mse_channel_gain_avg)
        mse_noise_variance_values.append(mse_noise_variance_avg)

    return sample_range, mse_channel_gain_values, mse_noise_variance_values

sample_range, mse_channel_gain_values, mse_noise_variance_values = calculate_mse_vs_samples()

# Step 10: Estimation of Alpha and Beta for Beta Distribution
# Generate beta-distributed data for estimation
beta_data = beta.rvs(a=shape_a, b=shape_b, size=1000)
# Maximum Likelihood Estimation (MLE) to estimate a and b
estimated_shape_a, estimated_shape_b, _, _ = beta.fit(beta_data, floc=0, fscale=1)
print("Estimated Alpha (a):", estimated_shape_a)
print("Estimated Beta (b):", estimated_shape_b)

# Step 11: Plotting Performance Analysis
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(sample_range, mse_channel_gain_values, marker='o')
plt.axhline(theoretical_mse_channel_gain, color='r', linestyle='--', label='Theoretical MSE')
plt.xlabel('Number of Samples')
plt.ylabel('MSE')
plt.title('Channel Gain Estimation Performance')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(sample_range, mse_noise_variance_values, marker='o')
plt.axhline(theoretical_mse_noise_variance, color='r', linestyle='--', label='Theoretical MSE')
plt.xlabel('Number of Samples')
plt.ylabel('MSE')
plt.title('Noise Variance Estimation Performance')
plt.legend()

plt.tight_layout()
plt.show()