import numpy as np
import ot
import matplotlib.pyplot as plt

# Function to perform Sinkhorn regularization for optimal transport
def sinkhorn_regularization(A, B, C):
    epsilon = 1  # Regularization parameter
    optimal_transport = ot.sinkhorn(A, B, C, epsilon)  # Compute the Sinkhorn optimal transport plan
    return optimal_transport

# Function to update the cost matrix based on input values
def update_cost_matrix(COST, x_value, y_value):
    Q = len(x_value)  # Number of value sets
    S, R = COST.shape  # Shape of the original cost matrix
    N, M = len(x_value[0]), len(y_value[0])  # Dimensions of the new cost matrices
    all_cost = []  # List to store all updated cost matrices
    for q in range(Q):
        C = np.zeros((N, M))  # Initialize a new cost matrix
        for i in range(N):
            for j in range(M):
                for s in range(S):
                    for r in range(R):
                        # Update cost matrix based on cosine functions of the original cost matrix
                        C[i, j] += COST[s, r] * np.cos(2 * np.pi * s * x_value[q][i]) * np.cos(2 * np.pi * r * y_value[q][j])
        all_cost.append(C)  # Add the updated cost matrix to the list
    return all_cost

# Function to compute the likelihood
def compute_likelihood(A, B, xi_ll, x_value, y_value, C_list):
    total_likelihood = 1.0  # Initialize total likelihood
    for q in range(len(x_value)):
        X = x_value[q]  # Current set of x values
        Y = y_value[q]  # Current set of y values
        initial_transport_plan = sinkhorn_regularization(A, B, C_list[q])  # Initial transport plan
        C = update_cost_matrix(xi_ll, [X], [Y])[0]  # Update cost matrix with current values
        new_pi = sinkhorn_regularization(A, B, C)  # New transport plan
        # Compute the likelihood
        likelihood_k = np.exp(-(np.linalg.norm(new_pi - initial_transport_plan) / 0.01) ** 2)
        total_likelihood *= max(likelihood_k, 1e-100)  # Update total likelihood
    return total_likelihood

# Function implementing the Metropolis-Hastings algorithm
def metro_hasting_algo(iterations, A, B, x_value, y_value, C_list):
    samples = []  # List to store samples
    current_cost_scaler = np.zeros(C_list[0].shape)  # Initial cost scaler
    list_proposal = [current_cost_scaler.copy()]  # List to store proposals

    for i in range(iterations):
        print(i)  # Print the current iteration
        epsilon_xi = np.random.normal(0, 0.05, size=C_list[0].shape)  # Random noise
        proposal_cost_scaler = current_cost_scaler + 0.1 * epsilon_xi  # New proposal

        # Compute likelihoods for current and new proposals
        likelihood_previous = compute_likelihood(A, B, current_cost_scaler, x_value, y_value, C_list)
        likelihood_new = compute_likelihood(A, B, proposal_cost_scaler, x_value, y_value, C_list)

        # Acceptance probability
        acceptance_probability = min(1, likelihood_new / likelihood_previous)
        decision = np.random.binomial(1, acceptance_probability)  # Accept or reject the new proposal

        if decision == 1:
            current_cost_scaler = proposal_cost_scaler  # Update current cost scaler if proposal is accepted

        list_proposal.append(current_cost_scaler.copy())  # Add the proposal to the list
        samples.append(current_cost_scaler.copy())  # Add the sample to the list

    # Compute the average cost scaler
    cost_scaler_average = np.mean(list_proposal, axis=0)
    return samples, cost_scaler_average

# Example usage
N, M = 4, 4  # Dimensions of the transport problem
S, R = 4, 4  # Dimensions of the original cost matrix
Q = 10  # Number of value sets

np.random.seed(42)  # Set seed for reproducibility

# Example COST matrix
COST_ORIGINAL = np.random.rand(S, R)

# Example x and y value lists
x_value = [np.random.rand(N) for _ in range(Q)]
y_value = [np.random.rand(M) for _ in range(Q)]

# Update cost matrix
C_list = update_cost_matrix(COST_ORIGINAL, x_value, y_value)

A = np.ones(N) / N  # Uniform distribution for A
B = np.ones(M) / M  # Uniform distribution for B

# Run Metropolis-Hastings algorithm
iterations = 100000
samples, cost_scaler_average = metro_hasting_algo(iterations, A, B, x_value, y_value, C_list)

print("Original Cost Matrix:")
print(COST_ORIGINAL)

print("Proposal Cost Matrix:")
print(cost_scaler_average)

# Plotting the results
num_samples = len(samples)
plt.figure(figsize=(10, 6))
sampler = [sample[1, 1] for sample in samples]

plt.plot(range(num_samples), sampler, label='State (0,0)')
plt.xlabel('Iterations')
plt.ylabel('State Value')
plt.title('Metropolis-Hastings Sampling')
plt.legend()
plt.grid(True)
plt.show()