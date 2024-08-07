import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import ot
import math
from scipy.stats import bernoulli

# Neural Network definition
class Model(nn.Module):
    def __init__(self):
        super().__init__() # The 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        return logits

# Generate Data
list_x_values = [torch.rand(4, 2).float() for _ in range(1000)]
list_y_values  = [torch.rand(4, 2).float() for _ in range(1000)]

# Sinkhorn Regularization
def sinkhorn_regularization(A, B, C, epsilon=0.5):
    A = np.ones(len(A)) / len(A)
    B = np.ones(len(B)) / len(B)
    optimal_transport = ot.sinkhorn(A, B, C, epsilon)
    return optimal_transport

# Compute Cost Matrix
def compute_C(X, Y):
    return np.array([[math.dist(x, y) ** 2 for y in Y] for x in X])

# Update the cost matrix using a neural network.
def update_cost_matrix(neural_network, X, Y):
    with torch.no_grad():
        # Initialize an empty cost matrix
        NN_cost = torch.zeros(len(X), len(Y))
        # Iterate over X and Y values
        for a, b in enumerate(X):
            for c, d in enumerate(Y):
                # Concatenate input data (x and y) and flatten it
                input_data = torch.cat([b, d]).flatten().unsqueeze(0).float()
                # Compute the cost using the neural network
                NN_cost[a, c] = neural_network(input_data)
        # Convert the cost matrix to a numpy array and return
        return NN_cost.numpy()

# Method that samples from a distrubtion 
def metro_hasting_algo(iterations, A, B, x_value, y_value, transport_plan_pi_initial_list):
    # Initialize an empty list to store samples from the posterior distribution
    samples = []
    # Instantiate a new neural network model
    model = Model()
    # Initialize the parameter vector theta by concatenating all model parameters
    theta = np.concatenate([
        model.fc1.weight.data.flatten(),
        model.fc1.bias.data.flatten(),
        model.fc2.weight.data.flatten(),
        model.fc2.bias.data.flatten(),
        model.fc3.weight.data.flatten(),
        model.fc3.bias.data.flatten(),
        model.fc4.weight.data.flatten(),
        model.fc4.bias.data.flatten()
    ]).astype(np.double)
    # Set the proposal step size (tau)
    tau = .08
    # Main loop for Metropolis-Hastings algorithm
    for num in range(iterations):
        print(num)  # Print the current iteration number
        # Generate a random proposal (epsilon) from a normal distribution
        epsilon = np.random.normal(0, 0.5, len(theta))
        # Create a new proposal for theta
        theta_proposal = theta + tau * epsilon
        # Compute the acceptance probability
        p = min(1, compute_likelihood(A, B, theta_proposal, x_value, y_value, transport_plan_pi_initial_list) / 
                   compute_likelihood(A, B, theta, x_value, y_value, transport_plan_pi_initial_list))
        # Accept or reject the proposal based on a Bernoulli random variable
        if bernoulli.rvs(p, size=1)[0]:
            theta = theta_proposal
        # Append the current theta to the samples list
        samples.append(theta)
    # Return the list of samples
    return samples

# Computes the likelihood given a list of transport plans
def compute_likelihood(A, B, theta, x_value, y_value, list_cost):
    # Initialize the total likelihood to 1.0
    total_likelihood = 1.0
    # Iterate over each set of X, Y, and initial transport plan
    for X, Y, cost in zip(x_value, y_value, list_cost):
        # Instantiate a new neural network model
        model = Model()
        # Set the weights and biases of the first layer (fc1) of the model
        model.fc1.weight.data = torch.tensor(theta[:32].reshape(8, 4)).float()
        model.fc1.bias.data = torch.tensor(theta[32:40]).float()
        # Set the weights and biases of the second layer (fc2) of the model
        model.fc2.weight.data = torch.tensor(theta[40:168].reshape(16, 8)).float()
        model.fc2.bias.data = torch.tensor(theta[168:184]).float()
        # Set the weights and biases of the third layer (fc3) of the model
        model.fc3.weight.data = torch.tensor(theta[184:312].reshape(8, 16)).float()
        model.fc3.bias.data = torch.tensor(theta[312:320]).float()
        # Set the weights and biases of the fourth layer (fc4) of the model
        model.fc4.weight.data = torch.tensor(theta[320:328].reshape(1, 8)).float()
        model.fc4.bias.data = torch.tensor(theta[328:329]).float()
        # Update the cost matrix using the neural network model
        COST = update_cost_matrix(model, X, Y)
        # Compute the new transport plan using Sinkhorn regularization
        new_pi_cost  = sinkhorn_regularization(A, B, COST)
        # Calculate the likelihood for this set of X, Y, and initial transport plan
        likelihood_k = np.exp(-(np.linalg.norm(new_pi_cost  - cost ) / 0.1) ** 2)
        # Update the total likelihood, ensuring it doesn't go below a threshold
        total_likelihood *= max(likelihood_k, 1e-100)
    # Return the total likelihood after processing all sets
    return total_likelihood


# Generate list of COST values
list_COST = [sinkhorn_regularization(np.ones(len(x)), np.ones(len(y)), compute_C(x, y)) for x, y in zip(list_x_values, list_y_values )]

# Run Metropolis-Hastings Algorithm
iterations = 1000
samples = metro_hasting_algo(iterations, np.ones(len(list_x_values[0])), np.ones(len(list_y_values [0])), list_x_values, list_y_values , list_COST)

theta_mean = np.mean(samples, axis=0)

# Update Model with Estimated Parameters
model = Model()
model.fc1.weight.data = torch.tensor(theta_mean[:32].reshape(8, 4)).float()
model.fc1.bias.data = torch.tensor(theta_mean[32:40]).float()
model.fc2.weight.data = torch.tensor(theta_mean[40:168].reshape(16, 8)).float()
model.fc2.bias.data = torch.tensor(theta_mean[168:184]).float()
model.fc3.weight.data = torch.tensor(theta_mean[184:312].reshape(8, 16)).float()
model.fc3.bias.data = torch.tensor(theta_mean[312:320]).float()
model.fc4.weight.data = torch.tensor(theta_mean[320:328].reshape(1, 8)).float()
model.fc4.bias.data = torch.tensor(theta_mean[328:329]).float()


list_COST = np.mean(list_COST, axis=0)
new_transport_plan = []

# Compare Transport Plans
for i in range(len(list_x_values)):
    COST = update_cost_matrix(model, list_x_values[i], list_y_values[i])
    new_transport_plan.append(sinkhorn_regularization(np.ones(len(list_x_values[i])), np.ones(len(list_y_values [i])), COST))

new_transport_plan_mean = np.mean(new_transport_plan, axis=0)

print(list_COST)
print("\n")
print(new_transport_plan_mean)

# Plotting the results
num_samples = len(new_transport_plan)
plt.figure(figsize=(10, 6))
sampler = [sample[1, 1] for sample in new_transport_plan]

plt.plot(range(num_samples), sampler, label='State (0,0)')
plt.xlabel('Iterations')
plt.ylabel('State Value')
plt.title('Metropolis-Hastings Sampling')
plt.legend()
plt.grid(True)
plt.show()








    








    

