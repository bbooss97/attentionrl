import torch
import torch.nn as nn
import torch.optim as optim
from parallelAgent import AgentNetwork
from parallelGymEnvironment import Gymenv1player
import wandb
import time
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)


# Meta-network to predict Q-values
class MetaNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(MetaNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output is a single Q-value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
num_parallel = 20
game = "starpilot"
learning_rate = 1e-4
num_iterations = 1000
meta_batch_size = 32

# Initialize agent and meta-network
agent = AgentNetwork(color=False, useLstm=True, extractorOutput=1, qDimension=3, kDimension=3, firstBests=10, num=num_parallel, useAttentionController=False, threshold=0)
meta_network = MetaNetwork(input_size=len(agent.getparameters()))

if torch.cuda.is_available():
    agent.cuda()
    meta_network.cuda()

optimizer = optim.Adam(meta_network.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

# Initialize WandB
use_wandb = False
if use_wandb:
    wandb.init(project='meta_learning_rl', entity='your_entity_here', name='meta_learning_run')
    wandb.watch(agent)
    wandb.watch(meta_network)

# Training loop
for iteration in range(num_iterations):
    # Collect actual Q-values
    env = Gymenv1player(agent=agent, maxsteps=500, verbose=False, gameName=game, num=num_parallel)
    actual_q_values = 100 - env.play()  # Assuming higher is better

    # Predict Q-values using meta-network
    agent_weights = torch.tensor(agent.getparameters(),requires_grad=True)
    if torch.cuda.is_available():
        agent_weights = agent_weights.cuda()
    predicted_q_value = meta_network(agent_weights)

    # Train meta-network
    loss = mse_loss(predicted_q_value, torch.tensor([actual_q_values]).double().cuda())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update agent weights
    updated_weights = agent_weights - learning_rate * agent_weights.grad
    agent.loadparameters(updated_weights.cpu().detach().numpy())

    # Logging
    print(f"Iteration {iteration}: Actual Q-value: {actual_q_values}, Predicted Q-value: {predicted_q_value.item()}, Loss: {loss.item()}")
    if use_wandb:
        wandb.log({
            "iteration": iteration,
            "actual_q_value": actual_q_values,
            "predicted_q_value": predicted_q_value.item(),
            "loss": loss.item()
        })

    # Save model periodically
    if iteration % 100 == 0:
        torch.save(agent.state_dict(), f"./agent_checkpoint_{iteration}.pt")
        torch.save(meta_network.state_dict(), f"./meta_network_checkpoint_{iteration}.pt")
        if use_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(f"./agent_checkpoint_{iteration}.pt")
            artifact.add_file(f"./meta_network_checkpoint_{iteration}.pt")
            wandb.log_artifact(artifact)

print("Training complete!")