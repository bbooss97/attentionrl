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
learning_rate = 1e-2
num_iterations = 100000000
meta_batch_size = 32

# Initialize agent and meta-network
agent = AgentNetwork(color=False, useLstm=False, extractorOutput=1, qDimension=3, kDimension=3, firstBests=10, num=num_parallel, useAttentionController=True, threshold=0)
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

for iteration in range(num_iterations):
    # Collect actual Q-values
    env = Gymenv1player(agent=agent, maxsteps=1000, verbose=False, gameName=game, num=num_parallel)
    actual_q_values = -env.play()*100  # Assuming higher is better

    # Predict Q-values using meta-network
    agent_weights = torch.tensor(agent.getparameters(), requires_grad=True)
    if torch.cuda.is_available():
        agent_weights = agent_weights.cuda()
    optimparams= optim.Adam([agent_weights], lr=1e-3)

    # Perform multiple updates manually
    num_updates = 1000  # You can adjust this number
    for i in range(num_updates):
        predicted_q_value = meta_network(agent_weights)
        if i==0:
            initial_predicted_q_value=predicted_q_value

        # Calculate loss
        loss = mse_loss(predicted_q_value, torch.tensor([actual_q_values]).double().cuda())
        
        # Manually calculate gradients
        loss.backward()
        
        # Manually update weights
        optimparams.step()
        optimparams.zero_grad()
        
    # Load the final updated weights into the agent
    agent.loadparameters(agent_weights.cpu().detach().numpy())

    # Optionally, update the meta-network parameters
    optimizer.zero_grad()
    final_predicted_q_value = meta_network(agent_weights)
    final_loss = mse_loss(final_predicted_q_value, torch.tensor([actual_q_values]).double().cuda())
    final_loss.backward()
    optimizer.step()
    # Logging
    print(f"Iteration {iteration}: Actual Q-value: {actual_q_values}, Predicted Q-value: {initial_predicted_q_value.item()}, Loss: {loss.item()} it learns this new value: {predicted_q_value.item()}")
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