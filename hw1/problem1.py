import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import StronglyEntanglingLayers

# Set seed
seed = 90921008  # replace with your student ID
np.random.seed(seed)
torch.manual_seed(seed)
pnp.random.seed(seed)

# Target function
def target_func(x):
    return torch.sin(torch.exp(x[:, 0]) + x[:, 1])

# Generate data
train_input = torch.zeros(1000, 2)  # 1000 samples, 2 variables
train_ranges = np.array([0, 0.5] * 2).reshape(2, 2)
test_input = torch.zeros(1000, 2)
test_ranges = np.array([0.5, 1] * 2).reshape(2, 2)
for i in range(2):
    train_input[:, i] = (
        torch.rand(1000) * (train_ranges[i, 1] - train_ranges[i, 0]) + train_ranges[i, 0]
    )
    test_input[:, i] = (
        torch.rand(1000) * (test_ranges[i, 1] - test_ranges[i, 0]) + test_ranges[i, 0]
    )

train_label = target_func(train_input)
test_label = target_func(test_input)

# Convert to numpy for PennyLane
train_x = train_input.numpy()
train_y = train_label.numpy()
test_x = test_input.numpy()
test_y = test_label.numpy()

# Quantum model setup
n_qubits = 3
n_layers = 2  # number of data reuploading layers
n_ansatz_layers = 2  # layers in StronglyEntanglingLayers

pl_qml_gpu_dev = "lightning.gpu"
# pl_qml_gpu_dev = "default.qubit"

dev = qml.device(pl_qml_gpu_dev, wires=n_qubits)

def S(x):
    """Data encoding"""
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    qml.RX(x[0] + x[1], wires=2)  # encode combination

def W(theta):
    """Trainable block"""
    StronglyEntanglingLayers(theta, wires=range(n_qubits))

@qml.qnode(dev)
def quantum_model(weights, x):
    for i in range(n_layers):
        W(weights[i])
        S(x)
    W(weights[-1])  # final W
    return qml.expval(qml.PauliZ(wires=0))

# Show model topology (text + optional plot)
weights_shape = (n_layers + 1, n_ansatz_layers, n_qubits, 3)
dummy_weights = pnp.zeros(weights_shape)
dummy_x = pnp.array([0.1, 0.2])

print("\nQML model topology (ASCII):")
print(qml.draw(quantum_model, level="device")(dummy_weights, dummy_x))

try:
    import matplotlib.pyplot as plt

    fig, ax = qml.draw_mpl(quantum_model, level="device")(dummy_weights, dummy_x)
    ax.set_title("QML model topology")
    plt.show()
except Exception as e:
    print(f"[Topology plot skipped] {e}")

# Initialize weights
weights = 2 * pnp.pi * pnp.random.random(size=weights_shape, requires_grad=True)

# Loss function
def mse_loss(predictions, targets):
    return pnp.mean((predictions - targets) ** 2)

def cost(weights, x, y):
    predictions = pnp.array([quantum_model(weights, x_i) for x_i in x])
    return mse_loss(predictions, y)

# Optimizer
opt = qml.AdamOptimizer(stepsize=0.01)
max_steps = 200
batch_size = 50

train_losses = []
test_losses = []

for step in range(max_steps):
    # Select batch
    batch_idx = pnp.random.randint(0, len(train_x), batch_size)
    x_batch = train_x[batch_idx]
    y_batch = train_y[batch_idx]

    # Update weights
    weights = opt.step(lambda w: cost(w, x_batch, y_batch), weights)

    # Compute losses
    train_loss = cost(weights, train_x, train_y)
    test_loss = cost(weights, test_x, test_y)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if (step + 1) % 20 == 0:
        print(f"Step {step+1}: Train MSE = {train_loss:.4f}, Test MSE = {test_loss:.4f}")

# Final evaluation
final_train_mse = cost(weights, train_x, train_y)
final_test_mse = cost(weights, test_x, test_y)
print(f"Final Train MSE: {final_train_mse:.4f}")
print(f"Final Test MSE: {final_test_mse:.4f}")

# Number of trainable parameters
n_params = pnp.prod(pnp.array(weights_shape))
print(f"Number of trainable parameters: {n_params}")