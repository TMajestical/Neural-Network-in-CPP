import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import os

seed = 76  # You can set this to any integer
torch.manual_seed(seed)  # Set the seed for CPU operations

## allowing only one thread to get sequential processing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

print("fTotal Threads Used : {1}")

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.softmax(x, dim=1)  # Apply softmax to the output
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split the training dataset into training and validation datasets
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the network, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=0.001)

# Training the model
num_epochs = 20

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    training_loss = running_loss / len(train_loader)
    training_accuracy = 100 * correct / total
    end_time = time.time()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Train Acc.: {round(training_accuracy,2)}%, Train Loss: {round(training_loss,2)}, time : {round(end_time-start_time,2)}s, Val. Acc.: {round(val_accuracy,2)}%, Val. Loss: {round(val_loss,2)}")

# Testing the model
model.eval()
correct = 0
total = 0
test_loss = 0
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_accuracy = round(100 * correct / total,2)
test_loss = round(test_loss/len(test_loader),2)

print(f'\nTest Acc.: {test_accuracy}% Test Loss:{test_loss}\n')
