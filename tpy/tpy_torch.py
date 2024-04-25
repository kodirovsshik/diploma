# import torch
# from torch import nn
# from torch.utils.data import DataLoader
import torchvision.datasets
# from torchvision.transforms import ToTensor

# # Get cpu, gpu or mps device for training.
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

training_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Download test data from open datasets.
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# # Define model

# model = NeuralNetwork().to(device)
# print(model)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break


# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)






import torch
from torch import nn
from torch.utils.data import DataLoader

leaky_rely_alpha = 0.1

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(leaky_rely_alpha),
            nn.Linear(512, 512),
            nn.LeakyReLU(leaky_rely_alpha),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

batch_size = 1
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
  for data, target in train_dataloader:
    optimizer.zero_grad()
    prediction = model(data)
    loss = loss_fn(prediction.squeeze(), target)
    loss.backward()
    optimizer.step()

    # Print loss (optional)
    if (epoch + 1) % 100 == 0:
      print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# After training, use the model for prediction on new data
# ... (replace with your prediction logic)
