import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from ViT import ViT
import torch
import torch.nn as nn


# Create a training config
training_config = {
    "d_model": 9,
    "n_classes": 10,
    "img_size": (32, 32),
    "patch_size": (16, 16),
    "n_channels": 1,
    "n_heads": 3,
    "n_layers": 3,
    "batch_size": 128,
    "epochs": 5,
    "alpha": 0.005,
}

# Create the transformations
transform = T.Compose([T.Resize(training_config["img_size"]), T.ToTensor()])

# Install the ds
train_ds = MNIST(root="./MNIST_train", download=True, train=True, transform=transform)
test_ds = MNIST(root="./MNIST_test", download=True, train=False, transform=transform)


# Load the ds
train_loader = DataLoader(
    train_ds, batch_size=training_config["batch_size"], shuffle=True
)
test_loader = DataLoader(
    test_ds, batch_size=training_config["batch_size"], shuffle=False
)


# Choose the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Using device: ",
    device,
    f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
)

vit = ViT(
    img_size=training_config["img_size"],
    patch_size=training_config["patch_size"],
    n_heads=training_config["n_heads"],
    d_model=training_config["d_model"],
    n_channels=training_config["n_channels"],
    n_layers=training_config["n_layers"],
    n_classes=training_config["n_classes"],
).to(device)


optimizer = Adam(vit.parameters(), lr=training_config["alpha"])
loss_function = nn.CrossEntropyLoss()


for epoch in range(training_config["epochs"]):

    loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = vit(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{training_config["epochs"]} loss: {loss  / len(train_loader) :.3f}"
    )


correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = vit(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"\nModel Accuracy: {100 * correct // total} %")
