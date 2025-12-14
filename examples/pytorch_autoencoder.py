"""
Example: Using EQODEC CarbonAwareLoss with a generic PyTorch autoencoder.
Works for images, video frames, or any latent-based compression model.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from eqodec import CarbonAwareLoss, get_local_carbon_intensity


# -------------------------------
# Dummy Autoencoder (placeholder)
# -------------------------------
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


# -------------------------------
# Setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleAutoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Fetch carbon intensity once
carbon_intensity = get_local_carbon_intensity()

# EQODEC loss
criterion = CarbonAwareLoss(
    lambda_recon=5.0,
    lambda_rate=0.5,
    lambda_carbon=0.005,
    carbon_intensity=carbon_intensity
)

# Dummy input batch (e.g., images or video frames)
x = torch.rand(8, 3, 256, 256).to(device)


# -------------------------------
# Training step
# -------------------------------
model.train()
optimizer.zero_grad()

x_hat, latent = model(x)

loss, metrics = criterion(x_hat, x, latent)
loss.backward()
optimizer.step()

print("Training step completed.")
print(f"Total Loss: {loss.item():.4f}")
print("Loss breakdown:", {k: float(v) for k, v in metrics.items()})
