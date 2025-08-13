import os
import random
import re

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Note: 'date_to_int' and 'Encoder' are not provided, so we will
# create placeholder functions/classes to make the script runnable.
# In a real-world scenario, you would import these from your files.

# Placeholder for the user's Encoder model
class Encoder(nn.Module):
    def __init__(self, dim, nhead, ratio, depth, max_length):
        super().__init__()
        # This is a simplified stand-in for the user's custom encoder.
        # It's a simple transformer encoder block.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=int(dim * ratio),
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.embedding = nn.Linear(1, dim)
        self.output_layer = nn.Linear(dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, lengths, mask_idx):
        # The user's original forward pass had a different signature
        # and logic. We'll adapt it to the training loop requirements.
        # This is a simplified forward pass for demonstration.
        # Assume x has shape [batch_size, sequence_length, 1].
        
        # 1. Embed the input
        x_embedded = self.embedding(x)

        # 2. Apply a mask for the masked language modeling task.
        # We need to create a padding mask for the Transformer.
        max_len = x.size(1)
        # Assuming lengths are not used for padding in this simplified example
        # Let's create a dummy mask for the masked indices
        
        # The user's logic was for masked language modeling.
        # We'll simulate this by zeroing out the input at the mask indices.
        masked_input = x.clone()
        for i in range(x.size(0)):
            # The mask_idx from the user's original code was a list of masks
            # for each item in the batch. Let's use that.
            if i < len(mask_idx):
                masked_input[i, mask_idx[i], :] = 0  # Zero out the masked positions
        
        # 3. Pass through the transformer
        transformer_output = self.transformer_encoder(self.embedding(masked_input))
        
        # 4. Get the output for the masked positions
        predictions = []
        target = []
        for i in range(x.size(0)):
            if i < len(mask_idx):
                predictions.append(self.output_layer(transformer_output[i, mask_idx[i], :]))
                target.append(x[i, mask_idx[i], :])

        if not predictions:
            # Handle empty predictions to avoid error
            return torch.tensor(0.0), None

        predictions = torch.cat(predictions)
        target = torch.cat(target)
        
        # 5. Calculate loss
        loss = self.loss_fn(predictions, target)
        
        return loss, self.output_layer(transformer_output)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataset(data, window_size, stride):
    """
    Creates a dataset using a sliding window with a specified stride.

    Args:
        data (np.array): The time series data.
        window_size (int): The size of each window.
        stride (int): The number of steps to move the window.

    Returns:
        tuple: A tuple containing the input sequences (X) and target values (y).
    """
    X, y = [], []
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def main():
    set_seed(42)
    
    # Check for GPU and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For this example, we will generate dummy data since the user's
    # 'train.csv' is not available and the 'groupby' loop is incomplete.
    # We'll use a single time series to demonstrate the core logic.
    print("Generating dummy time series data for demonstration...")
    sales_data = np.random.rand(500, 1) # A single time series of length 500
    
    # User-defined parameters
    window_size = 21
    stride = 7
    epochs = 10
    
    # Create the dataset using the sliding window with stride
    X, y = create_dataset(sales_data, window_size=window_size, stride=stride)
    
    # Reshape X to be [batch_size, sequence_length, feature_dim]
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Create a DataLoader for batch processing
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize the model, optimizer, and AMP scaler
    model = Encoder(dim=64, nhead=4, ratio=4.0, depth=6, max_length=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Start the training loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_x, batch_y in progress_bar:
            optimizer.zero_grad()
            
            batch_size = batch_x.size(0)
            
            mask_idx = []
            for i in range(batch_size):
                span = random.randint(5, 14)

                start_idx = random.randint(0, window_size - span)
                
                mask_idx.append(torch.arange(start_idx, start_idx + span, device=device))

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss, _ = model(x=batch_x, lengths=None, mask_idx=mask_idx)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
