import torch
import numpy as np

def compute_loss(model, loss_func, inputs, outputs, device) -> torch.tensor:
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    logits = model(inputs)
    logits = logits.reshape(-1, logits.shape[-1])
    loss = loss_func(logits, outputs.flatten().to(device))
    return loss

def train_mess3(model, optimizer, loss_func, experiment_params, writer, train_loader=None, valid_loader=None):

    for batch_idx, (x_shift, x, _) in enumerate(train_loader):
        model.train()
        loss = compute_loss(model, loss_func, x_shift, x, experiment_params.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step=batch_idx)
        
        if batch_idx % 100 == 0:
            model.eval()
            valid_loader.dataset.reset_counter()
            losses = []
            for _, (x_shift, x, _) in enumerate(valid_loader):
                loss = compute_loss(model, loss_func, x_shift, x, experiment_params.device).item() 
                losses.append(loss)
            avg_loss = torch.tensor(losses).mean().item()
            writer.add_scalar("valid/loss", avg_loss, global_step=batch_idx)
            print(f"Validation loss @ batch {batch_idx}:", avg_loss)


# This function to project ground truth/pred belief states was authored by GPT
def barycentric_to_cartesian(beliefs: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates (n_samples, 3) on a 2-simplex 
    to Cartesian coordinates for plotting.
    """
    # Vertices of an equilateral triangle simplex
    vertices = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
    ])
    return beliefs @ vertices  # (n_samples, 2)