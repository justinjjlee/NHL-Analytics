import torch
from torch import optim
from tqdm import tqdm
from .momentum_ssm import MomentumSSM

def fit_ssm(games, n_epochs=300, lr=1e-2, verbose=True):
    from torch.nn.utils.rnn import pad_sequence
    
    # Pre-pad all game sequences into batched tensors
    U_list = [g["U"] for g in games]
    Y_list = [g["Y"] for g in games]
    Z_list = [g["Z"] for g in games]
    dt_list = [g["dt"] for g in games]
    
    # padding_value=0.0 is fine since we use a mask to ignore NLL contributions
    U_pad = pad_sequence(U_list, batch_first=True, padding_value=0.0)
    Y_pad = pad_sequence(Y_list, batch_first=True, padding_value=0.0)
    Z_pad = pad_sequence(Z_list, batch_first=True, padding_value=0.0)
    dt_pad = pad_sequence(dt_list, batch_first=True, padding_value=0.0)
    
    # Create mask where 1.0 is valid event and 0.0 is padding
    lengths = torch.tensor([len(u) for u in U_list])
    mask = torch.arange(U_pad.shape[1])[None, :] < lengths[:, None]
    mask = mask.float()
    
    model = MomentumSSM(n_events=13, n_strength=4)
    # Add L2 Regularization (weight decay) to prevent coefficient explosion on rare events
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # Add Cosine Annealing scheduler to smoothly decay LR down to 1e-4 for precision convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    
    iterator = tqdm(range(n_epochs)) if verbose else range(n_epochs)
    for epoch in iterator:
        optimizer.zero_grad()
        
        # Batched forward pass
        total_nll = model(U_pad, Y_pad, Z_pad, dt_pad, mask)
            
        total_nll.backward()
        
        # Clip gradients to prevent unstable Kalman updates from outlier time deltas
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        if verbose and (epoch + 1) % 10 == 0:
            iterator.set_postfix({"Loss": f"{total_nll.item():.2f}", "LR": f"{scheduler.get_last_lr()[0]:.5f}"})
            
    return model

def bootstrap_B(games, n_boot=200, n_epochs=150):
    import numpy as np
    import random
    
    B_samples = []
    for _ in tqdm(range(n_boot), desc="Bootstrap"):
        # Resample games with replacement
        boot_games = [random.choice(games) for _ in range(len(games))]
        model = fit_ssm(boot_games, n_epochs=n_epochs, lr=1e-2, verbose=False)
        B_samples.append(model.B.detach().numpy().copy())
        
    return np.array(B_samples)
