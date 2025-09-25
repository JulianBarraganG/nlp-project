import torch
import torch.nn as nn
import numpy as np
import os


def train_lm(
        model: nn.Module, 
        train_dl: torch.utils.data.DataLoader, 
        valid_dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int, 
        device: torch.device,
        save_path: str,
        patience: int = 10
    ):
    '''
    Train a language model with early stopping on validation accuracy.
    '''
    losses = []
    best_acc = 0.0
    pcounter = 0

    for ep in range(n_epochs):

        loss_epoch = []

        for batch in train_dl:
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            inputs, input_lens, targets = batch
            logits = model(inputs, input_lens)
            # Mask out padding tokens using input_lens
            batch_size, seq_len, vocab_size = logits.size()
            mask = torch.arange(seq_len)[None, :].to(device) < input_lens[:, None]
            logits_flat = logits[mask]
            targets_flat = targets[mask]
            loss = nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=logits.size(-1)-1)
            loss_epoch.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(loss_epoch)
        losses.append(avg_train_loss)

        def evaluate_lm(model: nn.Module, device, valid_dl):
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in valid_dl:
                    # Move batch to device
                    batch = tuple(t.to(device) for t in batch)
                    inputs, input_lens, targets = batch
                    # Get logits from the model
                    logits = model(inputs, input_lens)
                    # Get predictions
                    preds = logits.argmax(dim=-1)
                    # Create mask to ignore padding in accuracy calculation
                    mask = (targets != logits.size(-1)-1)
                    # Calculate number of correct predictions
                    correct += ((preds == targets) & mask).sum().item()
                    # Calculate total number of tokens
                    total += mask.sum().item()

            return correct / total if total > 0 else 0.0
        
        acc = evaluate_lm(model, device, valid_dl)
        print(f'Validation accuracy: {acc}, train loss: {sum(loss_epoch) / len(loss_epoch)}')

        # Keep track of the best model based on the accuracy
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            pcounter = 0
        else:
            pcounter += 1
            if pcounter == patience:
                break
    model.load_state_dict(torch.load(save_path))
    return losses, best_acc

def train_classifier(
        model: nn.Module, 
        train_dl: torch.utils.data.DataLoader, 
        valid_dl: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int, 
        device: torch.device,
        save_path: str,
        patience: int = 10
    ):
    '''
    Train a language model with early stopping on validation accuracy.
    '''
    losses = []
    best_acc = 0.0
    pcounter = 0

    for ep in range(n_epochs):

        loss_epoch = []

        for batch in train_dl:
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            inputs, input_lens, labels = batch
            logits = model(inputs, input_lens)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            loss_epoch.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_train_loss = np.mean(loss_epoch)
        losses.append(avg_train_loss)

        def evaluate_lm(model: nn.Module, device, valid_dl):
            # Validation
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in valid_dl:
                    # Move batch to device
                    batch = tuple(t.to(device) for t in batch)
                    inputs, input_lens, targets = batch
                    
                    # Get logits from the model
                    logits = model(inputs, input_lens)
                    # Get predictions
                    preds = logits.argmax(dim=-1)
                    
                    # Calculate number of correct predictions
                    correct += (preds == targets).sum().item()
                    # Calculate total number of tokens
                    total += targets.size(0)

            return correct / total if total > 0 else 0.0
        
        acc = evaluate_lm(model, device, valid_dl)
        print(f'Validation accuracy: {acc}, train loss: {sum(loss_epoch) / len(loss_epoch)}')

        # Keep track of the best model based on the accuracy
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            pcounter = 0
        else:
            pcounter += 1
            if pcounter == patience:
                break
    model.load_state_dict(torch.load(save_path))
    return losses, best_acc





