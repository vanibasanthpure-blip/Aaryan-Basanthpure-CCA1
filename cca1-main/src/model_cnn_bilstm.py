"""Approach 3: compact 1D-CNN + BiLSTM hybrid for raw-beat temporal pattern
learning (PyTorch). Kept deliberately small -- two conv blocks and a single
BiLSTM layer -- since MIT-BIH beats are short (245 samples) and this is a
course-scale project, not a production model.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from utils import SEED


class CNNBiLSTM(nn.Module):
    def __init__(self, n_classes: int, lstm_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden, batch_first=True,
                             bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x):
        # x: (batch, length) -> (batch, 1, length)
        x = x.unsqueeze(1)
        x = self.conv(x)               # (batch, 32, length/4)
        x = x.permute(0, 2, 1)         # (batch, seq, 32) for LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # final time step, both directions
        out = self.dropout(out)
        return self.fc(out)


def train_cnn_bilstm(X_train, y_train, X_val, y_val, n_classes, class_weights=None,
                      epochs=25, batch_size=128, lr=1e-3, patience=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    model = CNNBiLSTM(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.long))
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.long).to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    # Model selection uses validation macro-F1, not val loss: with a rare
    # class (Q) at single-digit support, weighted val loss is dominated by
    # one or two examples and is too noisy to trust for checkpointing.
    best_val_f1, best_state, epochs_no_improve = -np.inf, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running_loss += loss.item() * len(xb)
        train_loss = running_loss / len(train_ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        print(f"[CNN-BiLSTM] epoch {epoch:02d} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[CNN-BiLSTM] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_cnn_bilstm(model, X, device=None, batch_size=256):
    device = device or next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
    return np.concatenate(preds), np.concatenate(probs)
