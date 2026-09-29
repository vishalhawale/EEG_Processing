import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class EEG1DCNN(nn.Module):
    def __init__(self, in_features):
        super(EEG1DCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        return torch.sigmoid(self.net(x)).squeeze(1)


def initialize_classical_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
    }


def train_cnn(X_train_scaled, y_train_res, X_test_scaled, epochs=15, batch_size=128, lr=0.001):
    print("--> Training PyTorch 1D-CNN Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EEG1DCNN(X_train_scaled.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_res, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        cnn_probs = model(test_tensor_x).cpu().numpy()
    
    cnn_preds = (cnn_probs >= 0.5).astype(int)
    return cnn_probs, cnn_preds