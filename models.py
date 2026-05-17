"""Modellarchitekturen: MLP und Random Forest."""

import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class MLP(nn.Module):
    """Mehrschichtiges NN mit flexibler Tiefe."""

    def __init__(self, input_dim: int, 
                 hidden_dim: int, 
                 num_layers: int,
                 dropout: float, 
                 activation: str, 
                 num_classes: int = 2):
        super().__init__()

        akt = nn.ReLU() if activation == "relu" else nn.Tanh()

        # erste layer: input_dim zu hidden_dim
        layers = [nn.Linear(input_dim, hidden_dim), akt, nn.Dropout(dropout)]

        # alle hidden layers
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), akt, nn.Dropout(dropout)]

        # letzte layer: hidden_dim zu num_classes
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RFModel:
    """Wrapper um RandomForestClassifier für einheitliche Schnittstelle."""

    def __init__(self,
                 n_estimators: int,
                 max_depth: int,
                 min_samples_split: int,
                 max_features: str,
                 min_samples_leaf: int,
                 criterion: str,
                 max_samples: float,
                 random_state: int = None):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            max_samples=max_samples,
            n_jobs=-1,
            random_state=random_state,
        )
