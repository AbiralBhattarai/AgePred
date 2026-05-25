import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

from core import config
from utils.data_transformation import preprocess


###MAE MODEL
class AgeRegressionModelMAE(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        if backbone_name == "resnet18":
            backbone    = resnet18(weights=None)
            in_features = 512
        elif backbone_name == "resnet34":
            backbone    = resnet34(weights=None)
            in_features = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        feats = torch.flatten(feats, 1)
        age   = self.regression_head(feats)
        return age.squeeze(1)

###CORAL MODEL



class AgeRegressionModelCORAL(nn.Module):
    def __init__(
        self,
        backbone_name: str = config.BACKBONE_CORAL,
        hidden_dim: int    = config.HIDDEN_DIM_CORAL,
        dropout: float     = config.DROPOUT_CORAL,
    ):
        super().__init__()

        backbone    = resnet18(weights=None) if backbone_name == "resnet18" else resnet34(weights=None)
        in_features = 512

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.coral_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, config.NUM_CLASSES_CORAL - 1),  # K-1 binary classifiers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.feature_extractor(x)
        feats  = torch.flatten(feats, 1)
        logits = self.coral_head(feats)     # (B, K-1)
        return logits

###CORAL_10bin
class AgeRegressionModelCORAL10(nn.Module):
    def __init__(
        self,
        backbone_name: str = config.BACKBONE_CORAL,
        hidden_dim: int    = config.HIDDEN_DIM_CORAL,
        dropout: float     = config.DROPOUT_CORAL,
    ):
        super().__init__()

        backbone    = resnet18(weights=None) if backbone_name == "resnet18" else resnet34(weights=None)
        in_features = 512

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.coral_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, config.NUM_CLASSES_CORAL_10 - 1),  # K-1 binary classifiers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.feature_extractor(x)
        feats  = torch.flatten(feats, 1)
        logits = self.coral_head(feats)     # (B, K-1)
        return logits



def load_model_mae() -> AgeRegressionModelMAE:
    model = AgeRegressionModelMAE(backbone_name = config.BACKBONE_MAE,
        hidden_dim = config.HIDDEN_DIM_MAE,
        dropout = config.DROPOUT_MAE)
    state_dict = torch.load(config.CHECKPOINT_PATH_MAE, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    print(f"[✓] Loaded checkpoint: '{config.CHECKPOINT_PATH_MAE}'  (device={config.DEVICE})")
    return model



def load_model_coral() -> AgeRegressionModelCORAL:
    model = AgeRegressionModelCORAL(backbone_name = config.BACKBONE_CORAL,
        hidden_dim = 128,
        dropout = 0.3)
    state_dict = torch.load(config.CHECKPOINT_PATH_CORAL, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    print(f"[✓] Loaded checkpoint: '{config.CHECKPOINT_PATH_CORAL}'  (device={config.DEVICE})")
    return model




def predict_age_mae(model, image_path: str) -> float:
    tensor = preprocess(image_path).to(config.DEVICE)
    with torch.no_grad():
        age = model(tensor).item()
    return age

def predict_age_coral_5(model, image_path: str) -> float:
    tensor = preprocess(image_path).to(config.DEVICE)
    with torch.no_grad():
        logits    = model(tensor)                          # (1, K-1)
        probs     = torch.sigmoid(logits)                  # (1, K-1)
        bin_index = (probs > 0.5).sum(dim=1).item()        # predicted bin
        age       = bin_index * config.BIN_SIZE_CORAL + config.BIN_SIZE_CORAL / 2  # midpoint
    return float(age)






model_mae = load_model_mae()
model_coral_5 = load_model_coral()





image_path = 'data/shreejal.jpg'
from time import perf_counter
time_mae_start = perf_counter()
age_mae = predict_age_mae(model_mae, image_path)
time_mae_end = perf_counter()
print(f"MAE inference time: {time_mae_end - time_mae_start:.3f} seconds")
time_coral_5_start = perf_counter()
age_coral_5 = predict_age_coral_5(model_coral_5,image_path)
time_coral_5_end = perf_counter()
print(f"CORAL inference time: {time_coral_5_end - time_coral_5_start:.3f} seconds")
print("*"*50)
print("MAE:")
print(f"Predicted age: {age_mae:.1f} yrs")
print("*"*50)
print("CORAL:")
print(f"Predicted age: {age_coral_5:.1f} yrs")
print("*"*50)
print