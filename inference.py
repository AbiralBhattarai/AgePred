import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.models import resnet18, resnet34

from core import config
from utils.data_transformation import preprocess


RESULTS_DIR = Path("results")


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


def save_prediction_grid(
    image_paths: list[str],
    mae_ages: list[float],
    coral_ages: list[float],
    output_name: str = "age_predictions_grid.png",
) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    num_images = len(image_paths)
    if num_images != len(mae_ages) or num_images != len(coral_ages):
        raise ValueError("image_paths, mae_ages, and coral_ages must have the same length")

    output_paths: list[Path] = []
    images_per_page = 10

    for page_index, start_index in enumerate(range(0, num_images, images_per_page), start=1):
        page_image_paths = image_paths[start_index:start_index + images_per_page]
        page_mae_ages = mae_ages[start_index:start_index + images_per_page]
        page_coral_ages = coral_ages[start_index:start_index + images_per_page]

        figure, axes = plt.subplots(2, 5, figsize=(4 * 5, 8))
        axes = axes.reshape(2, 5)

        for index, image_path in enumerate(page_image_paths):
            row_index = index // 5
            column_index = index % 5
            image = Image.open(image_path).convert("RGB")

            axis = axes[row_index][column_index]
            axis.imshow(image)
            axis.set_title(
                f"MAE: {page_mae_ages[index]:.1f} yrs\nCORAL: {page_coral_ages[index]:.1f} yrs",
                fontsize=11,
            )
            axis.axis("off")

        for index in range(len(page_image_paths), images_per_page):
            row_index = index // 5
            column_index = index % 5
            axes[row_index][column_index].axis("off")

        axes[0][0].set_ylabel("Row 1", fontsize=14, rotation=0, labelpad=35, va="center")
        axes[1][0].set_ylabel("Row 2", fontsize=14, rotation=0, labelpad=25, va="center")
        figure.suptitle(f"Age Predictions - Page {page_index}", fontsize=16)
        figure.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = RESULTS_DIR / f"{Path(output_name).stem}_page_{page_index}{Path(output_name).suffix or '.png'}"
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.2)
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths






model_mae = load_model_mae()
model_coral_5 = load_model_coral()





image_paths = ['data/test/galinalaofficial.jpg',
            'data/test/jamalliggin.jpg',
            'data/test/mattyicefitnesss.jpg',
            'data/test/mo_zillah.jpg',
            'data/test/thiscurvygirlsfitness.jpg',
            "data/bishal.jpg",
            "data/shreejal.jpg",
            "data/akriti.png",
            "data/abiral.png",
            "data/abiral-headshot.jpg",
            'data/bishal.jpg',
            "data/shreejal.jpg",
            "data/akriti.png",
            "data/abiral.png",
            "data/abiral-headshot.jpg"]
from time import perf_counter
mae_ages = []
coral_ages = []
time_mae_start = perf_counter()
for item in image_paths:
    print(f'Name:{item[9:]}')
    age_mae = predict_age_mae(model_mae, item)
    mae_ages.append(age_mae)
    time_mae_end = perf_counter()
    print(f"MAE inference time: {time_mae_end - time_mae_start:.3f} seconds")
    time_coral_5_start = perf_counter()
    age_coral_5 = predict_age_coral_5(model_coral_5,item)
    coral_ages.append(age_coral_5)
    time_coral_5_end = perf_counter()
    print(f"CORAL inference time: {time_coral_5_end - time_coral_5_start:.3f} seconds")
    print("*"*50)
    print("MAE:")
    print(f"Predicted age: {age_mae:.1f} yrs")
    print("*"*50)
    print("CORAL:")
    print(f"Predicted age: {age_coral_5:.1f} yrs")
    print()
    print("*"*50)
    print
    print

grid_output_path = save_prediction_grid(image_paths, mae_ages, coral_ages)
for path in grid_output_path:
    print(f"Saved combined visualization: {path}")
    
    
    
    