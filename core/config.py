import torch

# COMMON
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
INSIGHTFACE_MODEL_NAME = "buffalo_sc"
INSIGHTFACE_MODULES = ["detection"]
# MAE
BACKBONE_MAE = "resnet18"
DROPOUT_MAE = 0.3
HIDDEN_DIM_MAE = 512
CHECKPOINT_PATH_MAE = "./pytorch_models/best_age_model_mae.pth"


# FAIRFACE
CHECKPOINT_PATH_FAIRFACE = "./pytorch_models/best_age_model_fairface.pth"
