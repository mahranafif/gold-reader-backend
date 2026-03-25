from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model(model_name: str, num_classes: int):
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


class ImageClassifier:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        self.model_name = checkpoint["model_name"]
        self.image_size = int(checkpoint["image_size"])
        self.class_to_idx = checkpoint["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.model = _build_model(self.model_name, len(self.class_to_idx))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

        return {
            "label": self.idx_to_class[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "all_probs": {
                self.idx_to_class[i]: float(probs[i].item())
                for i in range(len(probs))
            },
        }


class GoldPosterClassifier(ImageClassifier):
    pass


class GoldLayoutClassifier(ImageClassifier):
    pass
