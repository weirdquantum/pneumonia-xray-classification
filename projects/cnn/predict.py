"""Offline prediction using the final baseline checkpoint."""
import argparse
import json
from PIL import Image
import torch
from dataset import CLASS_TO_IDX, make_transform
from models import build_model


def predict(checkpoint_path, image_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['config']['class_to_idx'] != CLASS_TO_IDX:
        raise ValueError('Unexpected checkpoint label mapping')
    model = build_model(checkpoint['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with Image.open(image_path) as image:
        tensor = make_transform()(image.convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        probability = model(tensor).softmax(dim=1)[0, 1].item()
    return {'label': 'PNEUMONIA' if probability >= 0.5 else 'NORMAL',
            'pneumonia_probability': probability, 'threshold': 0.5}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image), indent=2))
