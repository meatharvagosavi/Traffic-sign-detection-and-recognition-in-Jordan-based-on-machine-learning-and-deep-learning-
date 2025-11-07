import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import joblib

def get_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def load_feature_extractor(device):
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    feature_extractor = torch.nn.Sequential(*modules)
    feature_extractor.to(device).eval()
    return feature_extractor

def infer(args):
    clf = joblib.load(args.model)
    class_names = open(os.path.join(os.path.dirname(args.model), "../features/class_names.txt")).read().splitlines()
    device = torch.device(args.resnet_device)
    feat_extractor = load_feature_extractor(device)

    img = Image.open(args.image).convert('RGB')
    tr = get_transform()
    x = tr(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = feat_extractor(x)
        feats = feats.view(feats.size(0), -1).cpu().numpy()  # shape (1, 2048)
    pred = clf.predict(feats)[0]
    probs = clf.predict_proba(feats)[0]
    print("Predicted:", pred, class_names[pred])
    topk = np.argsort(probs)[::-1][:3]
    for k in topk:
        print(f"{class_names[k]}: {probs[k]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resnet_device", default="cpu")
    args = parser.parse_args()
    infer(args)
