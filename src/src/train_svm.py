
import os
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main(args):
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    X_train = np.load(os.path.join(args.feats_dir, "train_feats.npy"))
    y_train = np.load(os.path.join(args.feats_dir, "train_labels.npy"))
    X_val = np.load(os.path.join(args.feats_dir, "val_feats.npy"))
    y_val = np.load(os.path.join(args.feats_dir, "val_labels.npy"))
    print("Train shape:", X_train.shape, y_train.shape)
    print("Val shape:", X_val.shape, y_val.shape)

    print("Training linear SVM (may take a while)...")
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, class_weight=None))
    clf.fit(X_train, y_train)
  
    y_pred = clf.predict(X_val)
    print("Validation results:")
    print(classification_report(y_val, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    joblib.dump(clf, args.out_model)
    print(f"Saved SVM pipeline to {args.out_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_dir", type=str, required=True)
    parser.add_argument("--out_model", type=str, required=True)
    args = parser.parse_args()
    main(args)
