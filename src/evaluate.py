import os
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def main(args):
    X_test = np.load(os.path.join(args.feats_dir, "test_feats.npy"))
    y_test = np.load(os.path.join(args.feats_dir, "test_labels.npy"))
    class_names = open(os.path.join(args.feats_dir, "class_names.txt")).read().strip().splitlines()
    clf = joblib.load(args.model)

    y_pred = clf.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    out_dir = os.path.dirname(args.model)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)
    print(f"Saved predictions and confusion matrix to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)
