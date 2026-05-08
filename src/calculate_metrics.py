import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


def evaluate_model(model, dataloader, device):
    """
    Evaluates model on a DataLoader. Reports metrics at 0.5 threshold
    and finds the optimal threshold using Youden's J statistic.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    print("Running Evaluation...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # --- Default 0.5 threshold report ---
    print("\n--- PERFORMANCE REPORT (Threshold 0.5) ---")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
    plt.title('Nodule Classification Confusion Matrix'); plt.ylabel('Ground Truth'); plt.xlabel('Prediction')
    plt.savefig('checkpoints/confusion_matrix.png')
    print("Saved: checkpoints/confusion_matrix.png")

    # --- AUC-ROC & Optimal Threshold ---
    if len(np.unique(all_labels)) > 1:
        from sklearn.metrics import roc_curve
        auc = roc_auc_score(all_labels, all_probs)
        print(f"\nAUC-ROC: {auc:.4f}")

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

        # Youden's J — optimal threshold selection
        # [Full threshold calibration logic omitted from showcase]
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
        print(f"Optimal Threshold (Youden's J): {optimal_threshold:.4f}")

        # ROC plot
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.scatter([fpr[best_idx]], [tpr[best_idx]], marker='o', color='red', label='Optimal')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve'); plt.legend(loc="lower right")
        plt.savefig('checkpoints/roc_curve.png')
        print("Saved: checkpoints/roc_curve.png")

        return cm, all_probs, optimal_threshold
    else:
        print("AUC-ROC: N/A (need both classes in test set)")
        return cm, all_probs, 0.5
