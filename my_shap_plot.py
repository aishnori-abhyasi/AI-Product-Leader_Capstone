import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 1. Example model & data (replace with your own trained model and X_test DataFrame)
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
feature_names = ["income", "upi_tx_count", "avg_spend", "bill_delay", "tenure"]
model = RandomForestClassifier(random_state=42).fit(X, y)

# 2. Initialize the SHAP Explainer and compute values
explainer = shap.Explainer(model, X, feature_names=feature_names)
shap_values = explainer(X)

# For binary classification, slice [:, :, 1] to explain the positive class (Class 1)
shap_values_class1 = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

# 3. Print text-based feature importance in the VS Code terminal
print("Top features for sample 0:")
for name, val in zip(feature_names, shap_values_class1[0].values):
    print(f"  {name}: {val:+.4f}")

# 4. Save visual plots into your script's folder so you can click & view them in VS Code
out_dir = Path(__file__).resolve().parent

# Summary Beeswarm Plot (Global feature importance)
plt.figure()
shap.plots.beeswarm(shap_values_class1, show=False)
plt.savefig(out_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()

# Waterfall Plot (Local explanation for a single prediction/worker)
plt.figure()
shap.plots.waterfall(shap_values_class1[0], show=False)
plt.savefig(out_dir / "shap_waterfall_sample0.png", bbox_inches="tight", dpi=150)
plt.close()

print("Saved shap_summary.png and shap_waterfall_sample0.png!")
