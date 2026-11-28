"""
Explainability for the Ridge logistic-regression scorer.

Why this is SHAP (and why it is exact): for a linear model with a background distribution, the SHAP value of
feature i is  phi_i = w_i * (x_i - E[x_i])  (independent-features / interventional form).  The model is
standardised on the training set, so in z-units phi_i = coef_i * z_i, and
    base_value + sum(phi_i) = log-odds of repayment    (additivity, checked in tests).
If the `shap` package is installed, crosscheck() compares our values with shap.LinearExplainer.
Caveat: attributions explain the MODEL, not causation, and assume features are independent;
correlated features (e.g. active days vs. income) share credit arbitrarily.

Customer-facing output uses only plain words (REASON_TEXT) and an actionable "how to improve" (recourse).
Staff-facing output exposes raw SHAP values.
"""
import numpy as np
import pandas as pd

from matching_engine import FEATURES, REASON_TEXT

# Recourse: only features a person can reasonably change, with their best achievable value
ACTIONABLE = {"bill_on_time_ratio": ("pay every bill on time", 1.0),
              "low_balance_days_ratio": ("keep some money in your account so it rarely drops to zero", 0.0)}

def _logit(p): return float(np.log(p / (1 - p)))

class LinearShap:
    def __init__(self, scorer):
        self.scorer, self.sc, self.lr = scorer, scorer.sc, scorer.lr
        self.base = float(self.lr.intercept_[0])                # log-odds at the average person (z = 0)

    def values(self, row):
        z = self.sc.transform(pd.DataFrame([row[FEATURES]]))[0]
        return dict(zip(FEATURES, (self.lr.coef_[0] * z).astype(float)))

    def logit(self, row): return self.base + sum(self.values(row).values())

    def plain_reasons(self, row, k=3):
        """Top-k factors by size, worded for the customer; sign decides the wording."""
        phi = self.values(row)
        top = sorted(phi.items(), key=lambda kv: -abs(kv[1]))[:k]
        return [{"feature": f, "text": REASON_TEXT[f][0 if v > 0 else 1], "effect": "helps" if v > 0 else "holds back"} for f, v in top]

    def recourse(self, row, p, cuts):
        """What change in an actionable feature would lift the person to the next band?"""
        nxt = next((c for c, _ in cuts if c > p), None)
        if nxt is None or nxt >= 1: return []
        need = _logit(min(nxt + 1e-4, .999)) - _logit(p)
        out = []
        for f, (label, best) in ACTIONABLE.items():
            i = FEATURES.index(f)
            w = self.lr.coef_[0][i] / self.sc.scale_[i]                     # log-odds per raw unit
            gain = w * (best - row[f])
            if gain > 0.05:
                out.append({"feature": f, "enough": bool(gain >= need),
                            "text": (f"If you {label}, your level may go up" if gain >= need else
                                     f"If you {label}, it will help, but may not be enough on its own")})
        return out

    def global_importance(self, F=None):
        F = self.scorer.Fte if F is None else F
        Z = self.sc.transform(F[FEATURES]) * self.lr.coef_[0]
        return dict(sorted(zip(FEATURES, np.abs(Z).mean(axis=0).round(4).tolist()), key=lambda kv: -kv[1]))

    def crosscheck(self, F=None):
        """Compare against the shap library when available. Returns max abs difference or None."""
        try:
            import shap
        except ImportError:
            return None
        F = self.scorer.Fte.head(200) if F is None else F
        X, bg = self.sc.transform(F[FEATURES]), self.sc.transform(self.scorer.Fte[FEATURES])
        sv = shap.LinearExplainer(self.lr, bg).shap_values(X)
        ours = X * self.lr.coef_[0]
        return float(np.abs(np.asarray(sv) - ours).max())
