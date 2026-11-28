"""
Fairness & bias audit for the credit-eligibility decision (pipeline output `credit_ok`).

How this is meant to be used as EVIDENCE (not a one-off):
  1. Sensitive attributes (gender, region, age band, language) are NEVER model inputs. They are collected
     separately - voluntarily, with their own consent and purpose, or from an audit panel - and used only here.
  2. Run on HELD-OUT data at every retrain and on a schedule in production; archive each report.
  3. Metrics (Fairlearn MetricFrame semantics):
       selection rate by group; demographic-parity difference & ratio ("four-fifths" screen);
       TPR/FPR gaps -> equalized-odds / equal-opportunity difference; AUC gap; calibration gap by group.
  4. Proxy check: which model inputs correlate with group membership (the usual route to indirect bias).
  5. A failed screen means REVIEW (explain, justify business necessity, mitigate or document) - not auto-fail.
Uses `fairlearn` when installed (and cross-checks it against the native code); otherwise native metrics.

!! In this repo the sensitive attributes and the group differences are SYNTHETIC, built so the test has
   something to find. It proves the machinery works. It is NOT evidence that any real model is fair.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from matching_engine import FEATURES, POLICY

MODEL_VERSION = "ridge-lr+kmeans-0.1.0-synthetic"
MIN_N = 30                                                     # smaller groups are reported but not judged
THRESH = {"dp_ratio_min": 0.80, "eo_diff_max": 0.10, "auc_gap_max": 0.05, "calibration_gap_max": 0.05}

try:
    import fairlearn.metrics as flm
    HAVE_FAIRLEARN = True
except ImportError:
    HAVE_FAIRLEARN = False

def assign_sensitive_attributes(meta: pd.DataFrame, seed=3) -> pd.DataFrame:
    """SYNTHETIC scenario: structural disadvantage correlates with earner archetype (e.g. caregiving hours,
    rural customer base). Real audits use real, consented attribute data."""
    r = np.random.default_rng(seed)
    p_f = meta["archetype"].map({"steady": .30, "mixed": .42, "irregular": .60}).values
    p_rural = meta["archetype"].map({"steady": .25, "mixed": .35, "irregular": .55}).values
    out = pd.DataFrame(index=meta.index)
    out["gender"] = np.where(r.random(len(meta)) < p_f, "female", "male")
    out["region"] = np.where(r.random(len(meta)) < p_rural, "rural", "urban")
    out["age_band"] = r.choice(["18-25", "26-40", "41+"], len(meta), p=[.30, .50, .20])
    out["language"] = r.choice(["Hindi", "Tamil", "Telugu", "English"], len(meta), p=[.4, .25, .2, .15])
    out["gender_x_region"] = out["gender"] + "/" + out["region"]       # intersectional view
    return out

def group_table(y, pred, score, groups):
    rows = {}
    for g in sorted(pd.unique(groups)):
        m = groups == g
        yt, pr, sc = y[m], pred[m], score[m]
        rows[g] = {"n": int(m.sum()),
                   "selection_rate": float(pr.mean()),
                   "tpr": float(pr[yt == 1].mean()) if (yt == 1).any() else None,
                   "fpr": float(pr[yt == 0].mean()) if (yt == 0).any() else None,
                   "auc": float(roc_auc_score(yt, sc)) if len(set(yt)) > 1 else None,
                   "mean_score": float(sc.mean()), "actual_repay": float(yt.mean()),
                   "calibration_gap": float(abs(sc.mean() - yt.mean())),
                   "calibration_se": float(np.sqrt(max(yt.mean() * (1 - yt.mean()), 1e-4) / m.sum()))}
    return pd.DataFrame(rows).T

def _rng(s): s = pd.Series(s).dropna().astype(float); return float(s.max() - s.min()) if len(s) else 0.0

def summarise(tab, pv=None):
    """pv: optional permutation-test p-values {metric: p}. A threshold breach is a CLEAR breach only if the gap is
    also larger than chance (p < 0.05, from shuffling group labels); otherwise it is 'possible' (inconclusive)."""
    v = tab[tab["n"] >= MIN_N]
    sel = v["selection_rate"].astype(float)
    s = {"groups_judged": list(v.index), "groups_too_small": list(tab.index.difference(v.index)),
         "demographic_parity_difference": _rng(sel),
         "demographic_parity_ratio": float(sel.min() / sel.max()) if len(sel) and sel.max() > 0 else 1.0,
         "equal_opportunity_difference": _rng(v["tpr"]),
         "equalized_odds_difference": max(_rng(v["tpr"]), _rng(v["fpr"])),
         "auc_gap": _rng(v["auc"]), "calibration_gap_max": float(v["calibration_gap"].astype(float).max()) if len(v) else 0.0}
    sig = v[(v["calibration_gap"].astype(float) > THRESH["calibration_gap_max"]) &
            (v["calibration_gap"].astype(float) > 2 * v["calibration_se"].astype(float))]      # ignore pure noise
    s["calibration_flagged_groups"] = list(sig.index)
    clear, possible = [], []
    def judge(label, breached, certain):
        if breached: (clear if certain is not False else possible).append(label)
    pv = pv or {}
    sure = lambda k: None if k not in pv else pv[k] < 0.05
    judge("selection-rate ratio below 0.80", s["demographic_parity_ratio"] < THRESH["dp_ratio_min"], sure("dp_ratio"))
    judge("equalized-odds gap above 0.10", s["equalized_odds_difference"] > THRESH["eo_diff_max"], sure("eo_diff"))
    judge("AUC gap above 0.05", s["auc_gap"] > THRESH["auc_gap_max"], sure("auc_gap"))
    if len(sig): clear.append(f"calibration gap above 0.05 and statistically significant for {list(sig.index)}")
    s["flags"], s["possible_flags"], s["p_values"] = clear, possible, {k: round(x, 3) for k, x in pv.items()}
    s["verdict"] = ("INSUFFICIENT DATA" if len(v) < 2 else "REVIEW REQUIRED" if clear else
                    "INCONCLUSIVE (collect more data)" if possible else "PASS")
    return s

def permutation_pvalues(y, pred, score, groups, B=300, seed=0):
    """How often does shuffling group labels (i.e. NO real group effect) give a gap at least this large?"""
    obs = summarise(group_table(y, pred, score, groups))
    r, c = np.random.default_rng(seed), {"dp_ratio": 0, "eo_diff": 0, "auc_gap": 0}
    for _ in range(B):
        s = summarise(group_table(y, pred, score, r.permutation(groups)))
        c["dp_ratio"] += s["demographic_parity_ratio"] <= obs["demographic_parity_ratio"]
        c["eo_diff"] += s["equalized_odds_difference"] >= obs["equalized_odds_difference"]
        c["auc_gap"] += s["auc_gap"] >= obs["auc_gap"]
    return {k: (v + 1) / (B + 1) for k, v in c.items()}

def fairlearn_crosscheck(y, pred, groups, tab):
    if not HAVE_FAIRLEARN: return {"available": False}
    ok = tab[tab["n"] >= MIN_N].index
    if len(ok) < 2: 
        return {"available": True, "skipped": "insufficient data"}
    m = pd.Series(groups).isin(ok).values
    if not m.any(): 
        return {"available": True, "skipped": "no groups with enough data"}
    f = {"dp_diff": flm.demographic_parity_difference(y[m], pred[m], sensitive_features=groups[m]),
         "dp_ratio": flm.demographic_parity_ratio(y[m], pred[m], sensitive_features=groups[m]),
         "eo_diff": flm.equalized_odds_difference(y[m], pred[m], sensitive_features=groups[m])}
    s = summarise(tab)
    ours = {"dp_diff": s["demographic_parity_difference"], "dp_ratio": s["demographic_parity_ratio"], "eo_diff": s["equalized_odds_difference"]}
    return {"available": True, "max_abs_diff_vs_native": float(max(abs(f[k] - ours[k]) for k in f))}

def proxy_candidates(F, sens, top=5, min_abs=0.15):
    out = []
    for attr in ("gender", "region", "age_band"):
        for level in sorted(sens[attr].unique())[:3]:
            ind = (sens[attr] == level).astype(float)
            for f in FEATURES:
                r = float(np.corrcoef(F[f].values, ind.loc[F.index].values)[0, 1])
                if abs(r) >= min_abs: out.append({"attribute": f"{attr}={level}", "feature": f, "corr": round(r, 3)})
    return sorted(out, key=lambda d: -abs(d["corr"]))[:top]

def run_audit(ctx, seed=3):
    sc = ctx.scorer
    F, p, y = sc.Fte, sc.pte, sc.yte.values
    seg = ctx.seg.predict_many(F)
    pred = np.array([POLICY[s][sc.band(pp)][0] for s, pp in zip(seg, p)], dtype=int)
    sens = assign_sensitive_attributes(ctx.meta.loc[F.index], seed)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model_version": MODEL_VERSION,
              "engine": "fairlearn+native" if HAVE_FAIRLEARN else "native (fairlearn not installed)",
              "decision_tested": "credit_ok (may we mention credit?)", "n_test": int(len(F)),
              "thresholds": THRESH, "method_note": "breach = threshold crossed AND gap larger than chance (permutation p<0.05); 5 attributes x 3 metrics, no multiplicity correction", "attributes": {}, "data_note": "SYNTHETIC data and synthetic sensitive attributes"}
    for attr in sens.columns:
        groups = sens[attr].values
        tab = group_table(y, pred, p, groups)
        pv = permutation_pvalues(y, pred, p, groups) if (tab["n"] >= MIN_N).sum() >= 2 else None
        report["attributes"][attr] = {"table": json.loads(tab.to_json(orient="index")), "summary": summarise(tab, pv),
                                      "fairlearn_crosscheck": fairlearn_crosscheck(y, pred, groups, tab)}
    vs = [a["summary"]["verdict"] for a in report["attributes"].values()]
    report["overall_verdict"] = ("REVIEW REQUIRED" if "REVIEW REQUIRED" in vs else
                                 "INCOMPLETE (inconclusive or insufficient data for some attributes)"
                                 if any(x.startswith(("INCONCLUSIVE", "INSUFFICIENT")) for x in vs) else "PASS")
    report["proxy_candidates"] = proxy_candidates(F, sens)
    report["mitigation_playbook"] = [
        "Check proxy_candidates: can a feature be dropped, capped or replaced without losing legitimate signal?",
        "Prefer policy fixes over score tinkering: e.g. a graduated, savings-linked product path for irregular earners.",
        "Try in-processing mitigation with fairlearn.reductions.ExponentiatedGradient (constraint: EqualizedOdds / DemographicParity); "
        "never use the sensitive attribute as a model input or per-group threshold at decision time (disparate treatment).",
        "Re-run this audit after any change; archive before/after reports; record business-necessity reasoning.",
        "Add human review for borderline cases and a visible contest route (already in the app).",
    ]
    return report

def model_card(ctx, report, importance):
    return {"model": MODEL_VERSION, "intended_use": "Suggest government schemes / savings / insurance to gig workers; "
            "credit mentions are limited to partner-lender referral with separate consent.",
            "out_of_scope": ["automated loan approval or rejection", "pricing", "collections", "any use of data beyond the stated purpose"],
            "training_data": "SYNTHETIC 90-day UPI-derived features (1,500 workers). Replace before any real use.",
            "performance": ctx.scorer.metrics, "global_feature_importance_mean_abs_shap_logodds": importance,
            "fairness_overall": report["overall_verdict"], "fairness_report_generated": report["generated_at"],
            "human_oversight": "Contest route; staff review queue; no fully automated adverse outcome.",
            "limitations": ["synthetic data", "linear model assumes independent features", "no real-world outcome labels",
                            "sensitive attributes unavailable for real users unless collected with separate consent"],
            "sign_off": {"model_owner": None, "independent_validator": None, "compliance": None, "date": None}}
