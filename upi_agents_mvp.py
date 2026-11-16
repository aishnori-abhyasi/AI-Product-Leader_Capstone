"""
UPI Gig-Worker Agents MVP  (LangChain + LangGraph)
==================================================
90 days of (synthetic) UPI telemetry  ->  4 sequential agents  ->  safe, plain-English offer

    gatekeeper (rules)  ->  SegmentationAgent  ->  ScoringAgent  ->  MatchingAgent  ->  CommunicationAgent
                            K-Means                Ridge logistic     policy matrix +     script_rag.py
                                                   regression         scheme retrieval    (Flesch-Kincaid loop)

Design rules that keep an "agentic" credit pipeline safe
  1. Numbers come from TOOLS, never from LLM text. The LLM decides to call its tool and writes a
     one-line explanation; the worker_id argument is forced from state so it cannot be altered.
  2. Hard rules (consent, data sufficiency, bank account, occupation) run BEFORE any model.
  3. Order is enforced in code: match_offers refuses to run until segmentation + scoring exist.
  4. Every agent step is written to an audit log (tool, output, explanation, LLM used or not).

Install:   pip install langchain-core langgraph langchain-anthropic scikit-learn pandas numpy
Run:       python upi_agents_mvp.py                 # no key needed: deterministic mode
           ANTHROPIC_API_KEY=... python upi_agents_mvp.py      # LLM agents on
           python upi_agents_mvp.py --eval 300      # batch run + governance audit
Needs script_rag.py and matching_engine.py in the same folder.
"""
import argparse, json, operator, os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matching_engine import CLUSTER_FEATURES, FEATURES, MIN_REPAY_SHARE, POLICY, REASON_TEXT
from script_rag import KB, SchemeRetriever, optimise

# LangChain is optional at import time so the logic stays testable without it.
try:
    from langchain_core.tools import tool
    HAVE_LANGCHAIN = True
except ImportError:
    HAVE_LANGCHAIN = False

    class _Tool:                                   # minimal stand-in with the same surface we use
        def __init__(self, fn): self.fn, self.name, self.description = fn, fn.__name__, fn.__doc__
        def invoke(self, args): return self.fn(**args)

    def tool(fn): return _Tool(fn)

# ═════════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC 90-DAY UPI TELEMETRY  (replace with your real extract: one row per worker-day)
# ═════════════════════════════════════════════════════════════════════════════
ARCHETYPES = {   # hidden ground truth used ONLY to generate data and to sanity-check clusters
    "steady":    dict(p=.40, p_active=.86, mean=650, cv=.45, pool=8,  ess=.55, late=.06),
    "irregular": dict(p=.35, p_active=.50, mean=380, cv=1.00, pool=3,  ess=.78, late=.30),
    "mixed":     dict(p=.25, p_active=.70, mean=480, cv=.70, pool=12, ess=.66, late=.15),
}
OCCUPATIONS = ["delivery rider", "ride-app driver", "street vendor", "home-service worker"]
BILL_AMOUNT, LOW_BALANCE = 350, 100

def generate_upi_telemetry(n_workers=1500, days=90, seed=11):
    """Columns: worker_id, day, date, inflow, payer_id (-1 = no inflow), essential_out,
    closing_balance, bill_due, bill_on_time."""
    r = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp("2026-10-31"), periods=days)
    due_days = set(range(7, days, 15))
    names, probs = list(ARCHETYPES), [a["p"] for a in ARCHETYPES.values()]
    frames, meta = [], []
    for i in range(n_workers):
        wid, arch = f"W{i:04d}", r.choice(names, p=probs)
        c = ARCHETYPES[arch]
        p_act = float(np.clip(r.normal(c["p_active"], .06), .15, .98))
        mean = float(max(150, r.normal(c["mean"], c["mean"] * .12)))
        cv = float(max(.25, r.normal(c["cv"], .08)))
        pool = int(max(2, round(r.normal(c["pool"], 1.5))))
        ess_daily = float(np.clip(r.normal(c["ess"], .06), .25, 1)) * mean * p_act
        late = float(np.clip(r.normal(c["late"], .04), 0, .6))
        active = r.random(days) < p_act
        k = 1 / cv ** 2
        inflow = np.where(active, r.gamma(k, mean / k, days), 0.0)
        payer = np.where(active, r.integers(0, pool, days), -1)
        other = r.gamma(2, 25, days)
        bal, closing, ess_paid = 500.0, np.zeros(days), np.zeros(days)
        due, ontime = np.zeros(days, bool), np.zeros(days, bool)
        for d in range(days):
            avail = bal + inflow[d]
            if d in due_days:
                due[d] = True
                ontime[d] = bool(avail >= BILL_AMOUNT and r.random() > late)
                if ontime[d]: avail -= BILL_AMOUNT
            e = min(ess_daily, avail); avail -= e; ess_paid[d] = e
            avail -= min(other[d], avail)
            avail -= 0.7 * max(0.0, avail - 300)          # sweep to own savings / cash-out
            bal = avail; closing[d] = bal
        frames.append(pd.DataFrame({"worker_id": wid, "day": np.arange(days), "date": dates,
                                    "inflow": inflow, "payer_id": payer, "essential_out": ess_paid,
                                    "closing_balance": closing, "bill_due": due, "bill_on_time": ontime}))
        meta.append({"worker_id": wid, "archetype": arch,
                     "occupation": r.choice(OCCUPATIONS, p=[.35, .30, .20, .15]),
                     "has_bank_account": bool(r.random() < .85),
                     "consent": bool(r.random() < .97), "days_of_data": days})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(meta).set_index("worker_id")

def engineer_features(tel: pd.DataFrame) -> pd.DataFrame:
    t = tel.assign(active=tel["inflow"] > 0, week=tel["day"] // 7)
    g = t.groupby("worker_id")
    weekly = t.groupby(["worker_id", "week"])["inflow"].sum().unstack()
    active_days = g["active"].sum().replace(0, np.nan)
    F = pd.DataFrame({
        "active_days_ratio": g["active"].mean(),
        "avg_daily_inflow": (g["inflow"].sum() / active_days).fillna(0),
        "inflow_cv": (weekly.std(axis=1) / weekly.mean(axis=1)).fillna(2),   # weekly volatility
        "payer_diversity": t[t["payer_id"] >= 0].groupby("worker_id")["payer_id"].nunique().reindex(g.size().index).fillna(0),
        "bill_on_time_ratio": (g["bill_on_time"].sum() / g["bill_due"].sum().replace(0, np.nan)).fillna(0),
        "low_balance_days_ratio": g["closing_balance"].apply(lambda s: (s < LOW_BALANCE).mean()),
        "essential_spend_share": (g["essential_out"].sum() / g["inflow"].sum().replace(0, np.nan)).clip(0, 1).fillna(1),
    })
    return F[FEATURES]

def synthetic_repayment_label(F: pd.DataFrame, seed=5) -> pd.Series:
    """SYNTHETIC outcome. In production this is observed repayment/default behaviour."""
    r = np.random.default_rng(seed)
    Z = (F - F.mean()) / F.std()
    w = dict(active_days_ratio=.5, avg_daily_inflow=.6, inflow_cv=-.7, payer_diversity=.1,
             bill_on_time_ratio=.6, low_balance_days_ratio=-.7, essential_spend_share=-.4)
    z = sum(w[c] * Z[c] for c in FEATURES) + r.normal(0, 1.6, len(F))     # unobserved factors
    return pd.Series((r.random(len(F)) < 1 / (1 + np.exp(-(.55 * z + .4)))).astype(int), index=F.index)

# ═════════════════════════════════════════════════════════════════════════════
# 2. THE THREE MODELS (each will sit behind its own agent)
# ═════════════════════════════════════════════════════════════════════════════
class SegmentationModel:
    """K-Means on behaviour only: no outcome label, no credit data."""
    NAMES = ["Steady daily earner", "Mixed-source earner", "Irregular earner"]   # best -> worst

    def fit(self, F, k=3):
        self.sc = StandardScaler().fit(F[CLUSTER_FEATURES])
        Z = self.sc.transform(F[CLUSTER_FEATURES])
        self.km = KMeans(k, n_init=10, random_state=1).fit(Z)
        cols = {c: i for i, c in enumerate(CLUSTER_FEATURES)}
        stability = [Z[self.km.labels_ == c][:, cols["active_days_ratio"]].mean()
                     - Z[self.km.labels_ == c][:, cols["inflow_cv"]].mean() for c in range(k)]
        self.label = {c: self.NAMES[rank] for rank, c in enumerate(np.argsort(stability)[::-1])}
        return self

    def predict_many(self, F):
        Z = self.sc.transform(F[CLUSTER_FEATURES])
        return pd.Series([self.label[c] for c in self.km.predict(Z)], index=F.index)

    def predict(self, row):
        z = self.sc.transform(pd.DataFrame([row[CLUSTER_FEATURES]]))
        d = np.sort(self.km.transform(z)[0])
        c = int(self.km.predict(z)[0])
        return {"segment": self.label[c], "cluster_id": c,
                "confidence": round(float((d[1] - d[0]) / (d[1] + d[0] + 1e-9)), 2)}

class RepaymentModel:
    """Ridge (L2) logistic regression -> repayment probability -> band."""
    BANDS = [(0.45, "A"), (0.62, "B"), (0.78, "C"), (1.01, "D")]       # policy parameters

    def fit(self, F, y, C=.5):
        Ftr, self.Fte, ytr, self.yte = train_test_split(F, y, test_size=.25, stratify=y, random_state=1)
        self.sc = StandardScaler().fit(Ftr[FEATURES])
        self.lr = LogisticRegression(C=C, max_iter=1000).fit(self.sc.transform(Ftr[FEATURES]), ytr)   # L2 default
        self.pte = self.lr.predict_proba(self.sc.transform(self.Fte[FEATURES]))[:, 1]
        self.metrics = {"AUC": round(roc_auc_score(self.yte, self.pte), 3),
                        "Brier": round(brier_score_loss(self.yte, self.pte), 3)}
        return self

    def band(self, p): return next(b for cut, b in self.BANDS if p < cut)

    def predict(self, row):
        x = self.sc.transform(pd.DataFrame([row[FEATURES]]))
        p = float(self.lr.predict_proba(x)[0, 1])
        contrib = self.lr.coef_[0] * x[0]
        order = np.argsort(contrib)
        helps = [FEATURES[i] for i in order[::-1][:2] if contrib[i] > 0]
        hurts = [FEATURES[i] for i in order[:2] if contrib[i] < 0]
        return {"p_repay": round(p, 3), "band": self.band(p),
                "helps": [REASON_TEXT[f][0] for f in helps], "hurts": [REASON_TEXT[f][1] for f in hurts]}

class MatchingEngine:
    """Rules + policy matrix + scheme retrieval. Deterministic and auditable."""
    def __init__(self, retriever): self.retriever = retriever

    def match(self, meta, feat, segment, score):
        credit_ok, kinds = POLICY[segment["segment"]][score["band"]]
        if not meta["has_bank_account"]:
            credit_ok = False
            kinds = list(dict.fromkeys(["enrolment"] + [k for k in kinds if k != "loan"]))
        monthly = feat["avg_daily_inflow"] * 30 * feat["active_days_ratio"]
        cap = MIN_REPAY_SHARE * monthly * (1 - min(feat["inflow_cv"], 1) * .5) if credit_ok else 0
        offers = self.retriever.search("", score["band"], meta["has_bank_account"],
                                       occupation=meta["occupation"], k=3, kinds=kinds)
        return {"segment": segment["segment"], "band": score["band"], "tier": score["band"],
                "credit_ok": bool(credit_ok), "lead_with": kinds, "max_monthly_repayment": int(round(cap)),
                "offers": [{"id": o["id"], "name": o["name"], "kind": o["kind"], "verified": o["verified"]} for o in offers],
                "reasons": score["helps"] + (score["hurts"] if score["band"] in ("A", "B") else []),
                "occupation": meta["occupation"], "has_bank_account": bool(meta["has_bank_account"])}

# ═════════════════════════════════════════════════════════════════════════════
# 3. CONTEXT + TOOLS  (one tool per agent; tools are the only source of numbers)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Context:
    features: pd.DataFrame
    meta: pd.DataFrame
    seg: SegmentationModel
    scorer: RepaymentModel
    matcher: MatchingEngine
    retriever: SchemeRetriever
    results: dict = field(default_factory=dict)       # worker_id -> outputs of earlier agents

def build_context(n_workers=1500):
    tel, meta = generate_upi_telemetry(n_workers)
    F = engineer_features(tel)
    y = synthetic_repayment_label(F)
    retriever = SchemeRetriever(KB)
    return Context(F, meta, SegmentationModel().fit(F), RepaymentModel().fit(F, y),
                   MatchingEngine(retriever), retriever)

def build_tools(ctx: Context):
    @tool
    def segment_worker(worker_id: str) -> dict:
        """Assign a gig worker to a behavioural segment with K-Means on 90 days of UPI features.
        Input: worker_id. Returns segment name, cluster id and confidence."""
        out = ctx.seg.predict(ctx.features.loc[worker_id])
        ctx.results.setdefault(worker_id, {})["segment"] = out
        return out

    @tool
    def score_repayment(worker_id: str) -> dict:
        """Estimate repayment probability with ridge logistic regression on 90 days of UPI features.
        Input: worker_id. Returns probability, band A-D and plain-language reasons."""
        out = ctx.scorer.predict(ctx.features.loc[worker_id])
        ctx.results.setdefault(worker_id, {})["score"] = out
        return out

    @tool
    def match_offers(worker_id: str) -> dict:
        """Match the worker to allowed government schemes using rules, the segment x band policy
        matrix and scheme retrieval. Requires segment_worker and score_repayment to have run."""
        res = ctx.results.get(worker_id, {})
        if "segment" not in res or "score" not in res:
            return {"error": "segmentation and scoring must run first"}
        out = ctx.matcher.match(ctx.meta.loc[worker_id], ctx.features.loc[worker_id], res["segment"], res["score"])
        res["match"] = out
        return out

    @tool
    def write_script(worker_id: str) -> dict:
        """Write a plain-English script (target Flesch-Kincaid grade <= 6) from the matched offers,
        using only verified scheme facts. Requires match_offers to have run."""
        match = ctx.results.get(worker_id, {}).get("match")
        if not match:
            return {"error": "matching must run first"}
        r = optimise(match, ctx.retriever)
        last = [h for h in r.history if "grade" in h][-1]
        return {"script": r.script, "grade": last["grade"], "reading_ease": last["ease"],
                "grounded": r.grounded["ok"], "meaning_ok": r.meaning["ok"],
                "facts_used": [f["id"] for f in r.facts], "safe_to_ship": r.safe_to_ship}

    return {"segmentation": segment_worker, "scoring": score_repayment,
            "matching": match_offers, "communication": write_script}

# ═════════════════════════════════════════════════════════════════════════════
# 4. AGENTS = LangChain tool-calling LLM (optional) around one tool each
# ═════════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict, total=False):
    worker_id: str
    blocked: str
    segmentation: dict
    scoring: dict
    matching: dict
    communication: dict
    audit: Annotated[list, operator.add]

SYSTEM = {
    "segmentation": "You are the Segmentation Agent for a gig-worker financial-inclusion service. "
                    "Call segment_worker exactly once.",
    "scoring":      "You are the Scoring Agent. Call score_repayment exactly once.",
    "matching":     "You are the Matching Agent. Call match_offers exactly once.",
    "communication": "You are the Communication Agent. Call write_script exactly once.",
}
EXPLAIN = ("Now explain the tool result in ONE short sentence for a credit-risk reviewer. "
           "Use only values present in the tool result. Do not add numbers or promises.")

def _text(msg):
    c = getattr(msg, "content", msg)
    return c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict))

def _template(name, out):
    if "error" in out: return out["error"]
    return {"segmentation": lambda o: f"Behaviour pattern matches '{o.get('segment')}' (confidence {o.get('confidence')}).",
            "scoring":      lambda o: f"Repayment probability {o.get('p_repay')} -> band {o.get('band')}.",
            "matching":     lambda o: f"Credit allowed: {o.get('credit_ok')}; lead with {o.get('lead_with')}; offers {[x['id'] for x in o.get('offers', [])]}.",
            "communication": lambda o: f"Script at grade {o.get('grade')}; safe_to_ship={o.get('safe_to_ship')}."}[name](out)

def make_agent_node(name, tool_obj, llm=None):
    def node(state: PipelineState):
        wid = state["worker_id"]
        explanation, used_llm, call_id = None, False, None
        if llm is not None:
            msgs = [{"role": "system", "content": SYSTEM[name]},
                    {"role": "user", "content": f"Worker id: {wid}"}]
            ai = llm.bind_tools([tool_obj]).invoke(msgs)
            all_calls = list(getattr(ai, "tool_calls", []) or [])
            call_id = next((c["id"] for c in all_calls if c["name"] == tool_obj.name), None)
        result = tool_obj.invoke({"worker_id": wid})          # worker_id forced from state, not from the LLM
        if llm is not None and call_id:
            replies = [{"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, default=str)}
                       for c in all_calls]
            follow = llm.invoke(msgs + [ai] + replies + [{"role": "user", "content": EXPLAIN}])
            explanation, used_llm = _text(follow).strip(), True
        explanation = explanation or _template(name, result)
        entry = {"agent": name, "tool": tool_obj.name, "worker_id": wid, "output": result,
                 "explanation": explanation, "llm_used": used_llm,
                 "llm_called_tool": call_id is not None if llm is not None else None,
                 "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return {name: result, "audit": [entry]}
    return node

def make_gatekeeper(ctx):
    def node(state: PipelineState):
        m = ctx.meta.loc[state["worker_id"]]
        reason = ("no consent to use UPI data" if not m["consent"] else
                  "fewer than 60 days of data" if m["days_of_data"] < 60 else None)
        entry = {"agent": "gatekeeper", "tool": "rules", "worker_id": state["worker_id"],
                 "output": {"blocked": reason}, "explanation": reason or "consent and data checks passed",
                 "llm_used": False, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return ({"blocked": reason, "audit": [entry]} if reason else {"audit": [entry]})
    return node

# ═════════════════════════════════════════════════════════════════════════════
# 5. ORCHESTRATION: LangGraph if installed, else the same nodes in a plain sequence
# ═════════════════════════════════════════════════════════════════════════════
ORDER = ["segmentation", "scoring", "matching", "communication"]

class SequentialRunner:
    def __init__(self, nodes): self.nodes = nodes
    def invoke(self, state):
        state = {**state, "audit": list(state.get("audit", []))}
        for name in ["gatekeeper"] + ORDER:
            upd = self.nodes[name](state)
            for k, v in upd.items():
                state[k] = state["audit"] + v if k == "audit" else v
            if state.get("blocked"): break
        return state

def build_pipeline(ctx, llm=None, force_sequential=False):
    tools = build_tools(ctx)
    nodes = {"gatekeeper": make_gatekeeper(ctx)}
    nodes.update({n: make_agent_node(n, tools[n], llm) for n in ORDER})
    if force_sequential:
        return SequentialRunner(nodes), "sequential-fallback"
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return SequentialRunner(nodes), "sequential-fallback (langgraph not installed)"
    g = StateGraph(PipelineState)
    gname = lambda n: n if n == "gatekeeper" else n + "_agent"   # node names must differ from state keys
    for n, fn in nodes.items(): g.add_node(gname(n), fn)
    g.set_entry_point("gatekeeper")
    g.add_conditional_edges("gatekeeper", lambda s: "stop" if s.get("blocked") else "go",
                            {"go": gname(ORDER[0]), "stop": END})
    for a, b in zip(ORDER, ORDER[1:]): g.add_edge(gname(a), gname(b))
    g.add_edge(gname(ORDER[-1]), END)
    return g.compile(), "langgraph"

def make_llm():
    if not os.getenv("ANTHROPIC_API_KEY"): return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        print("[langchain-anthropic not installed -> deterministic mode]"); return None
    return ChatAnthropic(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=400)

# ═════════════════════════════════════════════════════════════════════════════
# 6. GOVERNANCE AUDIT + CLI
# ═════════════════════════════════════════════════════════════════════════════
def model_audit(ctx):
    sc = ctx.scorer
    seg = ctx.seg.predict_many(sc.Fte)
    t = pd.DataFrame({"p": sc.pte, "y": sc.yte.values, "segment": seg.values})
    t["band"] = [sc.band(p) for p in t["p"]]
    t["credit"] = [POLICY[s][b][0] for s, b in zip(t["segment"], t["band"])]
    print("\nAUDIT 1 - AUC within each segment (overall AUC can hide weak groups)")
    for s, g in t.groupby("segment"):
        auc = roc_auc_score(g["y"], g["p"]) if g["y"].nunique() > 1 else float("nan")
        print(f"  {s:<22} n={len(g):>4}  AUC={auc:.3f}  actual repay={g['y'].mean():.2f}")
    print("AUDIT 2 - calibration by band (predicted vs actual)")
    for b, g in t.groupby("band"):
        print(f"  band {b}: n={len(g):>4}  predicted={g['p'].mean():.2f}  actual={g['y'].mean():.2f}")
    print("AUDIT 3 - share allowed credit messaging, by segment")
    print("  " + t.groupby("segment")["credit"].mean().round(2).to_string().replace("\n", "\n  "))

def show(state):
    a = {e["agent"]: e for e in state["audit"]}
    print(f"\n{'=' * 78}\nworker {state['worker_id']}  |  steps: {[e['agent'] for e in state['audit']]}")
    if state.get("blocked"):
        print("BLOCKED:", state["blocked"]); return
    for n in ORDER:
        if n in a: print(f"  [{n:<13}] {a[n]['explanation']}")
    m, c = state["matching"], state["communication"]
    print(f"  occupation={m['occupation']} bank={m['has_bank_account']} credit_ok={m['credit_ok']} "
          f"max_repay={m['max_monthly_repayment']}  offers={[o['id'] for o in m['offers']]}")
    print(f"  SCRIPT (grade {c['grade']}): {c['script']}")
    print(f"  safe_to_ship={c['safe_to_ship']}  grounded={c['grounded']}  meaning_ok={c['meaning_ok']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help="run a single worker id, e.g. W0007")
    ap.add_argument("--eval", type=int, default=0, help="batch-run N workers (deterministic) + summary")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--workers", type=int, default=1500, help="size of synthetic cohort")
    ap.add_argument("--audit-file", default="agent_audit_log.json")
    args = ap.parse_args()

    ctx = build_context(args.workers)
    llm = None if args.no_llm else make_llm()
    app, mode = build_pipeline(ctx, llm)
    print(f"orchestrator: {mode} | agents use LLM: {llm is not None} | langchain installed: {HAVE_LANGCHAIN}")
    print(f"models: Ridge-LR {ctx.scorer.metrics} | K-Means labels {ctx.seg.label}")
    xt = pd.crosstab(ctx.seg.predict_many(ctx.features), ctx.meta["archetype"])
    print("K-Means segment vs hidden archetype (sanity check on synthetic data):\n" + xt.to_string())

    if args.worker: ids = [args.worker]
    else:   # one worker per interesting case
        m = ctx.meta
        pick = lambda q: m.query(q).index[0]
        ids = [pick("archetype=='steady' and has_bank_account and consent"),
               pick("archetype=='irregular' and not has_bank_account and consent"),
               pick("occupation=='street vendor' and has_bank_account and consent"),
               pick("not consent")]
    log = []
    for wid in ids:
        out = app.invoke({"worker_id": wid, "audit": []})
        show(out); log += out["audit"]
    with open(args.audit_file, "w") as f: json.dump(log, f, indent=2, default=str)
    print(f"\naudit log -> {args.audit_file} ({len(log)} agent steps)")

    if args.eval:
        plain, _ = build_pipeline(ctx, None)
        rows = []
        for wid in ctx.meta.index[:args.eval]:
            s = plain.invoke({"worker_id": wid, "audit": []})
            if s.get("blocked"): rows.append({"segment": "blocked", "band": "-", "credit_ok": False, "safe": False}); continue
            rows.append({"segment": s["matching"]["segment"], "band": s["matching"]["band"],
                         "credit_ok": s["matching"]["credit_ok"], "safe": s["communication"]["safe_to_ship"]})
        d = pd.DataFrame(rows)
        print(f"\nBATCH of {len(d)} workers\nsegment x band:\n{pd.crosstab(d['segment'], d['band']).to_string()}")
        print(f"credit_ok share by segment:\n{d.groupby('segment')['credit_ok'].mean().round(2).to_string()}")
        print(f"scripts safe_to_ship: {d['safe'].mean():.0%}  (low until scheme facts are verified)")
        model_audit(ctx)

if __name__ == "__main__":
    main()
