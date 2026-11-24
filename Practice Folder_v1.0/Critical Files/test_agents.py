from types import SimpleNamespace as NS
import upi_agents_mvp as M

ctx = M.build_context(300)
wid = ctx.meta.query("consent and has_bank_account").index[0]

class Fake:
    """Mimics ChatAnthropic: bind_tools(...).invoke(msgs) -> tool call; invoke(msgs) -> text."""
    def __init__(self, mode): self.mode, self.seen = mode, []
    def bind_tools(self, tools):
        outer = self
        class B:
            def invoke(self, msgs):
                name = tools[0].name
                if outer.mode == "skip":   return NS(tool_calls=[], content="I'll just answer.")
                calls = [{"name": name, "args": {"worker_id": "BOGUS"}, "id": "c1"}]
                if outer.mode == "double": calls.append({"name": name, "args": {"worker_id": "BOGUS"}, "id": "c2"})
                return NS(tool_calls=calls, content="")
        return B()
    def invoke(self, msgs):
        self.seen.append(msgs)
        return NS(content=[{"type": "text", "text": "FAKE EXPLANATION"}])    # list-of-blocks content

# 1. hostile id + normal flow
for mode in ["normal", "skip", "double"]:
    ctx.results.clear()
    app, _ = M.build_pipeline(ctx, Fake(mode), force_sequential=True)
    out = app.invoke({"worker_id": wid, "audit": []})
    steps = {e["agent"]: e for e in out["audit"]}
    seg = steps["segmentation"]
    print(f"[{mode:6}] llm_used={seg['llm_used']} called_tool={seg['llm_called_tool']} "
          f"explanation={seg['explanation'][:42]!r} worker={seg['worker_id']} segment={out['segmentation']['segment']}")
    assert seg["worker_id"] == wid and "BOGUS" not in str(out)
# 2. double call -> every tool_use answered
f = Fake("double"); ctx.results.clear()
app, _ = M.build_pipeline(ctx, f, force_sequential=True); app.invoke({"worker_id": wid, "audit": []})
tool_replies = [m for m in f.seen[0] if isinstance(m, dict) and m.get("role") == "tool"]
print("double: tool replies sent for first agent =", len(tool_replies), "(must be 2)")
assert len(tool_replies) == 2

# 3. order is enforced in code
ctx.results.clear(); tools = M.build_tools(ctx)
print("match before models  ->", tools["matching"].invoke({"worker_id": wid}))
print("script before match  ->", tools["communication"].invoke({"worker_id": wid}))

# 4. gate stops the pipeline
nc = ctx.meta.query("not consent").index[0]
app, _ = M.build_pipeline(ctx, None, force_sequential=True)
o = app.invoke({"worker_id": nc, "audit": []}); print("no consent ->", o["blocked"], "| agents run:", [e["agent"] for e in o["audit"]])
print("ALL TESTS PASSED")
