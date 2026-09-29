import os, sys
from types import SimpleNamespace as NS
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from sklearn.datasets import load_wine
from tools import Ctx, call_tool, train_model, auto_analysis
from agent import run_agent

DF = load_wine(as_frame=True).frame
DF["cultivar"] = DF.pop("target").map({0: "A", 1: "B", 2: "C"})


def ctx():
    return Ctx(DF)


class FakeClient:
    """Scripted LLM so the agent loop is tested without network or API key."""
    def __init__(self, script, loop=False):
        self.script, self.loop, self.messages = list(script), loop, self

    def create(self, **kw):
        return self.script[0] if self.loop else self.script.pop(0)


def tool_use(name, args, i="t1"):
    return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=i, name=name, input=args)])


def final(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


def test_train_model_beats_baseline():
    r = train_model(ctx(), "cultivar")
    assert r["task"] == "classification" and r["cv_scores"][r["best_model"]] > r["cv_scores"]["baseline"] + 0.3


def test_aggregate_grouped_and_filtered():
    r = call_tool("aggregate", {"column": "alcohol", "group_by": "cultivar"}, ctx())
    assert set(r["result"]) == {"A", "B", "C"}
    r = call_tool("aggregate", {"column": "alcohol", "filters": [{"column": "alcohol", "op": ">", "value": 14}]}, ctx())
    assert r["rows_used"] < len(DF)


def test_bad_inputs_return_errors_not_crashes():
    assert "error" in call_tool("aggregate", {"column": "nope"}, ctx())
    assert "error" in call_tool("aggregate", {"column": "alcohol", "filters": [{"column": "alcohol", "op": "__import__", "value": 1}]}, ctx())
    assert "error" in call_tool("os_system", {}, ctx())


def test_plot_creates_figure():
    c = ctx(); call_tool("plot", {"kind": "hist", "x": "alcohol"}, c); assert len(c.figs) == 1


def test_agent_loop_uses_tool_then_answers():
    c = ctx()
    client = FakeClient([tool_use("describe_data", {}), final("Dataset has 178 rows.")])
    text, trace = run_agent("Describe it", c, client)
    assert "178" in text and trace[0]["tool"] == "describe_data" and trace[0]["output"]["rows"] == 178


def test_agent_step_limit():
    text, trace = run_agent("loop", ctx(), FakeClient([tool_use("describe_data", {})], loop=True), max_steps=3)
    assert "step limit" in text and len(trace) == 3


def test_auto_analysis_runs():
    r = auto_analysis(ctx(), "cultivar")
    assert r["outliers"]["n_outliers"] >= 1 and r["correlations"]["top_pairs"]
