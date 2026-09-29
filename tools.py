"""Safe, structured analysis tools an LLM agent can call. No arbitrary code execution."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0B0F1A", "axes.facecolor": "#131A2A", "savefig.facecolor": "#0B0F1A",
    "axes.edgecolor": "#2A3450", "axes.labelcolor": "#C9D1E3", "text.color": "#E6EAF2",
    "xtick.color": "#93A0B8", "ytick.color": "#93A0B8", "axes.titlecolor": "#E6EAF2",
    "axes.prop_cycle": matplotlib.cycler(color=["#7C5CFF", "#22D3EE", "#F472B6", "#FBBF24"]),
    "axes.grid": True, "grid.color": "#232C42", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
})
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_ROWS = 50_000


class Ctx:
    """Holds the dataset and generated figures for one session."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.head(MAX_ROWS).copy()
        self.figs = []


def _col(ctx, c):
    if c not in ctx.df.columns:
        raise ValueError(f"Unknown column '{c}'. Available: {list(ctx.df.columns)[:40]}")
    return c


OPS = {
    "==": lambda s, v: s == v, "!=": lambda s, v: s != v, ">": lambda s, v: s > v,
    "<": lambda s, v: s < v, ">=": lambda s, v: s >= v, "<=": lambda s, v: s <= v,
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, regex=False),
}


def _filter(ctx, filters):
    df = ctx.df
    for f in filters or []:
        c, op, v = _col(ctx, f["column"]), f["op"], f["value"]
        if op not in OPS:
            raise ValueError(f"Unsupported operator '{op}'. Use one of {list(OPS)}")
        if pd.api.types.is_numeric_dtype(df[c]) and op != "contains":
            v = float(v)
        df = df[OPS[op](df[c], v)]
    return df


def describe_data(ctx):
    df = ctx.df
    return {
        "rows": len(df),
        "columns": [{"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
                     "unique": int(df[c].nunique())} for c in df.columns],
        "numeric_summary": df.describe().round(3).to_dict(),
    }


def aggregate(ctx, column, agg="mean", group_by=None, filters=None, top_n=10):
    if agg not in ("mean", "sum", "median", "min", "max", "count", "std"):
        raise ValueError("agg must be one of mean,sum,median,min,max,count,std")
    df, column = _filter(ctx, filters), _col(ctx, column)
    if group_by:
        out = df.groupby(_col(ctx, group_by))[column].agg(agg).sort_values(ascending=False).head(int(top_n))
        return {"result": {str(k): round(float(v), 4) for k, v in out.items()}}
    return {"result": round(float(getattr(df[column], agg)()), 4), "rows_used": len(df)}


def correlations(ctx, top_n=10):
    c = ctx.df.select_dtypes("number").corr()
    pairs = [(a, b, float(c.loc[a, b])) for i, a in enumerate(c.columns) for b in c.columns[i + 1:]
             if not np.isnan(c.loc[a, b])]
    pairs.sort(key=lambda t: -abs(t[2]))
    return {"top_pairs": [{"a": a, "b": b, "corr": round(v, 3)} for a, b, v in pairs[:int(top_n)]]}


def detect_outliers(ctx, contamination=0.02):
    num = ctx.df.select_dtypes("number")
    if num.shape[1] == 0:
        raise ValueError("No numeric columns")
    X = SimpleImputer(strategy="median").fit_transform(num)
    flag = IsolationForest(contamination=float(contamination), random_state=42).fit_predict(X) == -1
    return {"n_outliers": int(flag.sum()), "share": round(float(flag.mean()), 4),
            "example_row_indexes": [int(i) for i in ctx.df.index[flag][:10]]}


def plot(ctx, kind, x, y=None):
    df = ctx.df
    fig, ax = plt.subplots(figsize=(6, 4))
    x = _col(ctx, x)
    if kind == "hist":
        ax.hist(df[x].dropna(), bins=30); title = f"Distribution of {x}"
    elif kind == "scatter":
        ax.scatter(df[x], df[_col(ctx, y)], s=8, alpha=0.6); ax.set_ylabel(y); title = f"{y} vs {x}"
    elif kind == "box":
        (df.boxplot(column=_col(ctx, y), by=x, ax=ax) if y else df.boxplot(column=x, ax=ax))
        title = f"Box plot of {y or x}" + (f" by {x}" if y else ""); fig.suptitle("")
    elif kind == "bar":
        s = df.groupby(x)[_col(ctx, y)].mean().sort_values().tail(15) if y else df[x].value_counts().head(15)
        s.plot.bar(ax=ax); title = f"Mean {y} by {x}" if y else f"Counts of {x}"
    else:
        plt.close(fig); raise ValueError("kind must be hist, scatter, box or bar")
    ax.set_xlabel(x); ax.set_title(title); fig.tight_layout()
    ctx.figs.append((title, fig))
    return {"status": "plot created", "title": title}


def train_model(ctx, target, features=None):
    df = ctx.df.dropna(subset=[_col(ctx, target)])
    y = df[target]
    cols = [_col(ctx, c) for c in features] if features else [c for c in df.columns if c != target]
    X = df[cols]
    X = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or X[c].nunique() <= 50]]
    task = "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10 else "classification"
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
    ])
    if task == "classification":
        scoring, cv = "f1_macro", int(max(2, min(5, y.value_counts().min())))
        models = {"baseline": DummyClassifier(), "logistic_regression": LogisticRegression(max_iter=2000),
                  "random_forest": RandomForestClassifier(n_estimators=200, random_state=42)}
    else:
        scoring, cv = "r2", 5
        models = {"baseline": DummyRegressor(), "ridge": Ridge(),
                  "random_forest": RandomForestRegressor(n_estimators=200, random_state=42)}
    scores = {n: float(cross_val_score(Pipeline([("pre", pre), ("m", m)]), X, y, cv=cv, scoring=scoring).mean())
              for n, m in models.items()}
    best = max(scores, key=scores.get)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42,
                                          stratify=y if task == "classification" else None)
    pipe = Pipeline([("pre", pre), ("m", models[best])]).fit(Xtr, ytr)
    imp = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=42, scoring=scoring)
    top = sorted(zip(X.columns, imp.importances_mean), key=lambda t: -t[1])[:8]
    return {"task": task, "metric": scoring, "cv_scores": {k: round(v, 4) for k, v in scores.items()},
            "best_model": best, "top_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in top]}


TOOLS = {"describe_data": describe_data, "aggregate": aggregate, "correlations": correlations,
         "detect_outliers": detect_outliers, "plot": plot, "train_model": train_model}

_FILTERS = {"type": "array", "description": "Optional row filters",
            "items": {"type": "object", "properties": {"column": {"type": "string"},
                      "op": {"type": "string", "enum": list(OPS)}, "value": {"type": ["string", "number"]}},
                      "required": ["column", "op", "value"]}}


def _schema(name, desc, props, req=()):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": list(req)}}


TOOL_SCHEMAS = [
    _schema("describe_data", "Row count, column dtypes, missing values and numeric summary. Call first.", {}),
    _schema("aggregate", "Compute mean/sum/median/min/max/count/std of a column, optionally grouped and filtered.",
            {"column": {"type": "string"}, "agg": {"type": "string"}, "group_by": {"type": "string"},
             "filters": _FILTERS, "top_n": {"type": "integer"}}, ["column"]),
    _schema("correlations", "Strongest pairwise correlations between numeric columns.", {"top_n": {"type": "integer"}}),
    _schema("detect_outliers", "Find anomalous rows with Isolation Forest.", {"contamination": {"type": "number"}}),
    _schema("plot", "Create a chart shown to the user. kind: hist, scatter, box or bar.",
            {"kind": {"type": "string"}, "x": {"type": "string"}, "y": {"type": "string"}}, ["kind", "x"]),
    _schema("train_model", "Auto-detect classification/regression, cross-validate baseline vs models, "
            "return best model and permutation feature importance for a target column.",
            {"target": {"type": "string"}, "features": {"type": "array", "items": {"type": "string"}}}, ["target"]),
]


def call_tool(name, args, ctx):
    """Dispatch a tool call; errors are returned to the model so it can self-correct."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return TOOLS[name](ctx, **(args or {}))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def auto_analysis(ctx, target):
    """No-LLM pipeline: profile, correlations, outliers, model comparison."""
    return {"profile": describe_data(ctx), "correlations": correlations(ctx),
            "outliers": detect_outliers(ctx), "model": train_model(ctx, target)}
