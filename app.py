import altair as alt
import pandas as pd
import streamlit as st
from sklearn.datasets import load_wine
from tools import Ctx, auto_analysis

ACCENT, MUTED = "#7C5CFF", "#3A4360"
st.set_page_config(page_title="AI Data Analyst Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {padding: 2rem 2.2rem; border-radius: 22px; color: #fff; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #3B2A8F 0%, #7C5CFF 48%, #22D3EE 100%);
       box-shadow: 0 10px 40px rgba(124,92,255,.25);}
.hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800; line-height: 1.15;}
.hero p {opacity: .93; margin: .5rem 0 0; font-size: 1.05rem;}
.pill {display: inline-block; padding: .22rem .8rem; margin: .8rem .45rem 0 0; border-radius: 999px;
       background: rgba(255,255,255,.18); font-size: .8rem; font-weight: 600;}
.card {background: #131A2A; border: 1px solid #232C42; border-radius: 16px; padding: 1rem 1.2rem; height: 100%;}
.card .lbl {color: #93A0B8; font-size: .74rem; text-transform: uppercase; letter-spacing: .07em;}
.card .val {font-size: 1.75rem; font-weight: 800; margin: .1rem 0;}
.card .sub {color: #93A0B8; font-size: .8rem;}
.insight {border-left: 4px solid #7C5CFF; background: #131A2A; padding: .9rem 1.2rem;
          border-radius: 10px; margin: 1rem 0; font-size: 1.02rem;}
h3 {margin-top: 1.2rem;}
</style>
""", unsafe_allow_html=True)


def card(label, value, sub=""):
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def bar(df, x, y, best=None, title=None):
    color = alt.condition(alt.datum[y] == best, alt.value(ACCENT), alt.value(MUTED)) if best else alt.value(ACCENT)
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=6)
            .encode(x=alt.X(x, title=title), y=alt.Y(y, sort="-x", title=None), color=color, tooltip=list(df.columns))
            .properties(height=max(120, 34 * len(df))).configure_view(strokeWidth=0)
            .configure_axis(gridColor="#232C42", labelColor="#C9D1E3", titleColor="#93A0B8"))


@st.cache_data
def sample():
    df = load_wine(as_frame=True).frame
    df["cultivar"] = df.pop("target").map({0: "A", 1: "B", 2: "C"})
    return df


@st.cache_data(show_spinner="Profiling data, running AutoML and outlier detection...")
def insights(df, target):
    return auto_analysis(Ctx(df), target)


with st.sidebar:
    st.markdown("### ⚙️ Setup")
    up = st.file_uploader("Upload a CSV", type=["csv"], help="Or leave empty to use the sample wine dataset.")
    api_key = st.text_input("Anthropic API key (for chat)", type="password",
                            help="Only needed for the Agent Chat tab. Never stored.")
    model = st.text_input("Model", "claude-sonnet-5-5")
    st.caption("🔒 Your key stays in this browser session. Instant Insights needs no key.")

df = pd.read_csv(up) if up else sample()
key = (up.name if up else "sample", len(df))
if st.session_state.get("key") != key:
    st.session_state.update(key=key, ctx=Ctx(df), chat=[])
ctx = st.session_state.ctx

st.markdown("""
<div class="hero">
  <h1>🤖 AI Data Analyst Agent</h1>
  <p>Drop in a CSV. An LLM agent plans, runs safe analysis tools, and explains what it finds, with every step visible.</p>
  <span class="pill">LLM Agent · Tool Use</span><span class="pill">AutoML</span>
  <span class="pill">Explainability</span><span class="pill">Anomaly Detection</span><span class="pill">Guardrails</span>
</div>
""", unsafe_allow_html=True)

d = ctx.df
k1, k2, k3, k4 = st.columns(4)
k1.markdown(card("Rows", f"{len(d):,}", "records analysed"), unsafe_allow_html=True)
k2.markdown(card("Columns", d.shape[1], f"{d.select_dtypes('number').shape[1]} numeric"), unsafe_allow_html=True)
k3.markdown(card("Missing values", f"{d.isna().mean().mean():.1%}", "across all cells"), unsafe_allow_html=True)
k4.markdown(card("Dataset", "Uploaded" if up else "Wine (sample)", key[0] if up else "scikit-learn"), unsafe_allow_html=True)
with st.expander("👀 Preview data"):
    st.dataframe(d.head(10), width="stretch")

tab1, tab2, tab3 = st.tabs(["⚡ Instant Insights", "💬 Agent Chat", "🛡️ How it works"])

with tab1:
    target = st.selectbox("🎯 What do you want to predict?", list(d.columns), index=len(d.columns) - 1)
    r = insights(d, target)
    m, scores = r["model"], r["model"]["cv_scores"]
    best = m["best_model"]
    gain = scores[best] - scores["baseline"]
    top = m["top_features"]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Task", m["task"].title(), f"metric: {m['metric']}"), unsafe_allow_html=True)
    c2.markdown(card("Best model", best.replace("_", " ").title(), "cross-validated"), unsafe_allow_html=True)
    c3.markdown(card("Score", f"{scores[best]:.3f}", f"+{gain:.2f} vs baseline"), unsafe_allow_html=True)
    c4.markdown(card("Outliers", r["outliers"]["n_outliers"], f"{r['outliers']['share']:.1%} of rows flagged"),
                unsafe_allow_html=True)
    drivers = ", ".join(f"**{t['feature']}**" for t in top[:3])
    st.markdown(f'<div class="insight">💡 <b>{best.replace("_", " ")}</b> reaches {scores[best]:.2f} '
                f'({m["metric"]}) versus {scores["baseline"]:.2f} for a naive baseline. '
                f'Strongest drivers of <code>{target}</code>: {drivers}.</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("### Model comparison")
        st.altair_chart(bar(pd.DataFrame({"model": list(scores), "score": list(scores.values())}),
                            "score:Q", "model:N", best, m["metric"]), width="stretch")
    with b:
        st.markdown("### What drives the prediction")
        st.altair_chart(bar(pd.DataFrame(top), "importance:Q", "feature:N", None, "permutation importance"),
                        width="stretch")
    st.markdown("### Strongest correlations")
    st.dataframe(pd.DataFrame(r["correlations"]["top_pairs"]), width="stretch", hide_index=True,
                 column_config={"corr": st.column_config.ProgressColumn("corr", min_value=-1, max_value=1, format="%.2f")})

with tab2:
    if not api_key:
        st.info("🔑 Add your Anthropic API key in the sidebar to chat with the agent. "
                "Instant Insights works without one.")
    num = d.select_dtypes("number").columns.tolist()
    ideas = ["Give me an overview of this dataset", f"Which features best predict {d.columns[-1]}?",
             "Find outliers and explain them", f"Plot the distribution of {num[0]}" if num else "Show correlations"]
    st.caption("Try one:")
    cols = st.columns(len(ideas))
    for i, idea in enumerate(ideas):
        cols[i].button(idea, key=f"idea{i}", disabled=not api_key, width="stretch",
                       on_click=lambda t=idea: st.session_state.update(pending=t))
    for turn in st.session_state.chat:
        st.chat_message("user").write(turn["q"])
        with st.chat_message("assistant"):
            st.write(turn["a"])
            for title, fig in turn["figs"]:
                st.pyplot(fig)
            with st.expander(f"🔧 Agent trace: {len(turn['trace'])} tool call(s)"):
                for step in turn["trace"]:
                    st.markdown(f"**{step['tool']}** `{step['input']}`")
                    st.json(step["output"], expanded=False)
    q = st.chat_input("Ask about your data...", disabled=not api_key) or st.session_state.pop("pending", None)
    if q and api_key:
        from anthropic import Anthropic
        from agent import run_agent
        n0 = len(ctx.figs)
        with st.spinner("Agent is thinking and running tools..."):
            try:
                a, trace = run_agent(q, ctx, Anthropic(api_key=api_key), model,
                                     [(t["q"], t["a"]) for t in st.session_state.chat])
            except Exception as e:
                a, trace = f"⚠️ {e}", []
        st.session_state.chat.append({"q": q, "a": a, "trace": trace, "figs": ctx.figs[n0:]})
        st.rerun()

with tab3:
    st.markdown("""
### Architecture
`Question` → **LLM (tool use)** → `describe_data · aggregate · correlations · detect_outliers · plot · train_model`
→ results fed back → **grounded answer + full trace**

### Guardrails
- 🚫 **No `eval` / `exec`**: typed tool arguments, validated column names, whitelisted filter operators
- ⏱️ Step limit, row cap and output truncation
- 🔁 Tool errors are returned to the model so it can self-correct
- 🛡️ Dataset content is treated as untrusted data (prompt-injection defence)
- 🔒 Your API key lives only in your browser session

### Stack
Python · scikit-learn · pandas · Altair · Streamlit · Anthropic API · pytest · GitHub Actions · Docker
""")
