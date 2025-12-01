import os
import io
import tempfile
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chainlit as cl
import google.generativeai as genai

import matplotlib
matplotlib.use("Agg")

# Single model for text analysis
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_AVAILABLE = False

# Initialise Gemini
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set. Gemini disabled.")
    else:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
except Exception as e:
    print(f"Gemini init failed: {e}")


def save_fig(fig):
    """Save a Matplotlib figure to a temporary PNG file and return the path."""
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(f.name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return f.name


def df_info_string(df, max_rows=5):
    """
    Build a markdown style string with:
    pandas info
    preview of first rows
    missing values summary
    """
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()

    head_str = df.head(max_rows).to_string(index=False)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values." if missing.empty else str(missing)

    return (
        "### Schema:\n"
        "```\n"
        f"{schema}"
        "```\n\n"
        "### Preview:\n"
        "```\n"
        f"{head_str}\n"
        "```\n\n"
        f"### Missing:\n{missing_info}"
    )


def build_stats_context(df):
    """
    Build a compact numeric summary for Gemini.
    This avoids sending raw images and reduces safety triggers.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    parts = []

    # Basic describe
    if numeric_cols:
        desc = df[numeric_cols].describe().to_string()
        parts.append("Numeric summary statistics:\n" + desc)

        # Correlation limited to at most 6 columns
        limited_cols = numeric_cols[:6]
        corr = df[limited_cols].corr().round(3).to_string()
        parts.append("\nCorrelation matrix on selected numeric columns:\n" + corr)

    # Simple grouped summary for a categorical column if present
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols and numeric_cols:
        cat = cat_cols[0]
        num = numeric_cols[0]
        grouped = (
            df.groupby(cat)[num]
            .agg(["count", "mean", "sum"])
            .round(2)
            .to_string()
        )
        parts.append(
            f"\nGrouped statistics by {cat} for {num}:\n" + grouped
        )

    if not parts:
        return "Dataset has no numeric columns. Only schema is available."

    return "\n\n".join(parts)


def extract_gemini_text(res):
    """
    Safely extract text from a Gemini response.
    Handles safety blocks and empty candidates.
    """
    try:
        if getattr(res, "text", None):
            return res.text
    except Exception:
        pass

    candidates = getattr(res, "candidates", None)
    if not candidates:
        return "Model returned no candidates."

    cand = candidates[0]

    finish_reason = getattr(cand, "finish_reason", None)
    finish_name = getattr(finish_reason, "name", None)
    if finish_reason == 2 or finish_name == "SAFETY":
        return "Response blocked by Gemini safety filters."

    content = getattr(cand, "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return "Model returned an empty response."

    text_pieces = [
        getattr(p, "text", "")
        for p in parts
        if getattr(p, "text", "")
    ]
    if not text_pieces:
        return "Model returned no text parts."

    return "\n".join(text_pieces)


async def ai_text_analysis(prompt_type, context_text):
    """
    Ask Gemini for a plan or insights based only on text context.
    """
    if not GEMINI_AVAILABLE:
        return "Gemini AI not available."

    prompts = {
        "plan": (
            "You are a helpful data analyst. "
            "Given the dataset schema, preview and missing value info below, "
            "suggest a concise high level analysis plan. "
            "Focus on simple steps such as cleaning, feature understanding and basic plots.\n\n"
            f"{context_text}"
        ),
        "insights": (
            "You are a helpful data analyst. "
            "Based on the numeric summary statistics and correlation info below, "
            "explain the main patterns and relationships in simple language. "
            "Avoid any speculation about individuals. Focus only on aggregate trends.\n\n"
            f"{context_text}"
        ),
    }

    if prompt_type not in prompts:
        return "Unknown prompt type."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        res = await model.generate_content_async(
            prompts[prompt_type],
            generation_config={
                "max_output_tokens": 600,
                "temperature": 0.3,
            },
        )

        return extract_gemini_text(res)
    except Exception as e:
        return f"Gemini error: {e}"


def generate_visuals(df):
    """
    Generate basic EDA visuals and return:
    visualizations: list of (title, path).
    """
    visualizations = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        col
        for col in df.select_dtypes(include="object")
        if 1 < df[col].nunique() < 30
    ]

    try:
        # Correlation heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr,
                mask=mask,
                cmap="coolwarm",
                annot=True,
                fmt=".2f",
                ax=ax,
            )
            ax.set_title("Correlation Heatmap")
            path = save_fig(fig)
            visualizations.append(("Correlation Heatmap", path))

        # Pairplot on up to 5 numeric columns
        if len(numeric_cols) >= 3:
            sns.set(style="ticks")
            pair_df = df[numeric_cols[:5]].dropna()
            if not pair_df.empty:
                g = sns.pairplot(pair_df)
                g.fig.suptitle("Pairplot of Numeric Features", y=1.02)
                path = save_fig(g.fig)
                visualizations.append(("Pairplot", path))

        # Violin plots for first three numeric columns
        for col in numeric_cols[:3]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=df, y=col, ax=ax)
            ax.set_title(f"Violin Plot for {col}")
            path = save_fig(fig)
            visualizations.append((f"Violin Plot for {col}", path))

    except Exception as e:
        print(f"Visualization error: {e}")
        plt.close("all")

    return visualizations


async def cleanup(files):
    """Remove temporary files."""
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass


@cl.on_chat_start
async def start():
    await cl.Message(
        content="📊 Upload a CSV file for AI powered EDA using Gemini."
    ).send()

    files = await cl.AskFileMessage(
        content="Upload a CSV file",
        accept=["text/csv"],
    ).send()

    if not files:
        await cl.Message(content="❌ No file received.").send()
        return

    processing_msg = cl.Message(content="💬 Processing your dataset...")
    await processing_msg.send()

    try:
        # Read CSV directly from Chainlit temp path
        df = pd.read_csv(files[0].path)

        if df.empty:
            processing_msg.content = "❌ Empty dataset."
            await processing_msg.update()
            return

        cl.user_session.set("df", df)

        # Show schema and preview
        info = df_info_string(df)
        await cl.Message(content=info).send()

        # Ask Gemini for analysis plan using schema info only
        if GEMINI_AVAILABLE:
            plan = await ai_text_analysis("plan", info)
            await cl.Message(content=f"### AI Plan\n{plan}").send()
        else:
            await cl.Message(
                content="⚠ Gemini is not available. Set GOOGLE_API_KEY to enable AI insights."
            ).send()

        # Generate visuals for the user (no images sent to Gemini)
        visuals = generate_visuals(df)
        temp_paths = []
        for title, path in visuals:
            temp_paths.append(path)
            await cl.Message(
                content=f"**{title}**",
                elements=[cl.Image(name=title, path=path)],
            ).send()

        # Build numeric stats context and ask Gemini for insights
        if GEMINI_AVAILABLE:
            stats_context = build_stats_context(df)
            insights = await ai_text_analysis("insights", stats_context)
            await cl.Message(content=f"### Data Insights\n{insights}").send()

        processing_msg.content = "✅ Analysis complete."
        await processing_msg.update()

        await cleanup(temp_paths)

    except Exception as e:
        traceback.print_exc()
        processing_msg.content = f"❌ Error: {e}"
        await processing_msg.update()
