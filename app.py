import os
import io
import asyncio
import tempfile
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chainlit as cl
from PIL import Image
import google.generativeai as genai

import matplotlib
matplotlib.use("Agg")

# Single multimodal model for text and vision
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
    - pandas info
    - preview of first rows
    - missing values summary
    """
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()

    # Use plain text table so we do not need tabulate
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


def extract_gemini_text(res):
    """
    Safely extract text from a Gemini response.
    Handles safety blocks and empty candidates.
    """
    # Try the quick accessor, but catch the safety case
    try:
        if getattr(res, "text", None):
            return res.text
    except Exception:
        # This is where you were getting:
        # Invalid operation... no valid Part... finish_reason is 2
        pass

    candidates = getattr(res, "candidates", None)
    if not candidates:
        return "Model returned no candidates."

    cand = candidates[0]

    # finish_reason can be int or enum depending on client version
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


async def ai_text_analysis(prompt_type, df_context):
    """
    Ask Gemini for an analysis plan or a final summary using gemini 2 point 5 flash.
    """
    if not GEMINI_AVAILABLE:
        return "Gemini AI not available."

    prompts = {
        "plan": (
            "You are a helpful data analyst. "
            "Suggest a concise high level data analysis plan for the following dataset:\n"
            f"{df_context}"
        ),
        "final": (
            "You are a helpful data analyst. "
            "Summarize the key insights from the following dataset analysis:\n"
            f"{df_context}"
        ),
    }

    if prompt_type not in prompts:
        return "Unknown prompt type."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        res = await model.generate_content_async(
            prompts[prompt_type],
            generation_config={
                "max_output_tokens": 500,
                "temperature": 0.3,
            },
        )

        return extract_gemini_text(res)
    except Exception as e:
        return f"Gemini error: {e}"


async def ai_vision_analysis(img_paths):
    """
    Take a list of (title, path) and ask Gemini to explain each image.
    Uses the same gemini 2 point 5 flash model in multimodal mode.
    Returns list of (title, insight).
    """
    if not GEMINI_AVAILABLE:
        return [("AI Vision", "Gemini not available.")]

    results = []

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        return [("AI Vision", f"Gemini model init error: {e}")]

    for title, path in img_paths:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()

            prompt = f"Explain this plot titled '{title}' to a beginner data analyst."

            res = await model.generate_content_async(
                [
                    prompt,
                    {
                        "mime_type": "image/png",
                        "data": img_bytes,
                    },
                ],
                generation_config={
                    "max_output_tokens": 200,
                    "temperature": 0.2,
                },
            )

            text = extract_gemini_text(res)
            results.append((title, text))

        except Exception as e:
            results.append((title, f"Error: {e}"))

    return results


def generate_visuals(df):
    """
    Generate basic EDA visuals and return:
    - visualizations: list of (title, path)
    - saved_files: list of paths so that they can be cleaned up later
    """
    visualizations = []
    saved_files = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Categorical cols prepared for future use
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
            saved_files.append(path)

        # Pairplot on up to 5 numeric columns
        if len(numeric_cols) >= 3:
            sns.set(style="ticks")
            pair_df = df[numeric_cols[:5]].dropna()
            if not pair_df.empty:
                g = sns.pairplot(pair_df)
                g.fig.suptitle("Pairplot of Numeric Features", y=1.02)
                path = save_fig(g.fig)
                visualizations.append(("Pairplot", path))
                saved_files.append(path)

        # Violin plots for first three numeric columns
        for col in numeric_cols[:3]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=df, y=col, ax=ax)
            ax.set_title(f"Violin Plot for {col}")
            path = save_fig(fig)
            visualizations.append((f"Violin Plot for {col}", path))
            saved_files.append(path)

    except Exception as e:
        print(f"Visualization error: {e}")
        plt.close("all")

    return visualizations, saved_files


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

        # Save dataframe in user session
        cl.user_session.set("df", df)

        # Show schema and preview
        info = df_info_string(df)
        await cl.Message(content=info).send()

        # AI analysis plan
        if GEMINI_AVAILABLE:
            plan = await ai_text_analysis("plan", info)
            await cl.Message(content=f"### AI Plan\n{plan}").send()
        else:
            await cl.Message(
                content="⚠ Gemini is not available. Set GOOGLE_API_KEY to enable AI insights."
            ).send()

        # Generate visuals
        visuals, saved_files = generate_visuals(df)

        for title, path in visuals:
            await cl.Message(
                content=f"**{title}**",
                elements=[cl.Image(name=title, path=path)],
            ).send()

        # Ask Gemini to explain visuals
        if GEMINI_AVAILABLE and visuals:
            insights = await ai_vision_analysis(visuals)
            for title, insight in insights:
                await cl.Message(
                    content=f"### {title} Insight\n{insight}"
                ).send()

        processing_msg.content = "✅ Analysis complete."
        await processing_msg.update()

        # Clean up image files
        await cleanup(saved_files)

    except Exception as e:
        traceback.print_exc()
        processing_msg.content = f"❌ Error: {e}"
        await processing_msg.update()
