import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is available for imports
sys.path.append(os.path.abspath("."))
from src.predict import FakeNewsPredictor

# -------------------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Verascope | Fake News Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# CUSTOM CSS: MODERN DARK GLASSMORPHISM & TYPOGRAPHY
# -------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, h4, .headline-font {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphism containers */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.verdict-badge-fake {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.25) 0%, rgba(192, 57, 43, 0.4) 100%);
    border: 1px solid #e74c3c;
    color: #ff6b6b;
    padding: 12px 24px;
    border-radius: 12px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 1px;
    display: inline-block;
    width: 100%;
    box-shadow: 0 0 20px rgba(231, 76, 60, 0.2);
}

.verdict-badge-real {
    background: linear-gradient(135deg, rgba(46, 204, 113, 0.25) 0%, rgba(39, 174, 96, 0.4) 100%);
    border: 1px solid #2ecc71;
    color: #2ecc71;
    padding: 12px 24px;
    border-radius: 12px;
    font-size: 1.5rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 1px;
    display: inline-block;
    width: 100%;
    box-shadow: 0 0 20px rgba(46, 204, 113, 0.2);
}

.metric-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}

.word-chip-fake {
    background: rgba(231, 76, 60, 0.15);
    border: 1px solid rgba(231, 76, 60, 0.4);
    color: #ff7675;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 3px;
    display: inline-block;
}

.word-chip-real {
    background: rgba(46, 204, 113, 0.15);
    border: 1px solid rgba(46, 204, 113, 0.4);
    color: #55efc4;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 3px;
    display: inline-block;
}

.hero-banner {
    padding: 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# MODEL CACHING VIA STREAMLIT
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_predictor():
    return FakeNewsPredictor(models_dir="models")

predictor = get_predictor()

# -------------------------------------------------------------------------
# SAMPLE DATA ARTICLES (FOR QUICK DEMONSTRATION)
# -------------------------------------------------------------------------
SAMPLE_ARTICLES = {
    "Select a pre-loaded sample...": {"title": "", "text": ""},
    "🚨 Fake: Sensational Political Conspiracy": {
        "title": "BREAKING: Secret Bunker Found With Leaked Agendas Mainstream Media Won't Show You!",
        "text": "Shocking whistleblower documents prove high-ranking government elites have been conspiring secretly to manipulate upcoming national election dates. Uncensored footage reveals clandestine rendezvous that globalist news channels refuse to broadcast! Share this viral story immediately before authorities take it down!"
    },
    " Fake: Miracle Health Clickbait": {
        "title": "DOCTORS STUNNED: Ancient Household Spice Cures All Ailments Overnight!",
        "text": "Big pharma executives are panicking after independent researchers accidentally proved that drinking this boiling spice potion dissolves toxic fat and erases years of joint pain in just 24 hours. No prescription required! Click here to uncover the secret formula they tried to suppress!"
    },
    " Real: Central Bank Monetary Policy": {
        "title": "Federal Reserve Holds Benchmark Lending Rate Steady Amid Sustained Economic Growth",
        "text": "The Federal Open Market Committee concluded its two-day monetary policy summit today, voting unanimously to maintain target federal funds rates in the 5.00 to 5.25 percent range. Central bank officials noted persistent labor market resilience and moderate consumer price index cooling, emphasizing that future decisions remain strictly data-dependent."
    },
    " Real: Scientific Discovery Announcement": {
        "title": "NASA James Webb Space Telescope Detects Water Vapor in Habitable Exoplanet Atmosphere",
        "text": "Astrophysicists analyzing transmission spectroscopy data gathered by the James Webb Space Telescope announced the definitive detection of atmospheric water vapor on an Earth-sized exoplanet orbiting within its host star's habitable zone. The peer-reviewed findings were published Thursday in the journal Nature Astronomy."
    }
}

# -------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    model_choice = st.selectbox(
        "Classifier Architecture:",
        ["Logistic Regression", "Linear SVM", "DistilBERT (Transformer)"],
        index=0,
        help="Logistic Regression provides calibrated probabilities and direct word weights. Linear SVM offers high-dimensional hyperplane separation. DistilBERT uses self-attention."
    )

    st.markdown("---")
    st.markdown("### 🧪 Quick Demonstration")
    sample_key = st.selectbox(
        "Load Pre-configured Example:",
        list(SAMPLE_ARTICLES.keys())
    )

    st.markdown("---")
    st.markdown("### 📊 Model Architecture Specs")
    st.markdown("""
    - **N-Gram Features**: 50,000 (Unigrams + Bigrams)
    - **Weighting**: Sublinear TF-IDF (`1 + log(tf)`)
    - **Leakage Filter**: Dateline & Publisher stripped
    - **Evaluation Metric**: Macro-F1 & ROC-AUC
    """)

    st.markdown("---")
    st.markdown("👨‍🎓 **Final-Year Project by**: Anirudh\n\nMentor Guided • Production Grade")

# -------------------------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="margin: 0; font-size: 2.3rem; background: linear-gradient(90deg, #6c5ce7, #a29bfe, #00cec9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Verascope NLP
            </h1>
            <p style="margin: 5px 0 0 0; color: #a0aec0; font-size: 1.05rem;">
                Automated Fake News Detection & Explainable Decision Attribution
            </p>
        </div>
        <div style="background: rgba(108, 92, 231, 0.15); border: 1px solid rgba(108, 92, 231, 0.4); padding: 6px 14px; border-radius: 20px; color: #a29bfe; font-size: 0.85rem; font-weight: 600;">
            ⚡ LIVE INFERENCE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# INPUT TABS: MANUAL TEXT VS URL PARSER
# -------------------------------------------------------------------------
tab_manual, tab_url = st.tabs(["✍️ Paste Headline & Article Text", "🌐 Fetch Article from Live URL"])

with tab_manual:
    title_input = st.text_input(
        "News Article Headline / Title:",
        value=selected_sample["title"],
        placeholder="e.g. BREAKING: Historic Summit Concludes With Agreement..."
    )

    text_input = st.text_area(
        "News Article Body Content:",
        value=selected_sample["text"],
        height=180,
        placeholder="Paste full news article text or lead paragraphs here..."
    )

with tab_url:
    url_input = st.text_input(
        "Enter Online News Article URL:",
        placeholder="https://www.example-news.com/article-slug"
    )
    url_fetch_btn = st.button("📥 Extract Text from URL")
    
    if url_fetch_btn and url_input:
        with st.spinner("Fetching article content from web source..."):
            try:
                import urllib.request
                import re
                req = urllib.request.Request(url_input, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=7) as response:
                    html_content = response.read().decode("utf-8", errors="ignore")
                
                # Extract title from <title> tag
                title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
                extracted_title = title_match.group(1).strip() if title_match else ""
                
                # Extract text from <p> paragraph tags
                paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html_content, re.IGNORECASE | re.DOTALL)
                clean_paras = [re.sub(r"<.*?>", "", p).strip() for p in paragraphs if len(p.strip()) > 30]
                extracted_text = " ".join(clean_paras[:12])
                
                title_input = extracted_title
                text_input = extracted_text
                st.success(f"Extracted: '{extracted_title[:60]}...' ({len(extracted_text.split())} words)")
            except Exception as e:
                st.error(f"Could not extract article from URL: {e}")

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    analyze_clicked = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

# -------------------------------------------------------------------------
# INFERENCE & RESULTS
# -------------------------------------------------------------------------
full_article_content = (title_input.strip() + " " + text_input.strip()).strip()

if analyze_clicked or (selected_sample["title"] and not analyze_clicked):
    if len(full_article_content) < 15:
        st.warning("⚠️ Please enter a headline or article body of at least 15 characters to analyze.")
    else:
        with st.spinner("Analyzing linguistic patterns and calculating word attributions..."):
            try:
                res = predictor.predict(full_article_content, model_choice=model_choice)
                
                st.markdown("---")
                
                # Top Results Section: Verdict & Confidence
                col_verdict, col_prob = st.columns([1, 2])
                
                with col_verdict:
                    if res["verdict"] == "REAL":
                        st.markdown(f"""
                        <div class="verdict-badge-real">
                            ✅ REAL NEWS
                        </div>
                        <p style="text-align: center; color: #2ecc71; margin-top: 8px; font-weight: 500;">
                            Confidence: <b>{res['confidence']}%</b>
                        </p>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-badge-fake">
                            ⚠️ FAKE / SUSPICIOUS
                        </div>
                        <p style="text-align: center; color: #ff6b6b; margin-top: 8px; font-weight: 500;">
                            Confidence: <b>{res['confidence']}%</b>
                        </p>
                        """, unsafe_allow_html=True)
                
                with col_prob:
                    st.markdown("#### Veracity Probability Breakdown")
                    st.markdown(f"**Real News Score**: `{res['real_probability']}%` | **Fake News Score**: `{res['fake_probability']}%`")
                    # Progress bar with real probability
                    st.progress(res['real_probability'] / 100.0)
                    st.caption(f"Evaluated via **{model_choice}** with 50,000 n-gram features.")

                # Middle Section: Word-level Explainability
                st.markdown("### 🔎 Decision Explainability (Why did the model decide this?)")
                col_exp_fake, col_exp_real = st.columns(2)

                with col_exp_fake:
                    st.markdown("##### 🚨 Words Pushing Toward FAKE:")
                    if res["top_fake_words"]:
                        chips_html = "".join([f'<span class="word-chip-fake">{word} (-{score:.2f})</span>' for word, score in res["top_fake_words"]])
                        st.markdown(chips_html, unsafe_allow_html=True)
                    else:
                        st.caption("No strong negative (fake) indicator words detected.")

                with col_exp_real:
                    st.markdown("##### 📗 Words Pushing Toward REAL:")
                    if res["top_real_words"]:
                        chips_html = "".join([f'<span class="word-chip-real">{word} (+{score:.2f})</span>' for word, score in res["top_real_words"]])
                        st.markdown(chips_html, unsafe_allow_html=True)
                    else:
                        st.caption("No strong positive (real) indicator words detected.")

                # Bottom Section: Stylistic Telemetry
                st.markdown("### 📐 Stylistic & Structural Telemetry")
                stats = res["stylistic_stats"]
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Total Words", f"{stats['word_count']}")
                sc2.metric("ALL-CAPS Ratio", f"{stats['uppercase_ratio']}%")
                sc3.metric("Exclamation Rate", f"{stats['exclamation_ratio']}%")
                sc4.metric("Question Rate", f"{stats['question_ratio']}%")
                sc5.metric("Avg Sentence Len", f"{stats['avg_sentence_len']} w")

            except Exception as e:
                st.error(f"Inference error: {str(e)}")
                st.info("Ensure you have run `python src/train_ml.py` to generate the model artifacts.")

# -------------------------------------------------------------------------
# ACADEMIC DISCLAIMER & CITATIONS
# -------------------------------------------------------------------------
st.markdown("---")
with st.expander("🛡️ Academic Project Notice & Fact-Checking Disclaimer"):
    st.markdown("""
    **Model Scope & Limitations**:
    - This model classifies articles based on **linguistic style, lexical patterns, and sensationalist syntax** trained on published benchmark datasets.
    - An automated NLP model does **not** perform live epistemological truth verification of real-world claims.
    - Always verify controversial claims through accredited fact-checking agencies:
      - [Alt News](https://www.altnews.in/)
      - [PIB Fact Check](https://factcheck.pib.gov.in/)
      - [BOOM Live](https://www.boomlive.in/)
      - [Snopes Fact Check](https://www.snopes.com/)
    """)
