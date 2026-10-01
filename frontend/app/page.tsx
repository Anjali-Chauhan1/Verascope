"use client";

import { useState } from "react";
import styles from "./page.module.css";

type StylisticStats = {
  word_count: number;
  uppercase_ratio: number;
  exclamation_ratio: number;
  question_ratio: number;
  avg_sentence_len: number;
};

type PredictResult = {
  verdict: "REAL" | "FAKE" | "UNKNOWN";
  confidence: number;
  fake_probability: number;
  real_probability: number;
  top_fake_words: [string, number][];
  top_real_words: [string, number][];
  stylistic_stats: StylisticStats;
  error?: string;
};

const MODELS = ["Logistic Regression", "Linear SVM", "DistilBERT (Transformer)"];

const SAMPLES = [
  {
    label: "Conspiracy claim",
    title:
      "Secret bunker found with leaked agendas mainstream media won't show you",
    text: "Whistleblower documents allegedly prove high-ranking elites have been conspiring to manipulate upcoming election dates. Uncensored footage reveals meetings that major news channels refuse to broadcast. Share this before it gets taken down.",
  },
  {
    label: "Miracle cure",
    title: "Doctors stunned: ancient spice cures all ailments overnight",
    text: "Independent researchers claim that drinking this spice-based potion dissolves fat and erases joint pain within 24 hours, with no prescription required. Click here to uncover the formula pharmaceutical companies allegedly tried to suppress.",
  },
  {
    label: "Monetary policy",
    title:
      "Federal Reserve holds benchmark lending rate steady amid sustained growth",
    text: "The Federal Open Market Committee concluded its two-day policy meeting today, voting to maintain the target federal funds rate in the 5.00 to 5.25 percent range, citing labor market resilience and moderating consumer prices.",
  },
  {
    label: "Space science",
    title: "James Webb telescope detects water vapor on habitable exoplanet",
    text: "Astrophysicists analyzing transmission spectroscopy data from the James Webb Space Telescope announced the detection of atmospheric water vapor on an Earth-sized exoplanet orbiting within its host star's habitable zone. The findings were published in Nature Astronomy.",
  },
];

export default function Home() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [model, setModel] = useState(MODELS[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<PredictResult | null>(null);

  function loadSample(i: number) {
    setTitle(SAMPLES[i].title);
    setText(SAMPLES[i].text);
    setError("");
    setResult(null);
  }

  async function analyze() {
    const content = (title + " " + text).trim();
    if (content.length < 15) {
      setError("Please enter at least 15 characters of content.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text, model }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error || "Something went wrong while analyzing the article.");
        return;
      }
      setResult(data);
    } catch {
      setError(
        "Cannot reach the prediction API. Make sure the backend is running:\n\npython app/api.py"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <span className={styles.brand}>Verascope</span>
          <span className={styles.tagline}>NLP Fake News Detector</span>
        </div>
      </header>

      <main className={styles.main}>
        <section className={styles.hero}>
          <h1>Is this news real, or fake?</h1>
          <p>
            Paste a headline and article body. A TF-IDF linear classifier
            trained on WELFake scores it for authenticity and shows the
            specific words that drove the decision.
          </p>
        </section>

        <div className={styles.panel}>
          <span className={styles.samplesLabel}>Try an example</span>
          <div className={styles.samples}>
            {SAMPLES.map((s, i) => (
              <button
                key={s.label}
                className={styles.sampleBtn}
                onClick={() => loadSample(i)}
                type="button"
              >
                {s.label}
              </button>
            ))}
          </div>

          <hr className={styles.divider} />

          <div className={styles.field}>
            <label htmlFor="title-input">Headline</label>
            <input
              id="title-input"
              className={styles.input}
              type="text"
              placeholder="Enter the news headline"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </div>

          <div className={styles.field}>
            <label htmlFor="text-input">Article body</label>
            <textarea
              id="text-input"
              className={styles.textarea}
              placeholder="Paste the full article or key paragraphs"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div className={styles.formFooter}>
            <button
              className={styles.analyzeBtn}
              onClick={analyze}
              disabled={loading}
              type="button"
            >
              {loading ? "Analyzing…" : "Analyze article"}
            </button>

            <div className={styles.modelField}>
              <label htmlFor="model-select">Model</label>
              <select
                id="model-select"
                className={styles.select}
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                {MODELS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {error && <div className={styles.errorMsg}>{error}</div>}
        </div>

        {result && (
          <div className={styles.results}>
            <div
              className={`${styles.verdict} ${
                result.verdict === "REAL" ? styles.real : styles.fake
              }`}
            >
              <div>
                <div className={styles.verdictLabel}>
                  {result.verdict === "REAL" ? "Likely real" : "Likely fake"}
                </div>
                <div className={styles.verdictSub}>
                  Classified with {model}
                </div>
              </div>
              <div className={styles.confidenceBlock}>
                <div className={styles.confidenceValue}>
                  {result.confidence}%
                </div>
                <div className={styles.confidenceCaption}>Confidence</div>
              </div>
            </div>

            <div className={styles.panel}>
              <div className={styles.sectionTitle}>
                Veracity probability breakdown
              </div>
              <div className={styles.probRow}>
                <div>
                  <div className={`${styles.probHeader} ${styles.real}`}>
                    <span>Real</span>
                    <span>{result.real_probability}%</span>
                  </div>
                  <div className={styles.barTrack}>
                    <div
                      className={`${styles.barFill} ${styles.real}`}
                      style={{ width: `${result.real_probability}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className={`${styles.probHeader} ${styles.fake}`}>
                    <span>Fake</span>
                    <span>{result.fake_probability}%</span>
                  </div>
                  <div className={styles.barTrack}>
                    <div
                      className={`${styles.barFill} ${styles.fake}`}
                      style={{ width: `${result.fake_probability}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className={styles.panel}>
              <div className={styles.sectionTitle}>
                Why the model decided this
              </div>
              <div className={styles.chipGrid}>
                <div>
                  <div className={`${styles.chipColTitle} ${styles.fake}`}>
                    Pushes toward fake
                  </div>
                  {result.top_fake_words.length ? (
                    <div className={styles.chipList}>
                      {result.top_fake_words.map(([word, score]) => (
                        <span
                          key={word}
                          className={`${styles.wordChip} ${styles.fake}`}
                        >
                          {word}
                          <span className={styles.chipScore}>
                            −{score.toFixed(2)}
                          </span>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.emptyNote}>
                      No strong fake indicators detected.
                    </div>
                  )}
                </div>
                <div>
                  <div className={`${styles.chipColTitle} ${styles.real}`}>
                    Pushes toward real
                  </div>
                  {result.top_real_words.length ? (
                    <div className={styles.chipList}>
                      {result.top_real_words.map(([word, score]) => (
                        <span
                          key={word}
                          className={`${styles.wordChip} ${styles.real}`}
                        >
                          {word}
                          <span className={styles.chipScore}>
                            +{score.toFixed(2)}
                          </span>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div className={styles.emptyNote}>
                      No strong real indicators detected.
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className={styles.panel}>
              <div className={styles.sectionTitle}>
                Stylistic &amp; structural signals
              </div>
              <div className={styles.telemetryGrid}>
                <div className={styles.teleCard}>
                  <div className={styles.teleValue}>
                    {result.stylistic_stats.word_count}
                  </div>
                  <div className={styles.teleLabel}>Words</div>
                </div>
                <div className={styles.teleCard}>
                  <div className={styles.teleValue}>
                    {result.stylistic_stats.uppercase_ratio}%
                  </div>
                  <div className={styles.teleLabel}>All-caps ratio</div>
                </div>
                <div className={styles.teleCard}>
                  <div className={styles.teleValue}>
                    {result.stylistic_stats.exclamation_ratio}
                  </div>
                  <div className={styles.teleLabel}>Exclamation rate</div>
                </div>
                <div className={styles.teleCard}>
                  <div className={styles.teleValue}>
                    {result.stylistic_stats.question_ratio}
                  </div>
                  <div className={styles.teleLabel}>Question rate</div>
                </div>
                <div className={styles.teleCard}>
                  <div className={styles.teleValue}>
                    {result.stylistic_stats.avg_sentence_len}
                  </div>
                  <div className={styles.teleLabel}>Avg sentence length</div>
                </div>
              </div>
            </div>

            <p className={styles.disclaimer}>
              <strong>Academic disclaimer.</strong> This model judges
              linguistic writing style and lexical patterns, not live factual
              truth — a well-written falsehood can be misclassified. Always
              cross-check with a fact-checker such as{" "}
              <a href="https://www.altnews.in" target="_blank" rel="noopener noreferrer">
                Alt News
              </a>
              ,{" "}
              <a href="https://factcheck.pib.gov.in" target="_blank" rel="noopener noreferrer">
                PIB Fact Check
              </a>{" "}
              or{" "}
              <a href="https://www.snopes.com" target="_blank" rel="noopener noreferrer">
                Snopes
              </a>
              .
            </p>
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <span>Final-year NLP capstone project</span>
          <span>Trained on WELFake · validated on ISOT</span>
        </div>
      </footer>
    </div>
  );
}
