"""
Downloads and prepares the extension / cross-dataset corpora listed in the
project blueprint that do NOT require a Kaggle account or manual registration:

    - LIAR            (Wang, 2017)            -- UCSB direct download
    - COVID-19 Fake News (Patwa et al., 2021)  -- Constraint@AAAI2021 GitHub mirror
    - FakeNewsNet      (Shu et al.)            -- GitHub CSV mirror (title + url only,
                                                   no article body -- see note below)

Three datasets from the blueprint table are intentionally NOT auto-downloaded here
because they require credentials only the project owner has:

    - Kaggle "Fake News" (2018) competition  -- needs a Kaggle account + competition
                                                 rules accepted, then `kaggle competitions
                                                 download -c fake-news`
    - IFND (Sharma & Garg, 2021)              -- Kaggle mirror, same requirement
    - HinFakeNews                             -- IndiaAI AIKosh registration required,
                                                 and is Hindi-language (out of the
                                                 English-only scope of this project;
                                                 see Section 3 "Scope" of the blueprint)

For those three, call the matching `require_*` function below: it checks for the
file in `data/raw/<name>/` and raises a clear, actionable error naming exactly
what to download and where to put it.

Usage:
    python src/load_extra_datasets.py --all
    python src/load_extra_datasets.py --liar --covid
"""
import argparse
import os
import zipfile
import io

import pandas as pd
import requests

from src.preprocess import clean_text_classic

RAW_DIR = os.path.join("data", "raw")
PROCESSED_EXTRA_DIR = os.path.join("data", "processed", "extra")

UCSB_LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"

COVID_FILES = {
    "train": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Train.csv",
    "val": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/Constraint_Val.csv",
    "test": "https://raw.githubusercontent.com/diptamath/covid_fake_news/main/data/english_test_with_labels.csv",
}

FAKENEWSNET_FILES = {
    "politifact_fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv",
    "politifact_real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv",
    "gossipcop_fake": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv",
    "gossipcop_real": "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv",
}

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state_info", "party_affiliation", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context",
]


def _download(url: str, dest: str, timeout: int = 120) -> str:
    """Downloads a file to `dest` if it doesn't already exist. Returns dest."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


# ─────────────────────────────────────────────────────────────────────────
# LIAR (Wang, 2017)
# ─────────────────────────────────────────────────────────────────────────
def fetch_liar() -> pd.DataFrame:
    """
    Downloads the LIAR dataset directly from its original UCSB host (no auth
    needed) and collapses its 6 truth levels to binary, per the blueprint's
    pitfall note: "collapse to binary (true + mostly-true vs false +
    pants-fire, drop the middle labels)".
    """
    liar_dir = os.path.join(RAW_DIR, "liar")
    zip_path = os.path.join(liar_dir, "liar_dataset.zip")
    _download(UCSB_LIAR_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(liar_dir)

    frames = []
    for split, fname in [("train", "train.tsv"), ("val", "valid.tsv"), ("test", "test.tsv")]:
        df = pd.read_csv(os.path.join(liar_dir, fname), sep="\t", header=None, names=LIAR_COLUMNS)
        df["split"] = split
        frames.append(df)
    liar = pd.concat(frames, ignore_index=True)

    # Binary collapse: drop the ambiguous middle labels, keep the clear ends.
    label_map = {
        "true": 1, "mostly-true": 1,
        "false": 0, "pants-fire": 0,
    }
    liar = liar[liar["label"].isin(label_map)].copy()
    liar["orig_label"] = liar["label"]
    liar["content"] = liar["statement"].astype(str)
    liar["clean_text"] = liar["content"].apply(clean_text_classic)
    liar["label"] = liar["orig_label"].map(label_map).astype(int)

    out = liar[["content", "clean_text", "label", "orig_label", "split"]].dropna(subset=["clean_text"])
    out = out[out["clean_text"].str.len() > 0]

    os.makedirs(PROCESSED_EXTRA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_EXTRA_DIR, "liar_binary.csv")
    out.to_csv(out_path, index=False)
    print(f"[liar] {len(out)} rows (binary, middle labels dropped) -> {out_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────
# COVID-19 Fake News (Patwa et al., Constraint@AAAI 2021)
# ─────────────────────────────────────────────────────────────────────────
def fetch_covid_fake_news() -> pd.DataFrame:
    """
    Downloads the Constraint@AAAI2021 COVID-19 fake news shared-task CSVs
    (public GitHub mirror, no auth). Social-media posts, real/fake labelled.
    """
    covid_dir = os.path.join(RAW_DIR, "covid_fake_news")
    frames = []
    for split, url in COVID_FILES.items():
        dest = os.path.join(covid_dir, f"{split}.csv")
        _download(url, dest)
        df = pd.read_csv(dest)
        df["split"] = split
        frames.append(df)
    covid = pd.concat(frames, ignore_index=True)

    covid["label"] = covid["label"].str.lower().map({"real": 1, "fake": 0})
    covid["content"] = covid["tweet"].astype(str)
    covid["clean_text"] = covid["content"].apply(clean_text_classic)

    out = covid[["content", "clean_text", "label", "split"]].dropna(subset=["label", "clean_text"])
    out = out[out["clean_text"].str.len() > 0]

    os.makedirs(PROCESSED_EXTRA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_EXTRA_DIR, "covid_fake_news.csv")
    out.to_csv(out_path, index=False)
    print(f"[covid] {len(out)} rows -> {out_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────
# FakeNewsNet (Shu et al.) -- title-only cross-domain test
# ─────────────────────────────────────────────────────────────────────────
def fetch_fakenewsnet_titles() -> pd.DataFrame:
    """
    Downloads the FakeNewsNet GitHub CSV mirror. NOTE: this public mirror
    ships only `id, news_url, title, tweet_ids` -- the full article body is
    not included (FakeNewsNet's own tooling re-scrapes it from `news_url`,
    which is unreliable for dead links and needs no extra credentials but is
    slow and flaky). We use it as a title-only cross-domain generalization
    test, which lines up with the blueprint's own ablation experiment
    ("title-only vs title + text").
    """
    fnn_dir = os.path.join(RAW_DIR, "fakenewsnet")
    frames = []
    for name, url in FAKENEWSNET_FILES.items():
        dest = os.path.join(fnn_dir, f"{name}.csv")
        _download(url, dest, timeout=180)
        df = pd.read_csv(dest, usecols=["id", "news_url", "title"])
        domain, verdict = name.split("_")
        df["domain"] = domain
        df["label"] = 1 if verdict == "real" else 0
        frames.append(df)
    fnn = pd.concat(frames, ignore_index=True)

    fnn["content"] = fnn["title"].astype(str)
    fnn["clean_text"] = fnn["content"].apply(clean_text_classic)

    out = fnn[["content", "clean_text", "label", "domain"]].dropna(subset=["clean_text"])
    out = out[out["clean_text"].str.len() > 0]

    os.makedirs(PROCESSED_EXTRA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_EXTRA_DIR, "fakenewsnet_titles.csv")
    out.to_csv(out_path, index=False)
    print(f"[fakenewsnet] {len(out)} rows (title-only) -> {out_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────
# Manual-download datasets: Kaggle 2018, IFND, HinFakeNews
# ─────────────────────────────────────────────────────────────────────────
def require_kaggle_fake_news_2018() -> pd.DataFrame:
    """
    Kaggle competition 'fake-news' (2018). Requires a Kaggle account:
        1. pip install kaggle
        2. Place your kaggle.json API token in ~/.kaggle/kaggle.json
        3. Accept the competition rules at
           https://www.kaggle.com/competitions/fake-news/rules
        4. kaggle competitions download -c fake-news -p data/raw/kaggle_fake_news
        5. unzip data/raw/kaggle_fake_news/fake-news.zip -d data/raw/kaggle_fake_news
    Expects data/raw/kaggle_fake_news/train.csv with columns id,title,author,text,label
    (label: 1 = unreliable/fake, 0 = reliable/real).
    """
    path = os.path.join(RAW_DIR, "kaggle_fake_news", "train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Kaggle 'fake-news' (2018) dataset not found.\n"
            f"Expected file: {path}\n\n"
            "This dataset needs a Kaggle account and cannot be auto-downloaded.\n"
            "Steps:\n"
            "  1. pip install kaggle\n"
            "  2. Put your Kaggle API token at ~/.kaggle/kaggle.json\n"
            "  3. Accept rules: https://www.kaggle.com/competitions/fake-news/rules\n"
            "  4. kaggle competitions download -c fake-news -p data/raw/kaggle_fake_news\n"
            "  5. unzip data/raw/kaggle_fake_news/fake-news.zip -d data/raw/kaggle_fake_news\n"
        )
    df = pd.read_csv(path)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df["clean_text"] = df["content"].apply(clean_text_classic)
    df = df.rename(columns={"label": "label"})  # already 1=unreliable, 0=reliable
    out = df[["content", "clean_text", "label"]].dropna(subset=["clean_text"])
    os.makedirs(PROCESSED_EXTRA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_EXTRA_DIR, "kaggle_fake_news_2018.csv")
    out.to_csv(out_path, index=False)
    print(f"[kaggle-2018] {len(out)} rows -> {out_path}")
    return out


def require_ifnd() -> pd.DataFrame:
    """
    IFND (Sharma & Garg, 2021), Indian news dataset. Kaggle-mirrored, also
    needs a Kaggle account (same auth flow as above). Expects
    data/raw/ifnd/IFND.csv with a 'Statement'/'Label' style column pair --
    check the actual mirror's column names and adjust below once downloaded,
    since different Kaggle mirrors have renamed columns in the past.
    """
    path = os.path.join(RAW_DIR, "ifnd", "IFND.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "IFND dataset not found.\n"
            f"Expected file: {path}\n\n"
            "Search Kaggle for 'IFND fake news' (Sharma & Garg, 2021), download\n"
            "via the Kaggle API (same steps as the 2018 fake-news dataset) and\n"
            "place IFND.csv at the path above. Inspect its columns before reuse --\n"
            "different mirrors use different column names."
        )
    df = pd.read_csv(path)
    return df


def require_hinfakenews() -> pd.DataFrame:
    """
    HinFakeNews (Hindi). Needs IndiaAI AIKosh registration to download, and
    is Hindi-language text -- outside the English-only scope defined in the
    blueprint (Section 3). Only attempt this as the optional Hindi extension,
    with MuRIL/IndicBERT instead of the English TF-IDF/DistilBERT pipeline.
    """
    path = os.path.join(RAW_DIR, "hinfakenews", "HinFakeNews.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "HinFakeNews dataset not found.\n"
            f"Expected file: {path}\n\n"
            "Register and download from IndiaAI AIKosh:\n"
            "https://aikosh.indiaai.gov.in/home/datasets/details/hinfakenews_2.html\n"
            "Note: this is the optional Hindi extension (Section 7 of the blueprint) --\n"
            "it needs a MuRIL/IndicBERT tokenizer, not the English TF-IDF pipeline."
        )
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liar", action="store_true")
    parser.add_argument("--covid", action="store_true")
    parser.add_argument("--fakenewsnet", action="store_true")
    parser.add_argument("--all", action="store_true", help="Fetch all auto-downloadable datasets")
    args = parser.parse_args()

    if args.all or args.liar:
        fetch_liar()
    if args.all or args.covid:
        fetch_covid_fake_news()
    if args.all or args.fakenewsnet:
        fetch_fakenewsnet_titles()

    if not any([args.all, args.liar, args.covid, args.fakenewsnet]):
        parser.print_help()
