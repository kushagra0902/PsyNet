# MBTI Personality Type Prediction from Text

Can we figure out someone's personality type just from how they write? This project tries to answer that by building a machine learning pipeline that reads a person's text and predicts their **MBTI personality type** (like INFP, ENTJ, etc.).

Instead of trying to guess one of 16 types directly (which doesn't work well because some types are way more common than others in the dataset), we break it down into **4 yes-or-no questions** — one for each personality axis. That turns a messy 16-class problem into 4 cleaner binary classification tasks.

---

## What's MBTI?

The Myers-Briggs Type Indicator sorts people along 4 dimensions:

| Axis | One side | Other side |
|------|----------|------------|
| E/I | Extraversion | Introversion |
| S/N | Sensing | iNtuition |
| T/F | Thinking | Feeling |
| J/P | Judging | Perceiving |

Each person gets a 4-letter type (e.g., INFP = Introverted + iNtuitive + Feeling + Perceiving). There are 16 possible combinations.

---

## The Dataset

We used the [MBTI Personality Type Dataset from Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type), which has **8,675 users** from the PersonalityCafe forum. Each row has the person's MBTI type and their last 50 forum posts combined into one big text blob (separated by `|||`).

One catch — the dataset is *really* unbalanced. INFP makes up 21% of all users, while ESFJ is just 0.5%. That's roughly a 47:1 ratio. We had to deal with this throughout the project using SMOTE and class weights.

---

## How the Project is Organized

```
Prml Project/
│
├── phase1_eda.py / .ipynb              # Step 1 – Look at the data, find patterns
├── phase2_preprocessing.py / .ipynb    # Step 2 – Clean up the text, prevent cheating
├── phase3_baseline.py / .ipynb         # Step 3 – First models (Logistic Regression)
├── phase4_advanced.py / .ipynb         # Step 4 – Try fancier models (XGBoost, LightGBM)
├── phase5_inference.py / .ipynb        # Step 5 – Prediction function + web app
│
├── mbti_1.csv                          # Original dataset from Kaggle
├── mbti_cleaned.csv                    # Cleaned version (output of Step 2)
│
├── models/                             # Saved trained models (.pkl files)
│   ├── tfidf_vectorizer.pkl
│   ├── lr_models.pkl
│   └── training_metadata.pkl
│
└── *.png                               # Various charts and plots
```

Both `.py` scripts and `.ipynb` notebooks are provided — they contain the same code, just in different formats. The notebooks come pre-executed so you can see outputs without re-running.

---

## What Each Phase Does

### Phase 1 — Exploratory Data Analysis
We start by just looking at the data. How many users per type? How long are the posts? Which dimensions are the most imbalanced? This step doesn't build any models — it's all about understanding what we're working with.

### Phase 2 — Text Preprocessing
This is where we clean up the raw text. We strip out URLs, emails, and special characters, remove stopwords, and lemmatize everything.

The trickiest part here was **preventing data leakage**. People on this forum love saying stuff like "As an INFP, I think..." — if we leave those tokens in, the model just learns to pattern-match the self-reported type labels instead of actually learning writing style differences. So we built a triple-pass filter that removes all MBTI type names, cognitive function codes (like Ni, Fe, Ti), and related terms.

### Phase 3 — Baseline Models
We convert the cleaned text into numerical features using **TF-IDF** (5,000 features, both unigrams and bigrams), then train a **Logistic Regression** model for each of the 4 axes. Nothing fancy here — just a solid baseline to benchmark against.

### Phase 4 — Advanced Models
Here we tried to beat the baseline using **XGBoost** and **LightGBM** with hyperparameter tuning (RandomizedSearchCV). We also used **SMOTE** to generate synthetic minority samples for the Logistic Regression model.

Spoiler: the simpler LR model actually won on every axis. More on that below.

### Phase 5 — Inference Pipeline
This wraps everything up into a usable tool. There's a `predict_mbti()` function that takes raw text, cleans it, vectorizes it, and runs it through the 4 trained models to spit out a 4-letter MBTI type. We also built a simple **Gradio web app** so you can type in text and get predictions in the browser.

---

## Results

Here's how the three models compared (using **macro-average F1 score** — we went with this over accuracy because accuracy is misleading when classes are unbalanced):

| Axis | LR + SMOTE | XGBoost | LightGBM | Winner |
|------|-----------|---------|----------|--------|
| E/I | **0.6727** | 0.4991 | 0.5350 | LR + SMOTE |
| S/N | **0.6409** | 0.4666 | 0.5121 | LR + SMOTE |
| T/F | **0.7924** | 0.7374 | 0.7685 | LR + SMOTE |
| J/P | **0.6256** | 0.5519 | 0.5869 | LR + SMOTE |
| **Avg** | **0.6829** | 0.5637 | 0.6006 | **LR + SMOTE** |

### What we learned

- **Logistic Regression won everything.** Sounds surprising, but it makes sense — TF-IDF features are sparse and high-dimensional (5,000 columns, mostly zeros). Linear models handle that kind of data well. Tree-based models like XGBoost prefer denser features and typically need more data to shine on NLP tasks.

- **Thinking/Feeling was the easiest axis to predict** (F1 ~ 0.79). The other three were harder, especially S/N where the dataset is 86% iNtuitive types.

- **LightGBM consistently outperformed XGBoost** and trained about 2.5x faster, though neither beat plain old Logistic Regression here.

- **Accuracy would've been misleading.** A model that just predicts "Introvert" for everyone would be 77% accurate on E/I. Macro-F1 forces the model to actually learn both classes.

---

## Package Requirements

You will need Python 3.10+ (tested on 3.12) and the following libraries:

- `pandas` & `numpy` (Data handling)
- `scikit-learn` (TF-IDF, Logistic Regression)
- `imbalanced-learn` (SMOTE oversampling)
- `xgboost` & `lightgbm` (Gradient Boosted Trees)
- `nltk` (Text preprocessing)
- `matplotlib` & `seaborn` (Visualizations)
- `gradio` (Web interface)

You can install all dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm nltk gradio
```

## Run Instructions

Follow these steps to set up and execute the project:

### Step 1: Download NLTK Data
Before running anything, download the necessary Natural Language Toolkit datasets for text cleaning:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Step 2: Run the Pipeline (Scripts or Notebooks)
The project is split into 5 phases. You can run them locally via terminal or open the corresponding `.ipynb` notebooks. *Note: Running them in order saves intermediate files to disk.*

```bash
# Phase 1: Exploratory Data Analysis (Generates plots & stats)
python phase1_eda.py              

# Phase 2: Text Preprocessing (Cleans text, saves mbti_cleaned.csv)
python phase2_preprocessing.py    

# Phase 3: Baseline Models (Trains Logistic Regression models)
python phase3_baseline.py         

# Phase 4: Advanced Models (Trains XGBoost/LightGBM. Note: Takes ~60 mins)
python phase4_advanced.py         

# Phase 5: Inference Pipeline (Trains final models and saves them)
python phase5_inference.py        
```
*(Alternatively, simply open the Jupyter notebooks `*.ipynb` in JupyterLab/VS Code to view the pre-calculated outputs.)*

### Step 3: Launch the Web Interface
To test the models interactively, run Phase 5, which also launches a local web server:

```bash
python phase5_inference.py
```
Then, open **http://localhost:7860** in your web browser to use the Gradio UI. You can type in any text and see the predicted MBTI type in real-time.

The Gradio UI lets you type or paste any text and get an instant MBTI prediction with per-axis confidence scores. There are also some example texts you can click to try it out.

---

## Why We Made Certain Choices

**Why 4 binary models instead of one 16-class model?**
The 16-type distribution is extremely skewed (47:1 ratio). Splitting into 4 binary problems makes each one more manageable and lets us optimize each axis independently.

**Why strip MBTI labels from the text?**
Forum users constantly mention their own type ("As an INFP..."). Without removing these, the model would just memorize the labels instead of learning real writing patterns. We run three passes of filtering to make sure nothing slips through.

**Why did Logistic Regression beat the tree models?**
TF-IDF gives us a very sparse, high-dimensional feature space. Linear models are well-suited for this kind of data. Gradient boosted trees typically need denser representations (like word embeddings) or larger datasets to outperform simpler models on text tasks.

**Why macro-F1 and not accuracy?**
With 77% of users being Introverts, a dummy model that always predicts "I" would score 77% accuracy. Macro-F1 gives equal weight to both classes, so the model actually has to learn the minority class too.

---

## Tech Stack

- **Python 3.12** — core language
- **scikit-learn** — TF-IDF, Logistic Regression, evaluation metrics
- **XGBoost / LightGBM** — gradient boosted tree models
- **imbalanced-learn** — SMOTE oversampling
- **NLTK** — text preprocessing (stopwords, lemmatization)
- **Matplotlib / Seaborn** — charts and plots
- **Gradio** — web interface for predictions
- **Pandas / NumPy** — data handling
