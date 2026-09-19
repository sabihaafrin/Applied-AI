# Applied AI - Financial Sentiment Analysis

Coursework project for the Applied AI module of my MSc in Artificial Intelligence and Data Science at the University of Hull. The project classifies financial text as negative, neutral, or positive, and compares four approaches to the task, from classical machine learning baselines to a domain-specific transformer.

## Problem

Sentiment in financial language often depends on context rather than individual words (for example "profit declined" versus "decline in costs"). The goal is to build and fairly compare models that can pick up these signals, on a dataset where the classes are imbalanced (neutral statements dominate and negative statements are rare). Because of this imbalance, models are evaluated on macro-averaged F1, which weights all three classes equally regardless of how many examples each has.

## Dataset

A combined corpus built by merging two standard financial sentiment datasets, FiQA and the Financial PhraseBank, into a single labelled CSV of short financial statements.

## Approach

The notebook works end to end:

**Exploratory data analysis** - class distribution, and a word cloud of the corpus.

**Text preprocessing** - lowercasing, URL and HTML removal, preserving tickers and monetary values, and a stopword strategy that deliberately keeps financial negation words (increase, decrease, up, down). Bigrams and trigrams are used to retain phrases such as "stock_market" and "interest_rate_hike" that carry stronger signal than single tokens.

**Handling class imbalance** - random oversampling of the minority classes, applied consistently so that every model trains on the same balanced set and the comparison stays fair.

**Models compared:**
- Logistic Regression on TF-IDF features (baseline), tuned with GridSearchCV
- Linear Support Vector Machine on TF-IDF features, tuned with GridSearchCV
- Bidirectional LSTM (Keras / TensorFlow) trained on raw sequences
- FinBERT, a finance-specific transformer, fine-tuned with the Hugging Face Trainer API

**Evaluation** - classification reports, per-class F1 comparison across the four models, confusion matrices, and an overall accuracy versus macro-F1 comparison.

## Running the notebook

Open the notebook in Jupyter or a GPU-enabled environment such as Google Colab (recommended for the deep learning and FinBERT sections). Key libraries used:

```
pandas, numpy, scikit-learn, imbalanced-learn, gensim, nltk,
tensorflow / keras, transformers, datasets, wordcloud, matplotlib
```

## Skills demonstrated

Natural language processing, domain-aware text preprocessing, TF-IDF feature engineering, handling imbalanced classes, classical ML (Logistic Regression, SVM) with hyperparameter tuning, deep learning with a bidirectional LSTM, transformer fine-tuning (FinBERT), and model evaluation and comparison using macro-F1 and confusion matrices.
