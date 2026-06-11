---
layout: post
title: "Sentiment Analysis from Scratch: Classifying Text with Bag-of-Words and Logistic Regression"
date: 2026-04-22
series: "NLP Series"
series_author: "Mayank Sharma"
excerpt: "Learn how sentiment analysis works end to end by turning raw text into Bag-of-Words features and training a logistic regression classifier from scratch in a Jupyter notebook."
---

Here's the honest reality of production: users leave opinions everywhere. In reviews, tweets, support tickets, app store feedback, survey responses. And companies desperately want to know what those opinions are.

When a user writes:
- `this movie was fantastic`
- `the service was painfully slow`
- `the update is okay, but still buggy`

They're not just producing text. They're expressing measurable opinion that can inform business decisions.

**Sentiment analysis** is the task of teaching a model to extract that. At its simplest, it asks:

> Given this text, is the opinion positive, negative, or neutral?

Sounds straightforward. And it is, until the language gets real.

**The messy reality:**
- Sarcasm: "yeah, *great* customer service" (sarcasm detector: humans, not NLP)
- Negation: "not good" completely flips "good"
- Domain drift: "cheap" is positive for goods, negative for build quality
- Mixed sentiment: "great product but terrible support" — both sentiments in one review
- Implicit emotion: "I expected more" doesn't mention good/bad explicitly

That's why every production sentiment system needs a strong baseline first. Not to ship it, but to understand the problem clearly.

This tutorial builds exactly that: a complete, interpretable foundation using:
- Text preprocessing (keeping what matters, removing noise)
- Bag-of-Words features (simple but surprisingly useful)
- Logistic regression trained from scratch (linear, interpretable, fast)
- Proper evaluation (accuracy is misleading; we need precision/recall/F1)

## Table of Contents

1. [What Sentiment Analysis Actually Solves](#what-sentiment-analysis-actually-solves)
2. [Why This Problem Is Harder Than It Looks](#why-this-problem-is-harder-than-it-looks)
3. [The Classical Sentiment Analysis Pipeline](#the-classical-sentiment-analysis-pipeline)
4. [Preprocessing Text for Classification](#preprocessing-text-for-classification)
5. [From Sentences to Numbers](#from-sentences-to-numbers)
6. [Bag-of-Words Intuition](#bag-of-words-intuition)
7. [Logistic Regression for Sentiment Classification](#logistic-regression-for-sentiment-classification)
8. [Loss Function and Gradient Descent](#loss-function-and-gradient-descent)
9. [How We Evaluate a Sentiment Model](#how-we-evaluate-a-sentiment-model)
10. [What the Notebook Builds](#what-the-notebook-builds)
11. [Strengths and Limitations of This Baseline](#strengths-and-limitations-of-this-baseline)
12. [Conclusion and What Comes Next](#conclusion-and-what-comes-next)

## What Sentiment Analysis Actually Solves

Sentiment analysis is **text classification**. You have text, you want to label it with a category. In this case, the category is opinion.

Common formulations in the wild:
- **Binary**: positive vs negative (simplest, what we do here)
- **Three-way**: positive vs neutral vs negative (more nuanced)
- **Fine-grained**: 1-5 star ratings (ordered labels)
- **Aspect-based**: different sentiments for different aspects of the same review (hard, useful)

We use binary classification:

$$
y \in \{0, 1\}
$$

Where $y=1$ is positive and $y=0$ is negative. It's the cleanest way to see the mechanics.

### Why This Matters in Practice

Sentiment systems show up because companies actually need them:
- **Product teams** monitor reviews to catch systemic issues (tons of "battery sucks" = problem)
- **Support teams** use it to triage tickets (escalate angry customers faster)
- **Brand teams** track reputation shifts across social media
- **Analysts** mine survey responses without reading thousands of free-text answers
- **Researchers** analyze feedback at scale

The pure value proposition: convert thousands of unstructured opinions into one number you can act on. That number shapes product decisions, customer experience, and sometimes company strategy.

So this isn't theoretical. It's solving a real business problem.

## Why This Problem Is Harder Than It Looks (And Why That Matters)

Humans read "this phone is good" and instantly know positive sentiment. Models can't do that without us being explicit about the representation.

Two words change everything:
- `this phone is good` → positive
- `this phone is not good` → negative

One word flipped the sentiment completely. So sentiment analysis isn't just counting positive vs negative words. It's understanding how they interact.

**The really hard cases:**
- `great camera, terrible battery` (mixed: which dominates? depends on context)
- `I expected more` (implicit: no emotion words, but clearly disappointed)
- `thanks for the amazing customer support` (sarcasm: if the support was actually bad)
- `the app is intuitive` (domain shift: intuitive is good for software, irrelevant for food)

A Bag-of-Words baseline won't handle all of these. It genuinely can't. It'll struggle with negation, miss sarcasm, and overlook word order. 

**So why build it?** Because:
1. It teaches you the full supervised learning pipeline
2. It often works better than you'd expect on real data
3. It forces you to understand what embeddings and neural nets are *fixing*
4. Every production system starts with this baseline

A good baseline isn't about perfection. It's about having something defensible to improve from.

## The Classical Sentiment Analysis Pipeline

A practical classical workflow usually looks like this:

1. collect labeled text examples,
2. clean and tokenize the text,
3. build a vocabulary,
4. convert each sentence into a numeric feature vector,
5. train a classifier,
6. evaluate on held-out data,
7. inspect errors and iterate.

This pipeline is simple enough to implement from scratch and strong enough to work surprisingly well on many small and medium datasets.

In this module, we use **Bag-of-Words** for the representation and **logistic regression** for the classifier.

That choice is deliberate:

- Bag-of-Words makes feature construction transparent,
- logistic regression gives probabilities and interpretable weights,
- the whole system can be trained in a few lines of NumPy once the math is clear.

## Preprocessing Text for Classification

Raw text is messy. Before we train anything, we usually apply a few normalization steps.

Typical preprocessing steps include:

- lowercasing,
- removing unnecessary punctuation,
- tokenization,
- optionally removing stop words,
- optionally stemming or lemmatization.

For an educational sentiment baseline, simpler is often better. Over-cleaning can sometimes remove useful signals.

For example, punctuation can carry sentiment:

- `good`
- `good!!!`

Likewise, negation words such as `not`, `never`, and `no` are crucial. A preprocessing pipeline that removes them blindly will damage the classifier.

That is why the notebook uses a compact and explicit preprocessing function rather than an aggressive generic pipeline.

## From Sentences to Numbers

A classifier cannot learn directly from raw strings. We need to convert each sentence into a vector of numbers.

This is the feature engineering step.

There are many choices:

- one-hot token indicators,
- Bag-of-Words counts,
- TF-IDF,
- averaged word embeddings,
- learned sequence encoders,
- transformer embeddings.

To build intuition, we start with **Bag-of-Words**.

## Bag-of-Words: Dead Simple But It Works

Vocabulary: `["amazing", "bad", "boring", "great", "love", "slow"]`

Review 1: `love this great movie`
→ `[0, 0, 0, 1, 1, 0]` (1 "love", 1 "great", others are 0)

Review 2: `bad slow boring`
→ `[0, 1, 1, 0, 0, 1]` (1 "bad", 1 "boring", 1 "slow")

That's it. Count how many times each vocabulary word appears. Ignore word order completely.

**Yes, you lose information.** Word order matters. Grammar matters. But here's what's surprising: for sentiment, you lose less than you'd think.

### Why This Actually Works

Sentiment tasks have a lot of **lexical signal**. The vocabulary itself carries most of the meaning:
- Positive-leaning words: `great`, `excellent`, `love`, `amazing`, `fantastic`
- Negative-leaning words: `bad`, `terrible`, `disappointing`, `weak`, `waste`

A logistic regression model can learn: "when I see 'great' and 'love', push the prediction toward positive. When I see 'bad' and 'disappointing', push it toward negative."

It's not subtle. It's not handling negation or sarcasm. But for many real reviews, it's honest work. And it's fast, interpretable, and actually useful.

## Logistic Regression for Sentiment Classification

Once each review is a feature vector $\mathbf{x} \in \mathbb{R}^d$, we can train a binary classifier.

Logistic regression computes:

$$
z = \mathbf{w}^\top \mathbf{x} + b
$$

and then turns that raw score into a probability using the sigmoid function:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

This gives:

$$
P(y = 1 \mid \mathbf{x}) = \hat{y}
$$

Interpretation:

- if $\hat{y}$ is close to 1, the model thinks the review is positive,
- if $\hat{y}$ is close to 0, the model thinks the review is negative.

### Decision Rule

For binary classification, a standard decision threshold is:

$$
\hat{y}_{class} =
\begin{cases}
1 & \text{if } \hat{y} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

The model is linear in the feature space. That means each feature gets a weight:

- large positive weight: evidence for positive sentiment,
- large negative weight: evidence for negative sentiment.

This is one of the best reasons to use logistic regression in an educational tutorial: it is easy to inspect what the model learned.

## Loss Function and Gradient Descent

To train logistic regression, we need a loss that penalizes wrong probabilities.

For one example, the **binary cross-entropy loss** is:

$$
\mathcal{L}(y, \hat{y}) =
- \left[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

For a dataset of $N$ examples:

$$
J(\mathbf{w}, b) =
- \frac{1}{N}\sum_{i=1}^{N}
\left[
y^{(i)} \log(\hat{y}^{(i)}) +
\left(1-y^{(i)}\right)\log\left(1-\hat{y}^{(i)}\right)
\right]
$$

Gradient descent updates the parameters to reduce this loss:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial J}{\partial \mathbf{w}}
$$

$$
b \leftarrow b - \eta \frac{\partial J}{\partial b}
$$

where $\eta$ is the learning rate.

The notebook implements these updates directly in NumPy so the training loop remains transparent.

## How We Evaluate a Sentiment Model

Accuracy alone is often not enough. A sentiment model can look strong on accuracy while hiding poor behavior on one class.

That is why the notebook computes:

- **accuracy**,
- **precision**,
- **recall**,
- **F1 score**,
- **confusion matrix**.

### Accuracy

Accuracy is the fraction of correct predictions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

### Precision

Of the examples predicted as positive, how many were actually positive?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall

Of the truly positive examples, how many did the model recover?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Score

The harmonic mean of precision and recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

These metrics make evaluation less misleading, especially when classes are imbalanced.

## What the Notebook Builds

Everything for this tutorial lives in:

- [`jupyter/sentiment_analysis_tutorial.ipynb`](./jupyter/sentiment_analysis_tutorial.ipynb)

The notebook is organized as a full learning workflow.

### 1. Small Labeled Review Dataset

We start with a compact handcrafted dataset of positive and negative reviews. It is intentionally small enough to inspect manually while still large enough to make training meaningful.

### 2. Text Preprocessing and Tokenization

The notebook lowercases text, normalizes punctuation, tokenizes sentences, and preserves negation terms so the model does not lose crucial information.

### 3. Vocabulary and Bag-of-Words Encoding

Each training sentence is converted into a numeric vector using a Bag-of-Words representation built from the dataset vocabulary.

### 4. Logistic Regression from Scratch

The classifier is implemented directly with:

- parameter initialization,
- sigmoid activation,
- binary cross-entropy loss,
- gradient descent updates,
- probability and class prediction functions.

### 5. Evaluation and Interpretation

The notebook reports classification metrics, inspects common mistakes, and prints the most positive and most negative words according to the learned weights.

That last step is especially useful because it turns the model from a black box into something inspectable.

## Real Talk: What This Baseline Actually Does and Doesn't Do

### What It Does Well

- **Fast** — trains in seconds on modern hardware
- **Interpretable** — you can literally read the learned weights and know why it predicted what it did
- **Works for many real cases** — especially short, direct reviews
- **Minimal code** — you can implement it from scratch yourself
- **Educational** — forces you through the entire supervised learning loop

### What It Definitely Doesn't Do

- **Handle negation well** — "not good" looks like a negative review (low weight on "not", negative weight on "good")
- **Catch sarcasm** — "thanks for the *amazing* service" (when service was terrible) is just positive words
- **Understand word order** — `the movie was bad` and `bad the movie was` get identical vectors
- **Generalize to new vocabulary** — unseen words are just ignored
- **Compete with modern models** — fine-tuned transformers beat this on hard datasets

**The key insight:** These aren't bugs. They're features. Each limitation is exactly why the next method was invented. Can't handle negation with Bag-of-Words? That's why we invented RNNs. Still not enough? Transformers. Not generalizing to unseen words? FastText subword embeddings. Each step solves a real problem.

This baseline teaches you the landscape of NLP, not the final answer.

## Why This Tutorial Matters

Sentiment analysis is one of the rare NLP problems where the full supervised learning pipeline is transparent *and* the problem is real. You have to:

1. Get data (the dataset is right there)
2. Clean it (preprocessing)
3. Engineer features (Bag-of-Words)
4. Train a model (logistic regression, gradient descent)
5. Evaluate honestly (not just accuracy)
6. Inspect errors (what did the model learn?)

And here's the thing: this isn't historical. Every production NLP system, even the fancy ones with transformers, still does these same steps. The representation got fancier. The model got larger. But the structure is identical.

**The durable lesson isn't about Bag-of-Words or logistic regression.** It's this:

> Sentiment analysis is learning to map raw language into measurable signals. Everything else is implementation detail.

You understand the core? You can learn any new representation. RNNs, transformers, attention — they're all solving the same problem: better features, better training, better evaluation.

### Natural Next Steps

From here, obvious upgrades:
- **TF-IDF** instead of counts (weight rare words more)
- **Word embeddings** instead of one-hot (capture semantic similarity)
- **RNNs/LSTMs** instead of linear models (actually process word order)
- **Attention + Transformers** (best current approach, but conceptually similar to everything here)

Each one solves a specific problem the Bag-of-Words baseline couldn't. Once you understand this baseline, each upgrade makes sense because you know what it was fixing.
