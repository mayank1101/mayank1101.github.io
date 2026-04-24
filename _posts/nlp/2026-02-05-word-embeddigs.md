---
layout: post
title: "Word Embeddings from Scratch: CBOW and Skip-Gram"
date: 2026-02-05
series: "NLP: Foundations to Advance"
series_author: "Mayank Sharma"
excerpt: "Learn how Word2Vec learns dense word vectors by implementing both CBOW and Skip-Gram from scratch in a single hands-on Jupyter notebook."
---

So here's the thing about Bag-of-Words and TF-IDF, they're solid starting points. They work, they're fast, and they've powered enough production systems that we all have scars from tuning their parameters. But there's this fundamental problem that bugs you after a while - they have zero semantic awareness.

When you represent words as one-hot vectors, `king`, `queen`, and `banana` are equally different from each other. Your model can technically count them, but it doesn't *know* that `king` and `queen` are kindred spirits while `banana` is just... hanging out alone.

This is where word embeddings come in, and honestly, this is where NLP stopped being just counting things and started being something closer to *understanding*.

**Word embeddings** represent each word as a dense vector of learned numbers. You don't hand-craft these, the network learns them by watching patterns in how words appear around each other. Words that show up in similar contexts gradually learn similar vectors. It's like the model is picking up on linguistic vibes without you explicitly telling it what those vibes are.

In this tutorial, we're building this from scratch. No frameworks, no magical libraries doing the heavy lifting behind the scenes. We'll implement the two classic architectures that started this whole embedding revolution:

- **CBOW (Continuous Bag of Words)** — predict the missing word from its neighbors
- **Skip-Gram** — predict the neighbors from the target word

## Table of Contents

1. [Why Word Embeddings Matter](#why-word-embeddings-matter)
2. [The Distributional Hypothesis](#the-distributional-hypothesis)
3. [From Sparse Vectors to Dense Semantics](#from-sparse-vectors-to-dense-semantics)
4. [CBOW: Predict the Center Word](#cbow-predict-the-center-word)
5. [Skip-Gram: Predict the Context](#skip-gram-predict-the-context)
6. [The Shared Word2Vec Pipeline](#the-shared-word2vec-pipeline)
7. [The Math (It's Simpler Than You'd Think)](#the-math-its-simpler-than-youd-think)
8. [CBOW vs Skip-Gram: Which Do You Actually Use?](#cbow-vs-skip-gram-which-do-you-actually-use)
9. [What Actually Works Well (And What Doesn't)](#what-actually-works-well-and-what-doesnt)
10. [Conclusion and What Comes Next](#conclusion-and-what-comes-next)

## Why Word Embeddings Matter

Let's say you're looking at: `the king loves the queen`

A TF-IDF model will happily count that `king` appears once and `queen` appears once. But ask it whether those two words are similar, and it'll shrug. In the math, they're just different columns in a huge sparse matrix.

Embeddings fix this by giving each word its own learned vector:

$$
\mathbf{v}_w \in \mathbb{R}^d
$$

Instead of living in a 50,000-dimensional one-hot space (where one dimension is 1 and the rest are 0), a word now has a compact representation in, say, 300 dimensions. And here's the clever part, those 300 dimensions actually *encode meaning*.

In a well-trained embedding space, something magical happens:
- Words that appear in similar contexts end up near each other
- You can measure word similarity with simple cosine distance
- Downstream models get semantically richer input than raw counts
- You can even do arithmetic: `king - man + woman ≈ queen` (the classic example)

This shift from manually counting things to *learning* what words mean from context is why embeddings were such a inflection point in NLP. It's the difference between having a spell-checker and having something that actually understands language.

## The Distributional Hypothesis

Word2Vec's foundation is surprisingly simple. It's based on something linguists figured out a while ago:

> You can understand what a word means by looking at what words show up around it.

This is the **distributional hypothesis**, and it's kind of obvious once you think about it. If you see:

- `the king rules the kingdom`
- `the queen rules the kingdom`  
- `the king and queen are royalty`

You notice that `king` and `queen` tend to appear near the same kinds of words: `rules`, `kingdom`, `royalty`. Your brain picks up on this pattern immediately and concludes that these words are probably related. They hang out with similar crowds.

Word2Vec exploits this idea by turning it into a game the network can play:

1. Slide a window across your text
2. Create mini prediction tasks: "here are the surrounding words, what goes in the middle?" or "here's the center word, what surrounds it?"
3. Train the network to get these predictions right
4. Watch as meaningful word representations emerge

The beautiful part? You never explicitly tell the model what words mean. You just keep feeding it context clues, and the meaning figures itself out. It's like learning a language by osmosis.

## From Sparse Vectors to Dense Semantics

Before embeddings, we had one-hot encoding. It looks deceptively clean:

Vocabulary: `["king", "queen", "man", "woman", "cat", "dog"]`

Representations:
- `king  -> [1, 0, 0, 0, 0, 0]`
- `queen -> [0, 1, 0, 0, 0, 0]`
- `cat   -> [0, 0, 0, 0, 1, 0]`

Great for indexing. Terrible for understanding. Every word is equally distant from every other word in this space. According to the math, `king` is just as different from `queen` as `king` is from `cat`. There's no structure, no meaning baked in.

Word2Vec flips this entirely. Instead of one-hot vectors, we learn an embedding matrix:

$$
\mathbf{W} \in \mathbb{R}^{V \times d}
$$

Where $V$ is your vocabulary size and $d$ is your embedding dimension (usually something like 100-300). Each row of this matrix is the learned vector for one word.

The core challenge becomes:

> How do we train this matrix so words that are semantically similar actually get similar vectors?

That's the problem CBOW and Skip-Gram solve, just coming at it from slightly different angles.

## CBOW: Predict the Center Word

**CBOW (Continuous Bag of Words)** is the simpler of the two approaches. It's basically a fill-in-the-blank game.

Take: `the king loves the queen`

If we treat `loves` as the target word and use a window of 2 words on each side:

- Context: `the`, `king`, `the`, `queen`
- Target: `loves`

CBOW learns:

$$
(\text{the}, \text{king}, \text{the}, \text{queen}) \rightarrow \text{loves}
$$

The network's job is simple: given these context words, predict what should go in the middle.

### The Mechanics

CBOW takes the embeddings of all the context words, averages them together (or sometimes sums), and uses that combined representation to predict the center word. It's elegant and efficient - you're getting multiple training signals with each prediction.

### Why This Works Well

CBOW is genuinely fast. Because it combines context words upfront, frequent words get stable representations quickly. You're effectively learning from multiple context examples in a single forward pass.

The tradeoff? CBOW smooths out fine details. When you average context, you're sort of losing information about word order and specific relationships. `the king loves` and `loves the king` produce roughly the same averaged context, which isn't always what you want.

## Skip-Gram: Predict the Context

**Skip-Gram** inverts the prediction direction. Instead of "given context, find the center word," it asks "given this word, what should appear around it?"

Same sentence: `the king loves the queen`

With `loves` as the center and window size 2, Skip-Gram creates four separate training examples:

- `loves -> the` (left context)
- `loves -> king` (left context)
- `loves -> the` (right context)
- `loves -> queen` (right context)

So for every word that appears in your text, you get multiple mini-prediction tasks scattered around it.

### The Power of Multiplicity

This generates way more training examples per sentence. If CBOW gives you one "guess the middle word" task per window, Skip-Gram gives you multiple "guess the neighbor" tasks from the same position.

Why does this matter? **Rare words**. If a rare word only appears a few times, CBOW might not see enough examples to learn a stable representation. But Skip-Gram turns each appearance into 2-4 prediction tasks, so even infrequent words get decent training signals.

The downside is obvious: more training examples means more computation. Skip-Gram is slower to train than CBOW. It's a classic engineering tradeoff — more data accuracy versus faster training.

## The Shared Word2Vec Pipeline

Despite their different architectures, CBOW and Skip-Gram follow basically the same workflow. Here's what actually happens:

1. **Preprocess** — tokenize, lowercase, maybe remove common words
2. **Build vocab** — figure out your word-to-ID mapping  
3. **Generate examples** — slide your context window across the text (this is where CBOW and Skip-Gram diverge)
4. **Initialize weights** — random embeddings for every word
5. **Train** — gradient descent to make correct predictions
6. **Extract embeddings** — keep the word representation matrix

Both models use two weight matrices:

$$
\mathbf{W}_1 \in \mathbb{R}^{V \times d}, \quad
\mathbf{W}_2 \in \mathbb{R}^{d \times V}
$$

Think of it like this:
- $\mathbf{W}_1$ is your word embedding table (maps words to vectors)
- $\mathbf{W}_2$ is your prediction layer (maps those vectors to probabilities)

When training is done, you throw away $\mathbf{W}_2$ and keep $\mathbf{W}_1$. Those learned word vectors are your final product — they're dense, semantic, and ready to feed into downstream tasks.

## The Math (It's Simpler Than You'd Think)

Word2Vec is honestly just a softmax classifier wrapped around context windows. Let me show you what's actually happening.

### CBOW's Calculation

You have context words at positions around $t$:

$$
w_{t-m}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+m}
$$

CBOW averages their embeddings:

$$
\mathbf{h} =
\frac{1}{2m}
\sum_{-m \leq j \leq m,\; j \neq 0}
\mathbf{v}_{w_{t+j}}
$$

Then uses that average as the input to a softmax prediction layer:

$$
P(w_t \mid \text{context}) =
\text{softmax}(\mathbf{W}_2^\top \mathbf{h})
$$

During training, you compute the loss (cross-entropy) and backprop to make the true center word have high probability.

### Skip-Gram's Calculation

Skip-Gram starts with the center word embedding and predicts each context position independently:

$$
P(w_{t+j} \mid w_t) =
\text{softmax}(\mathbf{W}_2^\top \mathbf{v}_{w_t})
$$

You do this for every $j$ in your window. The loss is the sum of log-probabilities for all real context words:

$$
\sum_{-m \leq j \leq m,\; j \neq 0}
\log P(w_{t+j} \mid w_t)
$$

## CBOW vs Skip-Gram: Which Do You Actually Use?

They're cousins, not clones. Here's the honest breakdown:

### Choose CBOW If...

- **Speed matters** — you want embeddings trained yesterday, not tomorrow
- **You have tons of common words** — CBOW stabilizes quickly on frequent vocabulary
- **Your use case tolerates smoothing** — you don't need fine-grained word-context relationships
- **Compute is tight** — you get more bang per training epoch

CBOW averages context, so it trains faster and converges with fewer examples. That efficiency buys you something real in a production setting.

### Choose Skip-Gram If...

- **Rare words are important** — you're working with specialized domains (medical, legal, code) where uncommon terms matter
- **You have compute to spare** — you don't mind 2-3x longer training for better quality
- **Local relationships matter** — you care about *how* words relate to each other, not just whether they're related
- **You want to capture nuance** — Skip-Gram's per-word predictions preserve more detail

Skip-Gram gives each context position its own prediction, so even infrequent words see multiple training signals.

### The Fundamental Difference

CBOW: "Given `the`, `king`, `the`, `queen`, what word goes here?"

Skip-Gram: "Given `loves`, what word appears here? And here? And here?"

It's a tiny change in perspective that cascades into different behavior. Same algorithm, different data flow.

## What Actually Works Well (And What Doesn't)

### The Real Wins

- **Semantic signals** — vectors actually capture meaning, not just word counts
- **Simplicity** — you can understand the entire algorithm in an afternoon
- **Reusability** — these vectors transfer to downstream tasks (classification, clustering, QA)
- **Leverage** — if you understand Word2Vec, GloVe and FastText make immediate sense
- **Interpretability** — the notebook lets you trace every decision

### The Real Limitations (Let's Be Honest About These)

- **Static representations** — one vector per word, regardless of context. `bank` in "river bank" vs "savings bank" gets the same vector. Later methods (ELMo, BERT) fixed this.
- **Rare word blindness** — if a word appears 3 times, you might learn garbage. It needs frequency to stabilize.
- **Corpus dependency** — small corpora produce noisy embeddings. You need at least hundreds of thousands of tokens for decent results.
- **Computational scaling** — full softmax becomes expensive at scale. That's why production systems use negative sampling.
- **Out-of-vocabulary words** — unseen words have no representation. You have to average or use random vectors, which is hacky.

These aren't bugs in Word2Vec. They're exactly the problems that motivated the next generation of methods. Contextual embeddings, subword tokenization, and pre-training emerged because people kept bumping into these walls.

## Why This Matters (And What Comes Next)

Word2Vec's genius is its simplicity. One core insight:

> Words that hang out together in text learn similar vectors.

That's it. No hand-crafted features, no linguistic rules, no expensive annotations. Just context clues and gradient descent.

CBOW and Skip-Gram are two flavors of the same idea:
- **CBOW** — context → center word (fast, stable)
- **Skip-Gram** — center word → context (thorough, detailed)

### But There Are Obvious Next Questions

This is where the story gets interesting. After implementing Word2Vec, you realize:

- **Global statistics** — shouldn't we use *all* word co-occurrences, not just local windows? (This leads to **GloVe**)
- **Rare words** — we're basically throwing away information from words that don't appear often. Can we be smarter? (This leads to **FastText** and subword models)
- **Context sensitivity** — one vector per word is too rigid. What if we computed embeddings *in context*? (This leads to **ELMo**, **BERT**, and the entire pre-training era)

Those are the natural progressions, and they're all built on understanding Word2Vec first. You're now positioned to understand why later methods were invented and what problems they solve.
