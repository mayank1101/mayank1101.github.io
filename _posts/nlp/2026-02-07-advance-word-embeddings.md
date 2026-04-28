---
layout: post
title: "Advanced Word Embeddings from Scratch: GloVe and FastText"
date: 2026-02-07
series: "NLP: Foundations to Advance"
series_author: "Mayank Sharma"
excerpt: "Learn how advanced embeddings go beyond Word2Vec by implementing a toy GloVe model and a FastText-style subword model from scratch in a Jupyter notebook."
---

So you built Word2Vec. You trained it, you saw vectors cluster by meaning, you probably felt pretty clever. And then you deployed it on real data and things started breaking.

The core Word2Vec insight is still rock solid:

> words that appear together tend to mean similar things.

But reality introduces two problems that local context windows just can't fix.

**Problem 1: You're Only Seeing Local Patterns**

Word2Vec slides a small window across your text. It sees that `king` hangs out near `queen`, `palace`, and `royalty`. But it's learning this one window at a time. What if you asked a bigger question: "Across my entire corpus, what are the global patterns in how words appear near each other?" That's a different signal entirely, and it's what GloVe tries to capture.

**Problem 2: Your Vocabulary Is Incomplete**

You'll encounter words at inference time that didn't exist in your training data. `learnable`. `microservice`. A typo like `occurence` instead of `occurrence`. A standard embedding model just shrugs, gives you a unknown word, random vector, sorry. But humans don't work that way. We look at word pieces. We recognize that `learnable` is built from `learn` + `able`, and we can make an educated guess about what it means.

This module tackles both problems:

- **GloVe** — leverage the full co-occurrence structure of your corpus
- **FastText** — build word vectors from subword pieces for better generalization


## Table of Contents

1. [Where Word2Vec Hits Its Limits](#where-word2vec-hits-its-limits)
2. [GloVe: Actually Using the Full Dataset](#glove-actually-using-the-full-dataset)
3. [FastText: Subword Structure Is a Superpower](#fasttext-subword-structure-is-a-superpower)
4. [The Math (Simpler Than It Looks)](#the-math-simpler-than-it-looks)
5. [GloVe vs FastText: Which Problem Are You Solving?](#glove-vs-fasttext-which-problem-are-you-solving)
6. [The Big Picture](#the-big-picture)

## Where Word2Vec Hits Its Limits

Word2Vec is genuinely solid, but it makes some assumptions that break down in practice. They're fine for learning, they're fine for small projects, but they're restrictive once you hit real data.

### Problem 1: Local Windows Aren't Everything

Word2Vec predicts one word from its neighbors, or vice versa. That's great, because nearby words *are* useful. But you're only learning from local signals.

What if you wanted to ask: "What's the global pattern here?" Like, "Which words consistently hang out together across my entire corpus?" That's different information. You can count co-occurrence pairs across the whole dataset and ask: "What embedding space would explain all these co-occurrences?"

That's the insight behind GloVe.

### Problem 2: Closed Vocabulary Is Brittle

Here's where Word2Vec falls apart in practice: you encounter a word at test time that wasn't in training.

A typo. A new product name. An inflected form. An uncommon word in a specialized domain. Standard embeddings go - "Dunno. Have a random vector. Good luck."

Humans don't work that way. When you see `unhappiness`, you don't need it defined separately. You recognize `un-` and `-ness` and `happy` and you can infer something useful. That compositional property is powerful.

FastText exploits this. Instead of treating words as atomic units, it builds them from character n-grams. So even if `learnable` is new, it's built from pieces the model has seen: `lea`, `ear`, `arn`, `rni`, `nin`, `ing`. That's not a perfect representation, but it's way better than random noise.

## GloVe: Actually Using the Full Dataset

**GloVe** stands for **Global Vectors for Word Representation**, and it's basically Word2Vec's more statistically sophisticated cousin.

Here's the core insight: instead of predicting one word from its neighbors one window at a time, count *everything*. Build a co-occurrence matrix where entry $X_{ij}$ is "how many times does word $j$ appear near word $i$ in my whole corpus?"

Now you have global statistics. The question becomes: "Can I find word vectors whose geometry explains these global patterns?"

Mathematically, GloVe tries to learn vectors $\mathbf{w}_i$ and $\tilde{\mathbf{w}}_j$ so that:

$$
\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log(X_{ij})
$$

Think of it as a regression problem: the dot product of two word vectors should approximate the log of how often they co-occur.

### Why This Actually Works

You've got two different approaches in NLP history:

**Old approach:** Count co-occurrence explicitly, produce massive sparse vectors. Fast but huge.

**Neural approach:** Predict locally, learn dense vectors. Compact but ignoring the big picture.

GloVe says: "Why choose? Use the global counts as our training signal, but learn compact dense vectors."

Here's a concrete example: `king` and `queen` both show up near `royalty`, `palace`, `throne`, `crown`. Their rows in the co-occurrence matrix look similar. So GloVe learns vectors that put `king` and `queen` near each other — not because we told it to, but because they actually *do* show up in similar contexts across the entire dataset.

And because GloVe is trying to explain many relationships at once (not just one window at a time), it naturally produces vectors with geometric structure. The famous `king - man + woman ≈ queen` property emerges from this.

## FastText: Subword Structure Is a Superpower

If GloVe fixes the "how to use corpus statistics" problem, **FastText** fixes the "unknown words" problem by asking a different question.

What if words aren't atoms? What if you could build them from smaller pieces?

Instead of treating `learning` as one indivisible vocabulary entry, FastText breaks it into character n-grams:

```
lea, ear, arn, rni, nin, ing
```

Each of these gets a learned embedding. The word vector for `learning` is just the average of these piece vectors.

### The Payoff

You've trained on `learn`, `learned`, `learner`. Now at test time, you see `learnable`.

Standard approach: "Never seen it. Random vector for unknown words. Sorry."

FastText approach: "I haven't seen this exact word, but I've seen `lea`, `ear`, `arn`, `ing`. I can average their vectors and get something meaningful."

It's not perfect, but it's *vastly* better than random noise. And the beauty is that this happens automatically, no special OOV handling code required.

This matters in real scenarios:

- **Morphology** — languages with complex inflections (German, Finnish, etc.)
- **Rare words** — you'll hit words not in your training data
- **Typos** — the user won't always spell correctly
- **Domain shift** — your training data misses technical terms from your domain
- **Product names** — `CoolNewFeature2024` wasn't in your training set
- **Code/data** — identifiers, variable names, camelCase

FastText isn't magic, but it's the right kind of practical generalization.

## The Math (Simpler Than It Looks)

### GloVe in Equations

GloVe minimizes this loss function:

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log(X_{ij})\right)^2
$$

Breaking this down:
- $X_{ij}$ is how often word $j$ appears near word $i$ (from your co-occurrence matrix)
- $\mathbf{w}_i$ and $\tilde{\mathbf{w}}_j$ are the word vectors we're learning
- $\log(X_{ij})$ is what we're trying to predict
- $f(X_{ij})$ is a weighting function that prevents rare pairs from dominating or frequent pairs from taking over

The key insight here is, we're not predicting one word at a time. We're trying to make the dot product of two word vectors approximately equal to the log of their co-occurrence count. One loss function for the entire corpus pattern.

### FastText Composition

For a word, FastText builds its vector by averaging its character n-gram vectors:

$$
\mathbf{v}_w = \frac{1}{|G_w|} \sum_{g \in G_w} \mathbf{z}_g
$$

Where:
- $G_w$ is the set of character n-grams in word $w$
- $\mathbf{z}_g$ is the learned embedding for each n-gram

So when you encounter `learnable` at test time, even though you've never trained on it directly, you average together the embeddings for `lea`, `ear`, `arn`, `rni`, `nin`, `ing` and get a reasonable vector. That's why it generalizes to unseen words.

## GloVe vs FastText: Which Problem Are You Solving?

These two methods tackle totally different problems, even though people often lump them together as "improvements over Word2Vec."

### When GloVe Is the Right Call

GloVe shines when:
- You have a big, stable corpus and want to squeeze meaning from global patterns
- Your vocabulary is mostly known (you're not handling lots of new words)
- You care about stable, reproducible semantic geometry
- You want theoretical elegance — "explain the co-occurrence patterns"

Example: You're building embeddings for a well-understood domain (finance, academic papers). Your vocabulary is reasonably complete. You want embeddings that capture global relationships.

### When FastText Is the Right Call

FastText shines when:
- You encounter new words at test time regularly
- Morphology matters (your language has complex inflections)
- Users make typos or you have misspellings
- You're working in specialized domains with novel terminology
- Your text is noisy or unstructured

Example: You're processing user-generated content, code, product names, domain-specific jargon. Exact word matching is too brittle. You need graceful degradation when you see `lerning` or `FastAPI`.

### The Honest Limitation (Both Methods)

They both produce **static embeddings** — one vector per word, regardless of context.

So `bank` gets one vector whether you're talking about money or riverbanks. Later models (BERT, contextual embeddings) fixed this by computing vectors *in context*. But that's a different era of NLP.

## The Big Picture

By now, you've worked through three progressions:

**Word2Vec taught you:** If you predict context, you learn meaning.

**GloVe + FastText teach you:** It's not just about *what* you predict, it's about *what data you use* and *how you represent* the unit you're learning from.

- GloVe uses global statistics instead of local windows
- FastText uses subword composition instead of whole-word atoms

Each addresses a real limitation. Each improved real NLP systems.

### What Comes Next

You've now hit the ceiling of static embeddings. One vector per word is too rigid:

- `bank` means different things in different sentences
- `saw` is a verb in one sentence, a noun in another
- Context matters, and static vectors throw it away

The next leap is **contextual embeddings** — where the model computes a fresh vector for each word *in each sentence*. That's where ELMo came from, and where BERT made it mainstream.

To get there, you need recurrent networks or transformers that process entire sentences. But conceptually, you're building on everything here:
- Learn from context (Word2Vec)
- Use global patterns (GloVe)
- Handle subword structure (FastText)
- Plus: adjust vectors based on surrounding sentence (new)

So you're not abandoning embeddings. You're making them smarter and more context-aware.
