---
layout: post
title: "Text Representation: From Bag-of-Words to TF-IDF"
date: 2026-02-03
series: "NLP: Foundations to Advance"
series_author: "Mayank Sharma"
excerpt: "Learn how to transform raw text into numerical vectors that machines can understand using Bag-of-Words and TF-IDF techniques."
---

Continuing in our series in NLP, let's dive into text representation. Imagine you are a world-class chef standing in a massive, chaotic pantry filled with thousands of ingredients from across the globe. Some ingredients, like salt and water, are in almost every dish you make. Others, like rare saffron or white truffles, appear only in your most exquisite masterpieces. If a food critic asked you to describe a dish without using its name, how would you do it? You might list the ingredients and their quantities: "This dish has 200g of pasta, 50g of parmesan, and a pinch of black pepper." 

To the critic, this "list of ingredients" is a fingerprint of the dish. It doesn't tell them the order in which you added the ingredients or the technique you used, but it gives them a very good idea of what the dish *is*.

This is exactly how machines "read" text. A computer cannot understand the poetic beauty of a Shakespearean sonnet or the urgency of a breaking news headline in their raw, string-based form. To a machine, a sentence is just a sequence of characters. To perform tasks like sentiment analysis, document classification, or building a search engine, we must transform these "dishes" of text into a numerical "list of ingredients." This process is known as **Text Vectorization** or **Text Representation**.

In this article, we will explore the foundational techniques of text representation: the **Bag-of-Words (BoW)** model and its more sophisticated cousin, **TF-IDF (Term Frequency-Inverse Document Frequency)**. We will delve into the mathematical elegance of these methods, understand their limitations, and see how they form the backbone of many modern NLP systems.

---

## What You'll Learn

- **The Necessity of Vectorization**: Why machines require numerical representations of text.
- **Bag-of-Words (BoW)**: Construction, Binary vs. Frequency counts, and the "Sparsity" challenge.
- **N-grams**: How Bigrams and Trigrams capture local context that simple words miss.
- **TF-IDF Deep Dive**: Mathematical derivation and the logic behind penalizing common words.
- **Implementation**: Hands-on guide using Scikit-learn and NLTK.
- **Comparison & Trade-offs**: When to use which method and understanding their architectural limitations.

---

## 1. The Core Challenge: Why Machines Need Numbers

Computers are essentially glorified calculators. They excel at performing billions of mathematical operations per second, but they have no innate concept of "meaning." A string like `"Machine Learning"` is just a sequence of Unicode values (`M=77, a=97, c=99...`). 

To bridge the gap between human language and machine logic, we need a **Mapping Function** ($f$) that transforms a piece of text ($T$) into a vector ($\mathbf{v}$):
$$\mathbf{v} = f(T)$$
where $\mathbf{v} \in \mathbb{R}^n$, and $n$ is the dimensionality of our representation.

### The Properties of a Good Representation
A "good" vector representation should ideally satisfy three properties:
1.  **Fixed Length**: Regardless of whether the input is a 3-word tweet or a 500-page novel, the resulting vector should have a consistent size for the model to process.
2.  **Semantic Similarity**: Documents that are about similar topics should result in vectors that are "close" to each other in the mathematical space (measured by distance metrics like Euclidean distance or Cosine similarity).
3.  **Discriminative Power**: The representation should capture enough detail to distinguish between a "sports" article and a "politics" article.

---

## 2. Bag-of-Words (BoW): The Ingredient List

The **Bag-of-Words** model is the simplest form of text representation. It treats a document as an unordered collection (a "bag") of words, disregarding grammar, word order, and sentence structure. It only cares about **presence** and **frequency**.

### Step-by-Step Construction
Suppose we have a small collection of documents (a **Corpus**):
*   **Doc 1**: "The cat sat on the mat."
*   **Doc 2**: "The dog sat on the log."
*   **Doc 3**: "The cat and the dog are friends."

#### Step 1: Create a Vocabulary
We collect all unique words across the entire corpus (after basic preprocessing like lowercasing):
`Vocabulary = {"the", "cat", "sat", "on", "mat", "dog", "log", "and", "are", "friends"}`
Size of Vocabulary ($V$) = 10.

#### Step 2: Vectorization (Frequency-based)
Each document is now represented by a vector of length $V$, where each index corresponds to a word in the vocabulary.

| Document | the | cat | sat | on | mat | dog | log | and | are | friends |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Doc 1** | 2 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Doc 2** | 2 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| **Doc 3** | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 1 |

Vector for Doc 1: $[2, 1, 1, 1, 1, 0, 0, 0, 0, 0]$

### Binary BoW vs. Frequency BoW
*   **Binary BoW**: Only records if a word is present (1) or absent (0). This is useful for tasks like spam detection where the *mere presence* of words like "VIAGRA" or "FREE" is more important than how many times they appear.
*   **Frequency BoW**: Records the raw count. This captures the "emphasis" or "topic" of a document more accurately.

### The Problem of Sparsity
In a real-world corpus (e.g., all Wikipedia articles), the vocabulary might consist of 100,000+ words. However, a single paragraph might only contain 50 unique words. This means the vector for that paragraph will have 50 non-zero values and 99,950 zeros.
This is called **Sparsity**. Sparse vectors are computationally expensive to store and process unless optimized (using specialized sparse matrix formats like CSR).

### The "Stopword" Problem
Notice in our example that "the" appears in every document with a high frequency. In English, words like "the", "is", "at", "which", and "on" are extremely common but carry very little semantic value. In a Bag-of-Words model, these **Stopwords** often dominate the vectors, making it harder for the model to focus on the truly descriptive words (like "cat" or "log"). 

Standard practice is to remove these words during preprocessing, but as we'll see, TF-IDF provides a more elegant mathematical solution.

---

## 3. N-grams: Capturing Local Context

The biggest weakness of BoW is the total loss of word order. The sentences *"The dog bit the man"* and *"The man bit the dog"* result in identical BoW vectors, even though their meanings are drastically different (and one is much more news-worthy!).

**N-grams** attempt to fix this by considering sequences of $n$ adjacent tokens.
*   **Unigrams ($n=1$)**: "The", "cat", "sat". (Standard BoW)
*   **Bigrams ($n=2$)**: "The cat", "cat sat", "sat on".
*   **Trigrams ($n=3$)**: "The cat sat", "cat sat on".

### How Bigrams Change the Vocabulary
If we use both Unigrams and Bigrams, our vocabulary size explodes. For a vocabulary of size $V$, the number of possible bigrams is $V^2$. While most of these won't appear in the data, the effective vocabulary still grows significantly.

| Document | the | cat | the cat | cat sat | sat on | ... |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Doc 1** | 2 | 1 | 1 | 1 | 1 | ... |

**Advantage**: Captures phrases like "Machine Learning", "New York", or "Not happy" (negation).
**Disadvantage**: Drastically increases dimensionality and sparsity.

---

## 4. TF-IDF: The Precision Scalpel

If BoW is an "ingredient list," **TF-IDF (Term Frequency-Inverse Document Frequency)** is a "weighted importance list." It acknowledges that not all words are created equal. 

The core intuition is: **A word is important if it appears frequently in a specific document, but it is less important if it appears frequently across many documents in the entire corpus.**

### 1. Term Frequency (TF)
TF measures how frequently a term occurs in a document. There are several ways to calculate this:

*   **Binary**: $0, 1$
*   **Raw Count**: $f_{t,d}$ (Number of times term $t$ appears in document $d$)
*   **Term Frequency**: $\frac{f_{t,d}}{\sum_{k} f_{k,d}}$ (Normalized by document length)
*   **Logarithmic Scaling**: $1 + \log(f_{t,d})$ (This dampens the effect of a word appearing 100 times vs 10 times; a word appearing 100 times is important, but probably not 10 times *more* important than a word appearing 10 times).

### 2. Inverse Document Frequency (IDF)
IDF measures how "rare" or "informative" a term is across the whole corpus. If a word appears in every single document (like "the"), its IDF should be very low (close to 0).

The standard formula for IDF is:
$$IDF(t, D) = \log\left(\frac{N}{\lvert \{d \in D \mid t \in d\} \rvert}\right)$$

Where:
*   $N$: Total number of documents in the corpus.
*   $\lvert \{d \in D \mid t \in d\} \rvert$: Number of documents where term $t$ appears.

**Why the Logarithm?**
The log function ensures that the IDF doesn't explode for very rare words and stays within a manageable range. If $N=1,000,000$ and a word appears in only 1 document, the raw ratio is $1,000,000$. The $\log_{10}$ of that is $6$. This makes the weights much more stable for machine learning models.

### 3. The Final Calculation: $TF \times IDF$
The TF-IDF score for a term $t$ in document $d$ is:
$$w_{t,d} = TF(t,d) \times IDF(t)$$

### A Numerical Example
Suppose we have a corpus of **1,000 documents**.
We are looking at **Document A**, which has **100 words**.
The word **"Algorithm"** appears **5 times** in Document A.
The word **"Algorithm"** appears in **10 documents** across the entire corpus.

**Calculate TF:**
$$TF(\text{"Algorithm"}, \text{Doc A}) = \frac{5}{100} = 0.05$$

**Calculate IDF:**
$$IDF(\text{"Algorithm"}) = \log\left(\frac{1000}{10}\right) = \log(100) = 2$$

**Calculate TF-IDF:**
$$w = 0.05 \times 2 = 0.10$$

Now compare this to the word **"The"**, which appears **10 times** in Document A but appears in all **1,000 documents**.
*   $TF = 10/100 = 0.10$
*   $IDF = \log(1000/1000) = \log(1) = 0$
*   $TF-IDF = 0.10 \times 0 = 0$

**Result**: TF-IDF successfully filtered out the "noise" word ("the") and highlighted the "informative" word ("algorithm"), even though "the" appeared more frequently in the document!

---

## 5. Implementation Walkthrough

We will use `scikit-learn`, the industry standard for classical ML, to implement these vectors.

### Setting Up the Environment
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Download stopwords if you haven't
# nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
```

### Implementing Bag-of-Words (CountVectorizer)
```python
corpus = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends."
]

# 1. Initialize CountVectorizer
# We can include n-grams here using ngram_range=(1, 2)
vectorizer = CountVectorizer(stop_words='english')

# 2. Fit and Transform
X_bow = vectorizer.fit_transform(corpus)

# 3. View the Result
df_bow = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print("Bag-of-Words Representation:")
print(df_bow)
```

### Implementing TF-IDF (TfidfVectorizer)
```python
# 1. Initialize TfidfVectorizer
# norm='l2' ensures that the vectors have a length of 1 (useful for Cosine Similarity)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', norm='l2')

# 2. Fit and Transform
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

# 3. View the Result
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nTF-IDF Representation:")
print(df_tfidf.round(4))
```

### Advanced: Sublinear TF Scaling
In `scikit-learn`, you can use `sublinear_tf=True` which applies the $1 + \log(TF)$ scaling we discussed earlier.
```python
tfidf_vectorizer_log = TfidfVectorizer(sublinear_tf=True, stop_words='english')
```

---

## 6. Comparison Table: BoW vs. TF-IDF

| Feature | Bag-of-Words (BoW) | TF-IDF |
| :--- | :--- | :--- |
| **Logic** | Simple counting of occurrences. | Statistical weighting of importance. |
| **Weighting** | All words are treated equally. | Frequent words in corpus are penalized. |
| **Noise Sensitivity** | High (Stopwords dominate). | Low (Automatically suppresses noise). |
| **Interpretability** | High (Numbers = Counts). | Medium (Numbers = Relative importance). |
| **Best For** | Very small datasets, simple tasks. | Most classical NLP tasks, Search engines. |

---

## 7. Advantages & Limitations

### Advantages
1.  **Simplicity**: Extremely easy to understand and implement.
2.  **Efficiency**: Computing these vectors is very fast compared to training deep learning models.
3.  **No Training Data Needed**: Unlike word embeddings (Word2Vec) or Transformers, you don't need a massive pre-trained model. You can build the vocabulary directly from your own data.

### Limitations
1.  **The Vocabulary Problem**: If your model is trained on a vocabulary and encounters a new word ("Out of Vocabulary"), it simply ignores it.
2.  **Curse of Dimensionality**: Large vocabularies lead to massive, sparse vectors that can slow down models.
3.  **Semantic Blindness**: BoW and TF-IDF treat "Good" and "Great" as two completely different dimensions. They have no concept that these words are synonyms.
4.  **Loss of Sequence**: Even with N-grams, the long-term structure and dependencies of a sentence are lost.
5.  **Fixed Representation**: Every word has one vector regardless of context (the "Bank" problem: "river bank" vs. "money bank").

---

## 8. Practical Applications

### 1. Document Classification (Spam vs. Ham)
BoW is surprisingly effective for spam detection. If a message contains "win", "prize", and "cash" multiple times, a simple Naive Bayes classifier trained on BoW vectors can achieve 98%+ accuracy.

### 2. Information Retrieval (Search Engines)
Before modern neural search, almost every search engine used a variant of TF-IDF called **BM25**. When you type a query, the engine calculates the TF-IDF vectors for your query and all documents, then ranks them using **Cosine Similarity**.

### 3. Topic Modeling
In algorithms like **Latent Dirichlet Allocation (LDA)**, the input is typically a BoW or TF-IDF matrix. The algorithm then clusters documents based on the distribution of words they contain.

---

## 9. Conclusion & Summary

Text representation is the bridge between the fluid, ambiguous world of human language and the rigid, mathematical world of machines. While we have moved towards more advanced techniques like **Word Embeddings** (Word2Vec, GloVe) and **Contextual Embeddings** (BERT, GPT), the principles of Bag-of-Words and TF-IDF remain foundational.

They teach us that frequency matters, that rarity implies information, and that even a simple "ingredient list" can capture a surprising amount of meaning.

### Key Takeaways

- **Vectorization is the process** of converting text into numerical vectors.
- **Bag-of-Words (BoW)** counts occurrences but ignores order and semantic nuance.
- **N-grams** help capture local context (e.g., "not good") at the cost of higher dimensionality.
- **TF-IDF improves on BoW** by rewarding words that are frequent in a document but rare in the corpus.
- **Sparsity is a major challenge** in text representation that requires efficient data structures.
- **The "Bag" model's greatest weakness** is the loss of semantic similarity and word order.

---
**Further Reading:**
- [Introduction to Information Retrieval* by Manning, Raghavan, and Schütze](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)
- [Applied Text Analysis with Python* by Benjamin Bengfort](https://www.amazon.in/Applied-Analysis-Python-Benjamin-Bengfort/dp/1491963042)
- [Scikit-learn Documentation: Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
