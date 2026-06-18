---
layout: post
title: "How Computers Understand What Text Is About: A Guide for Product Managers"
date: 2026-01-16
series: "NLP for Product Managers"
series_author: "Mayank Sharma"
excerpt: "Computers don't read words, they work with numbers. Here's how text gets converted into something a machine can reason about, and what that means for your product."
---

Computers don't understand words the way you do. They work with numbers. So when your product sends a user's review, a search query, or a support ticket to an AI model, the first thing that model does is convert that text into a list of numbers.

This conversion is called **vectorization**, and the way you do it has a huge impact on how well your AI feature works.

This article explains the two most common approaches **Bag-of-Words** and **TF-IDF**, how they work, what they're good for, and shows you how they're used in real products.

---

## Why This Matters for Product Managers

Whenever your product uses language AI for search, recommendation, spam detection, sentiment tagging, and more, there's a representation layer underneath it. The choices made in that layer determine:

- Whether your search returns relevant results or random ones
- Whether your spam filter catches real spam without blocking legitimate emails
- Whether your recommendation engine surfaces content users actually want
- How well your AI handles unusual or uncommon text

You don't need to implement this yourself. But understanding what's happening helps you evaluate trade-offs, spot the root cause of bad behaviour, and have informed conversations with your data or engineering team.

---

## The Core Problem: Turning Words into Numbers

Imagine you have two customer reviews:

> Review A: "The battery lasts all day and the camera is amazing."
> Review B: "The camera quality is poor and the battery drains fast."

A human reads these and immediately knows A is positive and B is negative. But a machine needs numbers. It can't reason directly from the words, so it needs a structured format.

Vectorization is the process that creates that structure.

---

## Approach 1: Bag-of-Words — Counting What's There

**Bag-of-Words (BoW)** is the simplest approach. For each piece of text, count how many times each word appears. That's it.

The name is intentionally literal: you throw all the words into a bag and count them, ignoring order, grammar, and context.

### How It Works — A Simple Example

Say you have three reviews:

- "great product love it"
- "terrible product waste of money"
- "love the quality great value"

**Step 1:** Build a vocabulary list. This is the list of all unique words across all reviews.

`[great, product, love, it, terrible, waste, of, money, the, quality, value]`

**Step 2:** For each review, count how many times each vocabulary word appears.

| Review | great | product | love | terrible | waste | money | quality | value |
|--------|-------|---------|------|----------|-------|-------|---------|-------|
| Review 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| Review 2 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| Review 3 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 |

Each row is now a list of numbers. The machine can work with this.

### What Bag-of-Words Gets Right

- It's simple and fast to compute
- It works surprisingly well for short, direct text (support tickets, product reviews)
- A model can quickly learn that "great" and "love" signal positive sentiment and "terrible" and "waste" signal negative
- Easy to explain to stakeholders because it's no mystery

### What Bag-of-Words Gets Wrong

**It ignores word order completely.**

"Dog bites man" and "Man bites dog" produce identical BoW vectors, even though they mean very different things.

**It treats all words as equally important.**

"The," "a," and "is" appear constantly but mean almost nothing. Meanwhile, a specific word like "refund" might be rare but extremely important. BoW doesn't distinguish between these.

**It has no sense of meaning.**

"Happy," "joyful," and "delighted" all mean similar things, but BoW treats them as completely unrelated.

---

## Approach 2: TF-IDF — Measuring Word Importance

**TF-IDF** stands for Term Frequency–Inverse Document Frequency. The name sounds intimidating, but the idea is intuitive: give more weight to words that are distinctive, and less weight to words that appear everywhere.

Think of it like a journalist deciding which words to highlight in a story. "The" and "is" appear constantly, they tell you nothing specific. But "bankruptcy" appearing in a company's press release? That's a signal worth flagging.

### The Two Ideas Behind TF-IDF

**Term Frequency (TF):** How often does a word appear in *this specific document*? Words that appear more often are probably more relevant to the document's topic.

**Inverse Document Frequency (IDF):** How rare is this word *across all documents*? Words that appear in almost every document (like "the") carry little information. Words that appear in only a few documents are likely to be distinctive and meaningful.

TF-IDF multiplies these two scores together. The result: a word gets a high score when it appears often in one specific document *and* rarely in others.

### A Real Example

Imagine an email system processing thousands of emails. The word "the" appears in almost every email, so its IDF score is very low. The word "invoice" appears in some emails but not most, so its IDF score is higher. If "invoice" also appears many times in a specific email (high TF), that email is probably about billing.

This is the logic behind spam filters, email categorisation, document search, and content tagging.

### TF-IDF vs Bag-of-Words

| | Bag-of-Words | TF-IDF |
|---|---|---|
| **What it measures** | How often a word appears | How important a word is relative to other documents |
| **Common words** | Treated the same as rare words | Penalised (low importance) |
| **Rare meaningful words** | Treated the same as common words | Boosted (high importance) |
| **Best for** | Quick baselines, simple classification | Document search, content ranking, topic detection |

**The PM framing:** Bag-of-Words is your first draft. TF-IDF is the smarter version that says "stop counting every 'the' and pay attention to what actually matters."

---

## Measuring Similarity Between Documents

Once text is converted to numbers, you can measure how similar two pieces of text are. This is the foundation of search, recommendations, and deduplication.

The standard approach is called **cosine similarity**. You don't need to know the math behind it. It's just the intuition: two documents that use very similar words in similar proportions will have a high similarity score (close to 1). Two completely different documents will have a low score (close to 0).

**Where this shows up in products:**

- **Search:** "Which documents in our database are most similar to this search query?"
- **Recommendations:** "Which articles are most similar to the one this user just read?"
- **Duplicate detection:** "Is this support ticket basically the same as one we've already seen?"
- **Plagiarism detection:** "Is this essay too similar to another one in our system?"

The same mechanism that turn text into numbers, measure similarity, and powers all of these.

---

## Real-World Product Examples

**Search engine:** A user types "cheap flights to Paris." BoW and TF-IDF help surface documents that use these words in relevant combinations. Words like "Paris" and "flights" get high scores. Common words like "to" get low scores and barely influence the results.

**Spam filter:** The word "FREE" appears constantly in spam emails. But if your filter uses TF-IDF, it can tell the difference between "FREE" as a standalone claim versus a normal use in a customer conversation. Context matters.

**Support ticket routing:** "My payment isn't going through" should be routed to billing. TF-IDF helps a classifier detect that "payment" and "billing" are distinctive signals worth acting on.

**Job posting match:** A job platform wants to show candidates the most relevant postings. TF-IDF can identify distinctive skills in each posting ("Kubernetes," "HIPAA compliance") and match them to candidates with those specific terms in their profiles.

---

## The Big Limitation: These Methods Don't Understand Meaning

Both Bag-of-Words and TF-IDF share a fundamental weakness: they treat words as labels with no inherent meaning.

"Happy" and "joyful" are treated as completely different words. "Car" and "automobile" share no connection. A user searching for "automobile repair" won't find a document about "car maintenance" unless the words are identical.

This is why modern AI systems layer something more powerful on top: **word embeddings**, which actually capture semantic meaning. We cover that in the next article.

But BoW and TF-IDF are still used today, often as a first step or a baseline. They're fast, interpretable, and surprisingly good for focused, domain-specific text. Before you jump to a more complex solution, it's often worth asking: would TF-IDF actually solve this problem? Sometimes the answer is yes.

---

## Conclusion

- **Vectorization** is how text becomes numbers, a required step before any AI can process language
- **Bag-of-Words** counts word occurrences, simple, fast, works for many use cases
- **TF-IDF** improves on BoW by weighing words based on how distinctive they are, not just how frequent
- **Cosine similarity** lets you measure how similar two pieces of text are, which powers search and recommendations
- Both methods ignore word meaning, they can't tell that "happy" and "joyful" are related