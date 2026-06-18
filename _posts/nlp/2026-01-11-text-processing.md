---
layout: post
title: "How Computers Read Text: Tokenization, Stemming, and Lemmatization for PMs"
date: 2026-01-11
series: "NLP Explained"
series_author: "Mayank Sharma"
excerpt: "Before AI can understand language, it has to break it apart. Here's what that actually means, and why it matters for the products you build."
---

Imagine you hire someone to read thousands of customer reviews and tag them by theme. Before they can do that, they need to actually *read*, recognise words, understand that "running" and "run" mean the same thing, and figure out where one idea ends and another begins.

Computers have to do the same thing. They just do it differently.

This article explains the first three steps every language AI takes when processing text and why the decisions made at this stage affect everything downstream: search quality, chatbot accuracy, content tagging, and more.

---

## Why This Matters for Product Managers

You probably don't need to build these systems yourself. But you will:

- Write requirements for features that depend on them (search, filters, recommendations)
- Evaluate whether a vendor's NLP system is likely to handle your use case
- Diagnose why a search bar returns weird results, or why a chatbot misunderstands users
- Make trade-off decisions between speed and accuracy in text-heavy features

Understanding this layer helps you ask the right questions and catch problems before they ship.

---

## Step 1: Tokenization — Breaking Text into Pieces

**What it is:** Splitting a chunk of text into smaller units called *tokens*.

Think of it like a librarian who, before cataloguing a book, first has to separate every word on every page. The librarian can't work with a wall of text. They need individual units they can examine one by one.

A token is usually a word, but not always. Punctuation, numbers, and even parts of words can all be tokens depending on the system.

**Example:**

The sentence `"It's a beautiful day, isn't it?"` becomes:

`[It] ['s] [a] [beautiful] [day] [,] [is] [n't] [it] [?]`

Notice that "It's" was split into "It" and "'s" — the system is smarter than just splitting at spaces.

### Three Flavours of Tokenization

**Word tokenization** splits text into individual words. This is the most common approach and works well for languages like English.

**Sentence tokenization** splits a paragraph into individual sentences. This matters for tasks like summarisation or when you want to analyse one idea at a time.

**Subword tokenization** splits words into smaller meaningful pieces. Modern AI models like GPT and BERT use this approach. So "remarkably" might become ["remark", "ably"]. The benefit: even words the system has never seen before can be handled by recognising their parts.

### Why Subword Tokenization Matters for Products

Say you're building a product search feature and a user types "unboxable." Your AI has probably never seen that word. With simple word tokenization, it draws a blank. With subword tokenization, it recognises "un" and "box" and "able" and can make a reasonable guess.

This is the difference between a search bar that says "no results found" and one that at least tries to find something relevant.

---

## Step 2: Stemming — Chopping Words Down to Their Core

**What it is:** Reducing a word to its base form by stripping away prefixes and suffixes.

Think of it like a postal sorting system that treats "runner," "running," and "ran" as all belonging to the same pile. The system doesn't need perfect accuracy, it just needs to group similar words together fast.

**Example:**

| Original | After Stemming |
|----------|----------------|
| running  | run            |
| easily   | easili         |
| fairly   | fair           |

Notice "easily" becomes "easili", which is not a real word. Stemming is fast and pragmatic, but it's blunt. It chops, it doesn't think.

### Where Stemming Is Used

Stemming is common in **search engines** and **document indexing**, situations where you need to process millions of words quickly and a bit of roughness is acceptable. When someone searches for "running shoes," you want results for "run" and "runner" and "runs" to show up too.

The trade-off is accuracy. Stemming sometimes collapses words that shouldn't be grouped (like "caring" and "car"), or produces unrecognisable stems like "easili."

---

## Step 3: Lemmatization — Finding the Dictionary Form

**What it is:** Reducing a word to its proper dictionary form, called a *lemma*.

This is the more intelligent cousin of stemming. Instead of blindly chopping off endings, lemmatization understands grammar. It knows that "ran" is the past tense of "run," and it knows that "saw" could mean the tool *or* the verb "to see", depending on how it's used in the sentence.

**Example:**

| Original   | After Lemmatization |
|------------|---------------------|
| running    | run                 |
| better     | good                |
| saw (verb) | see                 |
| saw (noun) | saw                 |

Notice "better" becomes "good", but stemming would leave it as "better" or mangle it. Lemmatization understands the relationship.

### Why Context Matters

The word "saw" is a perfect example. Without knowing its context, you can't lemmatize it correctly:

- "She **saw** the film" → verb → lemma: **see**
- "He used a **saw** to cut the wood" → noun → lemma: **saw**

This is why lemmatization is slower but more accurate. It's doing real grammatical analysis, not just pattern matching.

### Where Lemmatization Is Used

Lemmatization is preferred for **chatbots**, **sentiment analysis**, and **content understanding**, anywhere you need semantic accuracy, not just speed. If you're building a feature that has to genuinely understand what users are saying, you'll usually find lemmatization somewhere in the pipeline.

---

## Stemming vs. Lemmatization: Which Should Your Product Use?

| | Stemming | Lemmatization |
|---|----------|---------------|
| **Speed** | Very fast | Slower |
| **Accuracy** | Rougher — may produce non-words | Higher — always a real word |
| **Best for** | Search indexing, large-scale filtering | Chatbots, sentiment analysis, content understanding |
| **Example output** | "easili" | "easy" |

If your product is processing millions of documents per second and a bit of error is acceptable, stemming works. If your product is having a conversation with a user, or trying to understand nuanced text, invest in lemmatization.

---

## Real-World Examples

**E-commerce search:** A user searches "running shoes." Stemming makes sure the search also returns results tagged "runner," "runs," and "run." Without this step, exact-match search would miss relevant products.

**Customer support chatbot:** A user types "I can't get my order cancelled." The bot needs to understand "cancelled," "cancellation," "cancelling" all point to the same intent. Lemmatization helps unify these variants.

**Content recommendation:** A media platform wants to group articles about similar topics. If stemming/lemmatization isn't applied, "election" and "elections" get treated as different topics. Your recommendations become noisier.

**Review analysis:** You're processing thousands of reviews to detect themes. "Loved it," "loving the product," "absolutely love this", all need to be recognized as expressing the same sentiment. Without lemmatization, you'd miss patterns.

---

## Conclusion

Before any AI can do something useful with text like search, summarise, classify, respond, and more, it has to first *clean and normalize* that text. Tokenization, stemming, and lemmatization are the first three steps of that process.

They're not glamorous. But they're load-bearing. Poor text preprocessing leads to poor search results, missed patterns in customer feedback, and chatbots that seem inexplicably confused.

When a language feature in your product isn't working the way users expect, this layer is often where the problem starts.
