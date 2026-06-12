---
layout: post
title: "Teaching Computers That Words Have Meaning: Word Embeddings for Product Managers"
date: 2026-01-24
series: "NLP for Product Managers"
series_author: "Mayank Sharma"
excerpt: "Bag-of-Words treats 'happy' and 'joyful' as strangers. Word embeddings teach computers that some words are neighbours. Here's why that changes everything."
---

In the previous articles, we looked at Bag-of-Words and TF-IDF methods that turn text into numbers by counting words. They're useful, but they have a fundamental blind spot: they treat every word as an isolated label with no connection to any other word.

"Happy" and "joyful" are completely unrelated, as far as BoW is concerned. So are "car" and "automobile." A user searching for "cheap flights" won't find a page titled "affordable airfare" unless the exact words match.

This is a serious product problem. Users don't search the way documents are written. They use synonyms, related concepts, and their own vocabulary. If your AI can't bridge those gaps, it fails the user, even when the relevant content exists.

**Word embeddings** solve this. They give every word a location in a kind of meaning-space, where similar words are placed near each other. "Happy" and "joyful" end up as neighbours. "Car" and "automobile" are close together. The machine develops a geography of language.

---

## The Core Idea: Words as Map Coordinates

The easiest way to understand word embeddings is to imagine a map.

Imagine a map where every word in the English language has its own location. Words that tend to be used in similar ways end up close together on this map. Words that have nothing in common end up far apart.

On this map:
- "King" and "queen" are near each other (both royalty, used in similar contexts)
- "Paris," "London," and "Tokyo" cluster together (all capital cities)
- "Run," "sprint," and "jog" are neighbours (all describe movement on foot)
- "Banana" and "stock market" are far apart

The "coordinates" of each word on this map are what we call its **embedding** — a list of numbers that encodes its position in meaning-space.

This is a big deal. Now when a user searches "affordable flights," the system can find content about "cheap airfare" because the words sit near each other on the map. When someone asks about "automobile insurance," results about "car insurance" come up. Meaning becomes searchable.

---

## How Does the Map Get Built?

Here's the surprising part: no human draws this map. The computer builds it by reading enormous amounts of text and observing patterns.

The key insight, discovered by linguists long before AI, does:

> **You can understand what a word means by looking at what words tend to appear around it.**

Read that again. It's a simple observation, but it's remarkably powerful.

Consider these sentences:

- "The king addressed his subjects from the palace."
- "The queen addressed her subjects from the palace."
- "The king and queen attended the royal ceremony."

The system notices that "king" and "queen" keep showing up near the same kinds of words: "palace," "subjects," "royal," "ceremony." It concludes they must be related, not because anyone told it so, but because they keep *hanging out together* in text.

Do this across billions of sentences, and a rich map of meaning emerges. Words build their map coordinates entirely from the company they keep.

---

## The Famous Example: Word Arithmetic

One of the most striking demonstrations of word embeddings is that you can do arithmetic with them, and get meaningful results.

The classic example:

**king − man + woman ≈ queen**

Subtract the concept of "man" from "king," add the concept of "woman," and you land very close to "queen" on the map.

Similarly:
- Paris − France + Germany ≈ Berlin
- Doctor − man + woman ≈ nurse (though this also reveals biases in training data — more on that below)

This isn't magic. It's the geometry of the map. If "king" and "man" are far apart in the direction of gender, and "queen" and "woman" are the same distance apart in the same direction — the math works out.

For product purposes, this means the AI doesn't just recognise exact word matches. It understands *relationships* between concepts.

---

## Two Ways to Build Word Embeddings: CBOW and Skip-Gram

The most well-known method for building word embeddings is called **Word2Vec**. It offers two flavours, which take slightly different approaches to learning the map.

You don't need to implement these yourself — but understanding the difference helps you set expectations for how your AI will perform in different situations.

### CBOW — Fill in the Blank

**CBOW (Continuous Bag of Words)** learns from a simple game: given the words around a blank, predict what word should go in the middle.

Example:

> "The ___ sat on the throne." (Context: The, sat, on, the, throne)
> The blank is probably: king, queen, prince, emperor...

The system plays this fill-in-the-blank game millions of times. Each time it predicts correctly, the map coordinates of those words shift slightly closer together. Over time, the map takes shape.

**What CBOW is good at:** Learning from common words that appear frequently in text. It's faster to train and works well when you have a lot of data and most of your vocabulary is everyday language.

### Skip-Gram — Predict the Neighbourhood

**Skip-Gram** reverses the game: given one word, predict the words that should appear around it.

Example:

> Given the word "throne," predict what words appear nearby:
> "king," "queen," "palace," "royal," "sat"...

Skip-Gram generates many prediction tasks from a single word. This makes it more thorough — especially for unusual or rare words that don't appear very often.

**What Skip-Gram is good at:** Building better representations for specialised vocabulary, rare terms, or technical language. It takes longer to train but gives more detailed results for words that don't appear often.

### CBOW vs Skip-Gram: The PM Trade-Off

| | CBOW | Skip-Gram |
|---|---|---|
| **Training speed** | Faster | Slower |
| **Common words** | Very good | Good |
| **Rare/specialised words** | Less reliable | Better |
| **Best for** | General-purpose language | Specialised domains (medical, legal, technical) |

**The practical question:** Is your product dealing with everyday language (customer reviews, social media, general support queries), or with specialised vocabulary (medical records, legal contracts, engineering documentation)? The answer influences which approach is more appropriate.

---

## Where Word Embeddings Show Up in Products

Once your system has a meaning-map of language, a lot of things become possible that weren't possible before.

### Semantic Search

Traditional search is exact-match: you search for "running shoes," you get results containing "running shoes." Semantic search finds results that are *about* the same thing, even if the exact words differ.

A user searching for "comfortable footwear for jogging" can find pages titled "best running shoes" — because "comfortable," "footwear," "jogging," and "running shoes" all cluster together in meaning-space.

This matters enormously for product discoverability.

### Content Recommendations

"Customers who read this article about 'machine learning' might also enjoy this piece on 'artificial intelligence' and 'data science.'" That recommendation works because those concepts live nearby on the embedding map, even if no specific words overlap.

### Autocomplete and Spell Correction

When a user starts typing "how do I can", your system can predict they probably meant "how do I cancel" or "how do I change," based on what words are semantically nearby and what patterns it has seen before.

### Chatbot Intent Recognition

A user might say "I want to stop my subscription," "I'd like to cancel my plan," or "I'm done with this service." These are all different sentences but express the same intent. With word embeddings, an AI assistant can map all of them to the same action, without needing a hard-coded list of every possible phrasing.

### Duplicate Detection and Clustering

If your team is reviewing hundreds of support tickets, embeddings can automatically group similar tickets together: all the billing questions cluster in one place, all the login issues in another. This makes triage dramatically faster.

---

## The Limitations: What Word Embeddings Still Get Wrong

Understanding where embeddings fail is just as important as knowing where they succeed.

### One Word, One Meaning

Traditional word embeddings assign one set of coordinates per word, *permanently*. But many words mean different things depending on context.

- "Bank" (financial institution) vs "bank" (riverbank)
- "Apple" (the fruit) vs "Apple" (the company)
- "Light" (not heavy) vs "light" (illumination) vs "light" (pale colour)

A basic word embedding puts "bank" in one fixed location on the map, even though it should be in two different places depending on how it's used. This is called the **polysemy problem**, and it leads to subtle errors in AI systems.

More advanced models (like BERT) solve this by computing the embedding in context, so the word's location on the map shifts depending on the surrounding sentence. We'll cover that in a later article.

### Bias in the Training Data

Remember the "Doctor − man + woman ≈ nurse" example? That's a real result from early word embeddings, and it reflects a bias in the text the system was trained on. If most historical writing describes doctors as men and nurses as women, the embedding map picks up that association and bakes it in.

For product teams building AI features, this is a real concern. Embeddings trained on historical data can perpetuate historical biases, leading to discriminatory outcomes in hiring tools, recommendation systems, and content filters.

This isn't a reason to avoid word embeddings, but it's a reason to audit them.

### Unknown Words

If a word never appeared in the training data, the system has no map coordinates for it. New product names, slang, technical terms, and typos can all fall into this gap. (The next article covers methods that address this.)

---

## The Big Picture

Word embeddings were a turning point in AI's ability to understand language. Before them, systems worked with word counts. After them, systems could reason about meaning, similarity, and relationships.

For product managers, this translates to:

- **Better search** that finds what users mean, not just what they typed
- **Smarter recommendations** based on conceptual similarity
- **More flexible chatbots** that understand intent regardless of exact phrasing
- **Automated content organisation** at a scale no human team could match

The limitation of word embeddings is that they assign one fixed meaning per word. This was the obvious next problem to solve. That's what the next article tackles.

---

## Key Takeaways

- **Word embeddings** place every word on a "meaning map" where similar words are physically near each other
- The map is built automatically by reading enormous amounts of text and observing which words tend to appear together
- **CBOW** is fast and reliable for common words; **Skip-Gram** is slower but better for rare and specialised terms
- Embeddings power semantic search, smart recommendations, intent recognition, and content clustering
- The main limitation: each word gets one fixed location, regardless of its meaning in context

---

## Further Reading

- Previous: [How Computers Understand What Text Is About]({% post_url nlp/2026-06-03-text-representation %})
- Next: [Smarter Word Understanding: GloVe and FastText Explained]({% post_url nlp/2026-06-08-advance-word-embeddings %})
