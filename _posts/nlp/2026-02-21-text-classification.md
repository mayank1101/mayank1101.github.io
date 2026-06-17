---
layout: post
title: "Text Classification for Product Managers: Teaching Machines to Sort Your Words"
date: 2026-02-21
series: "NLP for Product Managers"
series_author: "Mayank Sharma"
excerpt: "A plain-English guide to text classification for product managers: what it is, how it works, and why it quietly powers some of the most important features in modern products."
---

Think about the last time you opened your email and found that your spam folder had already filtered out fifty unwanted messages before you even looked.

You didn't set up a rule for each one. You didn't tell the system "this is spam" a hundred times. It just knew.

That's text classification.

Now think about what a company like Airbnb does with the hundreds of thousands of guest reviews they receive every month. They don't have people reading each one and deciding whether it's a complaint about cleanliness, a question about the neighbourhood, or a glowing compliment. A system does it automatically, at scale, in real time.

That's also text classification.

And if you've ever typed a message into a support chat and had it instantly routed to the right team for billing, technical support, account management, without anyone reading it first? Text classification again.

It's one of the most widely deployed AI capabilities in software products today. Not because it's the most sophisticated AI technique, it isn't, but because the problem it solves is genuinely everywhere, you just don't realise it. When you have a pile of text, and you need to know what *kind* of text it is, text classification is the tool for the job.

---

## What Is Text Classification?

Text classification is the task of reading a piece of text and assigning it to one or more predefined categories, called **labels** or **classes**.

Here are some everyday examples:

| Text | Label Assigned |
|---|---|
| "CONGRATULATIONS! You've won £1,000! Click here now!" | Spam |
| "Hey, are we still on for lunch tomorrow?" | Not Spam |
| "The battery died after just two hours. Terrible product." | Negative |
| "This is the best app I've ever used. Absolutely love it." | Positive |
| "My account was charged twice this month." | Billing Issue |
| "The app keeps crashing when I try to upload a photo." | Technical Issue |
| "Como puedo cancelar mi suscripción?" | Spanish |

In each case, the system reads the text and assigns a label. That label then drives some downstream action, like moving an email to spam, routing a ticket, translating content, flagging a review for a quality team.

The machine isn't actually understanding the text the way a human does, but it's learned to recognise patterns that reliably predict which category a piece of text belongs to.

---

## The Three Flavours of Text Classification

Not all classification tasks are the same. There are three main types worth knowing as a PM.

### Binary Classification

Two possible labels. The simplest form.

- Spam / Not Spam
- Appropriate / Inappropriate
- Relevant / Not Relevant
- Churn Risk / Not Churn Risk

Binary classification tends to be the most accurate because the model only has to decide between two options.

### Multi-Class Classification

More than two categories, but a piece of text gets exactly one label.

- Topic: Politics / Sports / Technology / Entertainment / Health
- Ticket type: Billing / Technical / Account / Feedback / Other
- Sentiment: Positive / Neutral / Negative
- Language: English / Spanish / French / German / Hindi

The challenge here is that as you add more categories, the model has more options to choose between, and accuracy typically decreases. Categories that are very similar to each other (e.g., "Technical Issue" vs. "Bug Report") are harder to distinguish than categories that are clearly different (e.g., "Billing" vs. "Feature Request").

### Multi-Label Classification

A piece of text can belong to *multiple* categories at the same time.

- A review might be both *Negative* and *About Shipping* and *About Packaging*
- A support ticket might be *Urgent*, *About Payment*, and *Related to the Mobile App*
- A news article might cover *Climate*, *Politics*, and *International Affairs*

Multi-label is the most complex of the three. It's also often the most accurate reflection of reality, because real-world text rarely fits neatly into a single box.

---

## Why This Matters for Product Managers

Here's the blunt case: **most products that handle user-generated text can benefit from text classification somewhere.**

If your product has users who write things like reviews, messages, tickets, comments, posts, forms, or emails, then you have a classification opportunity.

The value is always the same: turning unstructured text that would require a human to read and sort into structured, actionable data that a system can route, filter, prioritise, or analyse automatically.

That's a direct reduction in manual work, a direct improvement in response speed, and often a direct improvement in user experience.

---

## Real Product Use Cases

### 1. Support Ticket Routing

A customer writes to your support team: *"I was charged twice this month and I need this fixed urgently."*

A text classifier reads this and routes it immediately to the billing team, flagged as high priority. No human has to read it to make that routing decision.

Without classification: the ticket joins a general queue, gets read by a first-line agent, manually assigned, and potentially sits for hours.

With classification: the right team gets it in seconds, with the right priority label already attached.

**The product impact**: Faster resolution times, fewer tickets bounced between teams, lower support cost per ticket.

### 2. Spam and Content Moderation

Every platform that lets users post content needs to distinguish between legitimate content and spam, abuse, or policy violations.

Text classifiers can pre-screen content at the point of submission, flagging likely violations for human review or auto-removing content that exceeds a confidence threshold.

This is how platforms like YouTube, Twitter, and Reddit manage to moderate millions of posts per day with manageable human teams.

**The product impact**: Less moderation overhead, faster response to violations, cleaner user experience for everyone else.

### 3. Sentiment Analysis on User Feedback

You run a product and you want to know are things getting better or worse? Are users happier after the last release or less happy?

A sentiment classifier tags every review, support message, NPS comment, and in-app feedback response as Positive, Negative, or Neutral. Then you can track the ratio over time and get alerted when it shifts.

Instead of a quarterly qualitative review of feedback themes, you have a real-time dashboard of how users are feeling.

**The product impact**: Faster detection of regressions after releases, data-driven prioritisation of complaints, and a metric that tracks product health beyond NPS (Net Promoter Score).

### 4. Intent Detection in Chatbots

When a user types *"I want to return something"*, your chatbot needs to understand that the intent is a **return request**, not a general inquiry, not a billing question, not a technical issue.

Text classification is what maps user messages to intent categories. The chatbot then uses the detected intent to decide what to do next, like show the returns flow, ask a clarifying question, or hand off to a human.

Every chatbot that feels like it "understands" you is running text classification on your input in the background.

**The product impact**: More accurate chatbot responses, fewer "I didn't understand that" fallbacks, higher task completion rates.

### 5. Language Detection

A global product receives messages from users all over the world in English, Spanish, Portuguese, French, Arabic, and dozens more.

Language detection is text classification, where you read the text, predict which language it is. With the right label, the product can route the message to a native-speaker agent, auto-translate the interface, or apply language-specific processing.

**The product impact**: Better experience for non-English users, correct routing without users having to specify their language.

### 6. Review and Feedback Categorisation

You receive ten thousand app store reviews. Some are about bugs, some about missing features, some about pricing, some about the onboarding experience. All of them are unstructured text.

A multi-label classifier can tag each review with its topic categories. Now you can answer: *"How many reviews this week mentioned the checkout experience?"* or *"Is the share of reviews about performance increasing?"*

**The product impact**: Turns unstructured user feedback into quantitative, trackable product signals.

### 7. Risk and Urgency Flagging

Not all support tickets are equally urgent. *"I can't access my account and I have a presentation in an hour"* is different from *"Can you tell me how to change my email?"*

A classifier trained to detect urgency can prioritise the queue automatically, putting the most urgent issues at the top, regardless of when they were submitted.

**The product impact**: Critical issues get resolved faster, SLAs are easier to meet, and customer experience is dramatically better for users in genuine distress.

### 8. Legal and Compliance Screening

A fintech company receives thousands of messages from users. Some of them may contain suspicious phrases, potential fraud signals, or compliance-sensitive content.

A classifier can pre-screen messages and flag those that meet certain criteria for human review, reducing the burden on compliance teams and catching issues that would otherwise slip through.

**The product impact**: Reduced regulatory risk, more scalable compliance operations.

---

## How Does It Actually Work? (The Simple Version)

You don't need to understand the mechanics to make good product decisions around text classification. But a working mental model helps.

### The core idea: learning from examples

Imagine you're training a new employee to sort customer emails into categories. On their first day, you show them a hundred emails that have already been sorted by category. You walk through each one telling them *"This one is a billing question, see how they mention 'charged' and 'payment'? This one is a technical issue, see how they're saying the app isn't loading."*

After seeing enough examples, the employee starts to get it. They notice patterns. They can look at a new email they've never seen before and make a reasonable guess about which category it belongs to.

That's exactly how a text classification model learns. You give it thousands of labelled examples. It learns which words, phrases, and patterns tend to appear in each category. Then it applies those patterns to new, unseen text.

### What the model actually sees

Before the model reads text, the text needs to be converted into numbers, because computers work with numbers, not words. This conversion process is called **vectorisation** or **embedding**.

Think of it like turning words into coordinates on a map. Words that appear in similar contexts end up close together on the map. "Crash", "bug", and "error" end up near each other. "Billing", "Invoice", "charge", and "payment" end up near each other. The model learns that text with coordinates near the "bug cluster" is probably a technical issue.

You don't need to know how this works in detail. What matters is knowing that it exists, and that the quality of this "map" matters a lot for model accuracy.

### Zero-Shot Classification: No Training Data Required

Here's something that changes the build vs. buy calculus considerably for product managers: **zero-shot classifiers**.

A zero-shot classifier doesn't need training data from your specific domain. You give it your category labels in plain English like *"Billing Issue", "Technical Issue", "Feature Request"* and it can classify text into those categories without ever having seen a labelled example from your product.

It works because the underlying model has already been trained on vast amounts of text and has learned what these concepts mean in general. You're just applying that general understanding to your specific labels.

This is enormously valuable for product managers because it means you can prototype a classification system in minutes without a labelled dataset, no ML engineer needed, no training pipeline.

The trade-off is that zero-shot classifiers tend to be less accurate than a custom-trained model fine-tuned on your specific data. But as a starting point, for internal tools, or for low-stakes use cases, they're often more than good enough.

---

## What Makes a Good Training Dataset?

If you're going to train a custom classifier (rather than using zero-shot), the quality of your training data is the most important factor in model performance. More than the model architecture. More than the engineering choices. **The data**.

Here's what matters:

**Enough examples per category**

A model needs to see enough examples of each label to learn what it looks like. As a rough rule of thumb, aim for at least a few hundred examples per category. The more categories you have, and the more similar they are to each other, the more examples you need.

**Representative examples**

Your training data should look like your real-world data. If your users write informally, your training data should include informal text. If they write in multiple languages, your training data should reflect that. A model trained only on formal English will struggle with casual chat messages.

**Consistent labelling**

If two people would look at the same piece of text and disagree about which category it belongs to, your model will also be confused. Before training, spend time defining clear, unambiguous criteria for each category, and check that your team's human labellers are applying them consistently.

**Balanced categories**

If your dataset has 95% "Not Spam" and 5% "Spam", a model can achieve 95% accuracy by just labelling everything as "Not Spam". Make sure each category is reasonably represented, or use techniques to correct for imbalance.

---

## Understanding Model Performance

When engineers talk about how well a text classifier performs, they use a few metrics. These are all important, but they're often misunderstood. Let's try to clarify them in simple terms.

**Accuracy** — Out of all the texts the model classified, what percentage did it get right?

Simple, but misleading on its own. As the spam example above shows, a model can have high accuracy and still be useless.

**Precision** — When the model says something is in a category, how often is it actually right?

High precision = when the model makes a call, it's usually correct. Low precision = lots of false alarms.

**Recall** — Out of all the texts that actually belong in a category, what percentage did the model find?

High recall = the model catches most of the real examples. Low recall = it's missing a lot.

**The trade-off**: Precision and recall pull against each other. Making your model more cautious (only classifying something as Spam when it's very sure) increases precision but lowers recall. Making it more aggressive (flagging everything that could be spam) increases recall but lowers precision.

As a PM, you decide which matters more for your use case. For content moderation, recall is often more important because missing harmful content is worse than occasionally flagging something benign. For auto-routing support tickets, precision matters more because a miscategorised ticket that ends up with the wrong team is annoying, but at least it gets resolved eventually.

---

## What Can Go Wrong?

### Problem 1: Categories that overlap or are ambiguous

If your categories are too similar to each other like "Technical Issue" vs. "Bug Report", or "Complaint" vs. "Negative Feedback" then even a good model will mix them up. This is a labelling design problem, not a model problem.

**What to do**: Before building a classifier, write clear definitions for each category. Have multiple people independently label a sample of real examples, then check where they disagree. If humans struggle to agree, the model will too.

### Problem 2: Too many categories

The more categories you have, the harder the problem. A classifier with 3 categories will almost always outperform one with 30. This is because the model has fewer ways to be wrong and the categories are more distinct.

**What to do**: Start with the minimum viable set of categories that drives real downstream value. You can always add more later once the core classifier is working well.

### Problem 3: Category distribution shifts over time

A classifier trained on last year's data might struggle with this year's text if user language or topics have changed. A support classifier trained before your product launched a major new feature won't have seen tickets about that feature.

**What to do**: Monitor model performance over time, not just at launch. Set up a process to regularly retrain with fresh labelled data.

### Problem 4: The model is confident but wrong

Most classifiers produce a probability score alongside their prediction. This is not just "this is Spam" but "this is Spam with 87% confidence". A common mistake is ignoring that score and treating all predictions equally.

**What to do**: Route low-confidence predictions to a human review queue rather than acting on them automatically. Set a confidence threshold below which the model defers to a human.

### Problem 5: The labels don't match user reality

You might design a beautiful labelling scheme that makes sense to your team but doesn't reflect how users actually write. Users don't write in categories. They write in their own words and phrases, and they might use different words to mean the same thing. For example, a user might write "I can't log in" but your label is "Authentication Failure" instead.

**What to do**: Derive your categories from real user text, not from internal team intuition. Read a sample of actual messages before defining your labels.

---

## Build vs. Buy: Your Framework

| Option | What it means | Best for |
|---|---|---|
| **Zero-shot API** (Hugging Face, OpenAI) | Provide labels, get predictions immediately, no training data needed | Prototyping, low-stakes use cases, when you have no labelled data |
| **Pre-trained model fine-tuned on your data** | Take an existing model, train it further on your specific examples | Mostly for production use cases, good balance of accuracy and effort |
| **Cloud NLP APIs** (Google, AWS, Azure) | Managed classification services with pre-built categories (sentiment, topics) | Standard categories, no ML infrastructure, fast time to market |
| **Train from scratch** | Build a custom model entirely on your own data | Rarely, only when your domain is highly specialised and existing models genuinely can't reach you |

**Starting recommendation**: Use a zero-shot classifier to prototype and validate the concept. If the concept proves valuable, invest in labelling data and fine-tuning. Most products never need to train from scratch.

---

## The Most Important Question Before You Build

Before adding text classification to your product roadmap, answer this question:

> **what will you do with the label once you have it?**

Classification is never an end in itself. It's a means to an action.

If the answer is *"route the ticket to the right team"*, great, build the routing logic alongside the classifier.

If the answer is *"surface insights in a dashboard"*, great, build the dashboard.

If the answer is *"well, it would be interesting to know"*, now that's not enough. Don't build it yet. Know the action before you build the capability.

---

## Questions to Ask Your Engineering Team

**1. What categories do we actually need?**
Start from the downstream action, not from a taxonomy you think sounds right. What decisions will the label drive?

**2. Do we have labelled data?**
If yes, how many examples per category, and how consistent is the labelling? If no, can we start with a zero-shot approach?

**3. What's the cost of a wrong prediction?**
A miscategorised email is low stakes. A miscategorised medical alert is high stakes. Thre is a trade-off between accuracy and cost. The answer determines how much accuracy investment is justified and how much time investment is justified.

**4. How will we monitor the model after launch?**
Text classifiers degrade over time as language evolves. What's the plan to detect and correct for drift?

**5. What's our confidence threshold for automated action?**
At what confidence level do we act automatically vs. route to a human? This should be a deliberate decision, not a default.

---

## A Mental Model for Your Next Product Review

Here's a clean way to remember what text classification does and when to use it:

> **Text classification** is the post office for your text data. It reads each piece of text and puts it in the right box, so the right person or system gets it, at the right time, without anyone manually sorting through the pile.

The categories are the boxes. The model is the sorting machine. Your job as a PM is to define the right boxes, make sure the machine is sorting accurately, and design what happens to the content once it lands in each box.

---

## Conclusion

- **Text classification** assigns predefined labels to text like spam/not spam, positive/negative, billing/technical/other.
- It comes in three types - binary (two labels), multi-class (one of many labels), multi-label (many labels at once).
- The most common PM use cases are support routing, content moderation, sentiment tracking, chatbot intents, language detection, feedback categorisation, urgency flagging.
- The quality of your **training data** matters more than the model choice, consistent labels, enough examples per category, representative of real user text.
- **Zero-shot classifiers** let you prototype without any labelled data,  give it your category names and it classifies immediately.
- Monitor **precision and recall** separately and decide upfront which matters more for your use case.
- Know the **action** each label will trigger before building the classifier. Text classification without a downstream action is just an interesting fact.