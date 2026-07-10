// PM Gym — Defining The Problem page data.

window.pmData = {
  flashcards: [
    {
      category: "Foundation",
      title: "Problem Framing",
      desc: "The discipline of precisely defining what's broken, for whom, and why it matters \u2014 before touching a solution.",
      application: "Do this before any ideation session; a sharp problem statement keeps the whole team pointed at the same target."
    },
    {
      category: "Anatomy",
      title: "Problem Statement",
      desc: "A specific description with four parts: who is affected, what's observed, why it matters, and the evidence behind it.",
      application: "Replace \"users are unhappy\" with \"new sellers abandon the pricing step (38%), our top activation leak.\""
    },
    {
      category: "Technique",
      title: "5 Whys",
      desc: "Asking \"why\" repeatedly (commonly five times) on a problem to move past the first symptom and reach a root, actionable cause.",
      application: "Use it when a fix keeps not sticking \u2014 the real cause is usually structural, not the first thing you noticed."
    },
    {
      category: "Trap",
      title: "Solution-Shaped Problem",
      desc: "A problem statement that secretly names a feature (\"we need a chatbot\") instead of the pain that feature might solve.",
      application: "Whenever a \"problem\" statement contains a noun like \"app\" or \"feature,\" rewrite it around the underlying pain."
    },
    {
      category: "Concept",
      title: "Problem Space vs. Solution Space",
      desc: "The problem space is what's broken and why; the solution space is how you'll fix it. Framing should stay in the problem space as long as possible.",
      application: "If the team starts debating UI details before agreeing what's broken, pull the conversation back to the problem space."
    },
    {
      category: "Discipline",
      title: "Root Cause",
      desc: "The underlying structural reason a problem keeps happening, as opposed to its surface symptom.",
      application: "Fixing a symptom (add servers) without the root cause (no capacity planning) means the problem returns."
    }
  ],

  vocab: [
    {
      term: "Problem Statement",
      define: "A Problem Statement: a specific description of who is affected, what's observed, and why it matters \u2014 the target the rest of the process aims at.",
      clues: [
        "I name who's affected, what's happening, and why it matters.",
        "I should be specific enough that two teams picture the same problem.",
        "\"Users are unhappy\" is not a good version of me."
      ]
    },
    {
      term: "5 Whys",
      define: "The 5 Whys: a technique of repeatedly asking \"why\" to move past a symptom and reach an actionable root cause.",
      clues: [
        "I'm a technique, not a single question.",
        "I ask the same short word, repeatedly, usually five times.",
        "I take you from a symptom down to a root cause."
      ]
    },
    {
      term: "Root Cause",
      define: "Root Cause: the underlying structural reason a problem recurs, as distinct from its visible symptom.",
      clues: [
        "I'm the structural reason a problem keeps happening.",
        "Fix a symptom instead of me, and the problem comes back.",
        "I'm usually found at the bottom of a 5 Whys chain."
      ]
    },
    {
      term: "Solution-Shaped Problem",
      define: "A Solution-Shaped Problem: a \"problem\" statement that actually names a solution, hiding the real underlying need.",
      clues: [
        "I look like a problem statement but I'm secretly a feature request.",
        "\"We need a chatbot\" is a classic example of me.",
        "Rewriting me means asking what pain the feature was meant to fix."
      ]
    },
    {
      term: "Problem Space",
      define: "Problem Space: the territory of what's broken and why, explored before moving into how to fix it (the solution space).",
      clues: [
        "I come before the solution space.",
        "I'm about what's broken and why, not how to fix it.",
        "Teams that skip me end up debating features before agreeing on the pain."
      ]
    },
    {
      term: "Confirmation Bias",
      define: "Confirmation Bias: the tendency to seek and favor evidence that confirms what you already believe, distorting problem framing.",
      clues: [
        "I make you notice evidence that agrees with you.",
        "I quietly ignore evidence that doesn't.",
        "I'm dangerous during problem framing because I can validate the wrong problem."
      ]
    }
  ]
};
