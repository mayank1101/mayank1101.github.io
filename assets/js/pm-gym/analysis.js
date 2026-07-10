// PM Gym — Analysis page data.

window.pmData = {
  flashcards: [
    {
      category: "Foundation",
      title: "Analysis Mindset",
      desc: "Starting from a specific question tied to a decision, rather than browsing data with no particular goal.",
      application: "Before opening a dashboard, write down the exact question you're trying to answer."
    },
    {
      category: "Method",
      title: "Quantitative Analysis",
      desc: "Analyzing numbers at scale to measure what's happening and how much \u2014 strong on size and trend, weak on explaining why.",
      application: "Use it to find where a problem is, like a 40% drop-off at a specific funnel step."
    },
    {
      category: "Method",
      title: "Qualitative Analysis",
      desc: "Analyzing open-ended data like interviews or session recordings to understand why something is happening.",
      application: "Pair it with quantitative findings to explain the reasoning behind a number you've already found."
    },
    {
      category: "Framework",
      title: "Cohort Analysis",
      desc: "Grouping users by a shared starting point (like signup week) and tracking how their behavior changes over time.",
      application: "Use it to check whether retention is actually improving for newer users, which an overall flat average can hide."
    },
    {
      category: "Framework",
      title: "Funnel Analysis",
      desc: "Breaking a multi-step process into stages to find exactly where the biggest drop-off happens.",
      application: "Use it to locate the step in checkout or onboarding losing the most users, before digging into why."
    },
    {
      category: "Bias",
      title: "Correlation vs. Causation",
      desc: "Two things moving together doesn't mean one causes the other \u2014 both might be driven by a separate, shared factor.",
      application: "Before claiming X causes Y, check whether a third factor could be driving both."
    }
  ],

  vocab: [
    {
      term: "Cohort Analysis",
      define: "Cohort Analysis: grouping users by a shared starting point (like signup week) and tracking how their behavior changes over time.",
      clues: [
        "I group people by when they started, not by who they are.",
        "I track a group's behavior over time from that shared starting point.",
        "I reveal whether newer users retain better than older ones."
      ]
    },
    {
      term: "Funnel Analysis",
      define: "Funnel Analysis: breaking a multi-step process into stages to locate where the largest drop-off occurs.",
      clues: [
        "I break a process into sequential steps.",
        "I find exactly where the biggest drop-off happens.",
        "Checkout and onboarding are classic places to find me."
      ]
    },
    {
      term: "Correlation vs. Causation",
      define: "Correlation vs. Causation: the principle that two variables moving together doesn't prove one causes the other \u2014 a shared third factor might explain both.",
      clues: [
        "Two things moving together doesn't mean one of us causes the other.",
        "A hidden third factor can drive us both.",
        "Ice cream sales and drownings are my classic textbook example."
      ]
    },
    {
      term: "Confounding Variable",
      define: "Confounding Variable: a hidden third factor that influences two other variables, making them appear related when neither directly causes the other.",
      clues: [
        "I'm the hidden third factor.",
        "I make two unrelated things look connected.",
        "Hot weather driving both ice cream sales and drownings \u2014 I'm the weather."
      ]
    },
    {
      term: "Simpson's Paradox",
      define: "Simpson's Paradox: a trend that appears in several separate groups can reverse or disappear when those groups are combined, usually due to uneven group sizes.",
      clues: [
        "A trend can flip when you combine groups that had it individually.",
        "Uneven group sizes are usually behind me.",
        "I'm a warning to check the aggregate against the sub-groups."
      ]
    },
    {
      term: "Segmentation (in Analysis)",
      define: "Segmentation (in Analysis): slicing a metric by a meaningful dimension to check whether an aggregate number is hiding different sub-stories.",
      clues: [
        "I slice a metric by a meaningful dimension.",
        "Plan type, channel, or device are common ways to find me.",
        "I reveal when an aggregate number hides very different sub-stories."
      ]
    }
  ]
};
