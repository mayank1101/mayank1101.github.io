// PM Gym — Business Outcomes &amp; Product Outcomes page data.

window.pmData = {
  flashcards: [
    {
      category: "Core",
      title: "Output",
      desc: "What a team builds and ships \u2014 a feature, screen, or release. Easy to count, but no evidence on its own that anything improved.",
      application: "\"We launched a redesign\" is an output; whether it helped is a separate question."
    },
    {
      category: "Core",
      title: "Outcome",
      desc: "A measurable change in behavior or results created by the work \u2014 the thing that actually matters, as opposed to the work itself.",
      application: "\"Setup completion rose 30%\" is an outcome the redesign is meant to produce."
    },
    {
      category: "Type",
      title: "Business Outcome",
      desc: "A company-level result: revenue, retention rate, cost, market share. What the business ultimately cares about, but rarely movable directly by a product team.",
      application: "Annual revenue and churn rate are business outcomes \u2014 lagging and shaped by many forces."
    },
    {
      category: "Type",
      title: "Product Outcome",
      desc: "A change in user behavior that drives a business outcome: activation, weekly active use, tasks completed. The thing a product team can actually influence and should own.",
      application: "\"More users reach first value in week one\" is a product outcome that feeds retention."
    },
    {
      category: "Signal",
      title: "Leading Indicator",
      desc: "A metric that moves early and predicts a later result. Product outcomes are usually leading indicators \u2014 you steer by them.",
      application: "Rising weekly active use this month hints at higher retention next year."
    },
    {
      category: "Signal",
      title: "Lagging Indicator",
      desc: "A metric that confirms results after the fact, too late to act on directly. Business outcomes are usually lagging.",
      application: "Annual retention tells you what already happened \u2014 you can't steer by it in real time."
    }
  ],

  vocab: [
    {
      term: "Output",
      define: "Output: the work a team produces \u2014 features and releases \u2014 regardless of whether it changed anything.",
      clues: [
        "I'm what a team ships, not what changes.",
        "I'm a feature, a screen, a release.",
        "\"We launched a referral program\" is me."
      ]
    },
    {
      term: "Business Outcome",
      define: "Business Outcome: a company-level result like revenue or retention \u2014 usually lagging and hard to move directly.",
      clues: [
        "I'm what the company ultimately cares about.",
        "Revenue, retention, cost, market share \u2014 I'm one of these.",
        "Teams rarely move me directly."
      ]
    },
    {
      term: "Product Outcome",
      define: "Product Outcome: a change in user behavior that a product team can influence and that drives business results.",
      clues: [
        "I'm a change in what users do.",
        "I'm the behavior shift that drives the business result.",
        "Activation rate and weekly active use are me."
      ]
    },
    {
      term: "Leading Indicator",
      define: "Leading Indicator: a metric that moves early and predicts a later outcome, so you can steer by it.",
      clues: [
        "I move early.",
        "I predict a result before it fully arrives.",
        "Rising weekly active use hinting at future retention is me."
      ]
    },
    {
      term: "Lagging Indicator",
      define: "Lagging Indicator: a metric that confirms a result after it has happened, too late to steer by.",
      clues: [
        "I confirm things after the fact.",
        "I arrive too late to act on directly.",
        "Annual retention measured at year-end is me."
      ]
    },
    {
      term: "Feature Factory",
      define: "Feature Factory: a team that measures itself by output shipped rather than by outcomes achieved.",
      clues: [
        "I'm a team stuck in a bad habit.",
        "I measure success by shipping, not by impact.",
        "I ship endlessly and learn nothing about whether it helped."
      ]
    }
  ]
};
