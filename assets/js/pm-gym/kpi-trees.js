// PM Gym — KPI Trees page data.

window.pmData = {
  flashcards: [
    {
      category: "Core",
      title: "KPI Tree",
      desc: "A breakdown of one top-level goal metric into the sub-metrics that mathematically combine to produce it \u2014 turning a blunt number into a map of pullable levers.",
      application: "Revenue = Active Users \u00d7 Conversion \u00d7 ARPU splits one goal into three targetable drivers."
    },
    {
      category: "Top",
      title: "North Star Metric",
      desc: "The single metric that best captures the value your product delivers \u2014 usually the root of the KPI tree that everything else feeds.",
      application: "For a messaging app, 'messages sent between connected users' might be the North Star."
    },
    {
      category: "Branch",
      title: "Input Metric (Driver)",
      desc: "A sub-metric that feeds a metric above it in the tree. Inputs are the levers teams actually work on.",
      application: "Conversion rate is an input metric that drives revenue at the level above."
    },
    {
      category: "Branch",
      title: "Output Metric",
      desc: "A higher-level result produced by its input metrics \u2014 the parent in a branch. You move it only by moving its inputs.",
      application: "Revenue is an output metric; you don't move it directly, you move its inputs."
    },
    {
      category: "Caution",
      title: "Vanity Metric",
      desc: "A number that looks impressive and tends to only go up (like all-time sign-ups), but doesn't reflect current health or respond to decisions.",
      application: "'Total downloads ever' rises forever even as a product dies \u2014 avoid it as a core KPI."
    },
    {
      category: "Quality",
      title: "Actionable Metric",
      desc: "A metric a specific decision can move and that can go down as well as up \u2014 the kind worth putting in a KPI tree.",
      application: "Weekly active users falls when the product slips and rises when it improves \u2014 actionable."
    }
  ],

  vocab: [
    {
      term: "KPI Tree",
      define: "A KPI Tree: a top-down breakdown of a goal metric into the sub-metrics that mathematically produce it.",
      clues: [
        "I turn one blunt number into many levers.",
        "I break a goal metric into the drivers that compose it.",
        "\"Revenue = Users \u00d7 Conversion \u00d7 ARPU\" is me in action."
      ]
    },
    {
      term: "North Star Metric",
      define: "A North Star Metric: the single metric that best captures your product's value, at the top of the tree.",
      clues: [
        "I usually sit at the root of the tree.",
        "I capture the core value the product delivers.",
        "Everything else in the tree feeds me."
      ]
    },
    {
      term: "Input Metric",
      define: "An Input Metric (driver): a sub-metric that feeds a higher metric \u2014 the lever teams actually work.",
      clues: [
        "I'm a lever, not a result.",
        "I feed a metric above me in the tree.",
        "Conversion rate driving revenue is me."
      ]
    },
    {
      term: "Vanity Metric",
      define: "A Vanity Metric: an impressive-looking number that only rises and doesn't reflect real health.",
      clues: [
        "I look great in a slide.",
        "I tend to only ever go up.",
        "\"Total sign-ups since launch\" is me \u2014 I hide the truth."
      ]
    },
    {
      term: "Actionable Metric",
      define: "An Actionable Metric: one a decision can move and that can fall as well as rise \u2014 worth tracking.",
      clues: [
        "I respond to your decisions.",
        "I can go down as well as up.",
        "Weekly active users falling when the product slips is me."
      ]
    },
    {
      term: "MECE",
      define: "MECE: branch inputs that don't overlap and leave no gap, so they truly compose their parent.",
      clues: [
        "I'm a four-letter test for clean branches.",
        "No overlaps, no gaps.",
        "Mutually Exclusive, Collectively Exhaustive \u2014 that's me."
      ]
    }
  ]
};
