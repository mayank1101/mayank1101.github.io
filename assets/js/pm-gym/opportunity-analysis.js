// PM Gym — Opportunity Analysis page data and RICE simulator.

window.pmData = {
  flashcards: [
    {
      category: 'Terminology',
      title: 'Opportunities vs. Solutions',
      desc:
        'An opportunity is a customer pain point, friction, or desire. A solution is the concrete feature designed to fix it. Beginners fall in love with solutions; good product thinking starts with opportunities.',
      application:
        "Don't fall in love with features — fall in love with the user's problem."
    },
    {
      category: 'Frameworks',
      title: 'The Opportunity Solution Tree',
      desc:
        'A visual map by Teresa Torres that flows top-down: Desired Outcome → Opportunities → Solutions → Experiments. It keeps every feature idea tied to a measurable goal.',
      application:
        'When a stakeholder pitches an idea, place it on the tree — does it connect to a verified user pain that serves the outcome?'
    },
    {
      category: 'Sizing',
      title: 'The Opportunity Score',
      desc:
        'Score = Importance + Max(Importance − Satisfaction, 0). It surfaces needs that matter a lot to users but are badly served today.',
      application:
        'The sweet spot: high importance, low satisfaction. If users are already happy with existing options, the score collapses back to plain importance.'
    },
    {
      category: 'Prioritization',
      title: 'The RICE Framework',
      desc:
        'RICE = (Reach × Impact × Confidence) ÷ Effort. Reach: how many people. Impact: how much each gains. Confidence: how sure you are. Effort: how long it takes.',
      application:
        'High RICE scores point to features that help many people a lot, with solid evidence, at low cost.'
    }
  ],

  vocab: [
    {
      term: "Opportunity",
      define: "An Opportunity: an unmet user pain or need \u2014 the problem, not the feature that fixes it.",
      clues: [
        "I live in the problem space.",
        "I'm an unmet user pain, need, or desire.",
        "\"Users forget to track their hydration\" is me \u2014 no feature named yet."
      ]
    },
    {
      term: "Solution",
      define: "A Solution: a concrete feature or implementation designed to address an opportunity.",
      clues: [
        "I live in the solution space.",
        "I'm a specific way to fix a pain.",
        "\"A Bluetooth water-bottle widget\" is me \u2014 a concrete build."
      ]
    },
    {
      term: "Opportunity Solution Tree",
      define: "The Opportunity Solution Tree: a map linking a desired outcome down through opportunities, solutions, and experiments.",
      clues: [
        "I'm a visual map from Teresa Torres.",
        "I flow top-down through four levels.",
        "Outcome, opportunities, solutions, experiments \u2014 that's me."
      ]
    },
    {
      term: "RICE",
      define: "RICE: a priority score = (Reach x Impact x Confidence) / Effort.",
      clues: [
        "I'm a four-letter prioritization formula.",
        "I divide three multiplied factors by a fourth.",
        "(Reach x Impact x Confidence) / Effort \u2014 that's me."
      ]
    },
    {
      term: "Confidence",
      define: "Confidence (in RICE): how much evidence backs your estimates \u2014 the check against wishful thinking.",
      clues: [
        "I'm one factor in a famous scoring formula.",
        "I'm the honesty gate against gut-feel estimates.",
        "Scoring me at 50% halves a hyped feature's priority score."
      ]
    },
    {
      term: "Opportunity Score",
      define: "Opportunity Score = Importance + Max(Importance - Satisfaction, 0); it surfaces important, poorly-served needs.",
      clues: [
        "I rank problems, not features.",
        "I reward high importance paired with low satisfaction.",
        "Importance + Max(Importance - Satisfaction, 0) is my formula."
      ]
    }
  ]
};
