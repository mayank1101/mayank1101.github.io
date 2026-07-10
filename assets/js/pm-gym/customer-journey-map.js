// PM Gym — Customer Journey Map page data and Journey Insight builder.

window.pmData = {
  flashcards: [
    {
      category: 'Definition',
      title: 'What Is a Customer Journey Map?',
      desc:
        "A visual timeline of everything a customer does, thinks, and feels while trying to accomplish a goal with your product — from first hearing about it to becoming (or not becoming) a loyal user.",
      application:
        'One shared map keeps design, engineering, and marketing arguing about the same reality instead of five different imagined ones.'
    },
    {
      category: 'Anatomy',
      title: 'Stages and Swimlanes',
      desc:
        'Columns are the journey stages (awareness → consideration → purchase → onboarding → usage → loyalty). Rows are swimlanes: actions, thoughts, emotions, pain points, and opportunities at each stage.',
      application:
        'Reading down a column tells you everything about one moment; reading across a row shows how one dimension (like emotion) evolves.'
    },
    {
      category: 'Key Concept',
      title: 'The Emotion Curve',
      desc:
        'A line across the map tracking how the customer feels at each stage. The deepest dips mark the moments most likely to lose the customer — and the biggest opportunities.',
      application:
        'A food-delivery emotion curve often dips hardest at "waiting with no updates" — which is why live order tracking became standard.'
    },
    {
      category: 'Key Concept',
      title: 'Moments of Truth',
      desc:
        'The handful of make-or-break interactions where the customer decides to continue or abandon: first impression, first purchase, first failure, first support contact.',
      application:
        'Prioritize fixes at moments of truth over average moments — a great recovery from a first failure creates more loyalty than ten smooth ordinary sessions.'
    },
    {
      category: 'Method',
      title: 'Map Reality, Not Hope',
      desc:
        'A journey map must describe the journey as it actually is (backed by research and analytics), not the ideal journey you wish users had. Fantasy maps produce fantasy roadmaps.',
      application:
        'Build the "current state" map first from real data; only then draw a separate "future state" map as the target.'
    }
  ],

  vocab: [
    {
      term: "Journey Map",
      define: "A Customer Journey Map: a timeline of what a customer does, thinks, and feels across the whole experience.",
      clues: [
        "I show a product from the outside, over time.",
        "I lay out what a customer does, thinks, and feels.",
        "Stages across the top, swimlanes down the side \u2014 that's me."
      ]
    },
    {
      term: "Stage",
      define: "A Stage: one phase of the journey (awareness, purchase, onboarding\u2026), shown as a column.",
      clues: [
        "I'm a column on the map.",
        "I'm one phase of the journey in time order.",
        "Awareness, onboarding, loyalty \u2014 each is one of me."
      ]
    },
    {
      term: "Swimlane",
      define: "A Swimlane: a row tracking one dimension (actions, thoughts, emotions, pains) across all stages.",
      clues: [
        "I'm a row on the map.",
        "I track one dimension across every stage.",
        "Actions, thoughts, emotions, pains \u2014 each is one of me."
      ]
    },
    {
      term: "Emotion Curve",
      define: "The Emotion Curve: how the customer feels across the journey; the dips flag where you lose them.",
      clues: [
        "I'm the most valuable line on the map.",
        "I rise and fall with how the customer feels.",
        "My deepest dip is where customers quietly leave."
      ]
    },
    {
      term: "Moment of Truth",
      define: "A Moment of Truth: a decisive interaction (first value, first failure\u2026) that makes or breaks loyalty.",
      clues: [
        "I'm a make-or-break interaction.",
        "I decide whether the relationship continues.",
        "First impression, first failure, first support contact \u2014 each is one of me."
      ]
    }
  ]
};
