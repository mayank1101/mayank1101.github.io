// PM Gym — User Segmentation page data and Segment Builder tool.

window.pmData = {
  flashcards: [
    {
      category: 'Terminology',
      title: 'The User',
      desc:
        'The person who actually operates the product day to day — clicking, scrolling, getting things done. Users care about speed, low friction, and a predictable interface.',
      application:
        'In the Spotify Kids example: Leo, the six-year-old tapping through dinosaur stories on the iPad, is the user.'
    },
    {
      category: 'Terminology',
      title: 'The Customer',
      desc:
        'The person or organization that pays. Customers care about price, security, compliance, and whether the purchase is worth it.',
      application:
        'Sarah, the parent entering her card details for the family plan, is the customer — even though she rarely opens the app herself.'
    },
    {
      category: 'Data Strategy',
      title: 'Trust Behavior over Surveys',
      desc:
        'What people do in your product is more reliable than what they say in surveys. Product decisions should lean on real usage logs, not self-reported traits.',
      application:
        'Track which features people actually use daily rather than relying on what they claim in an intake form.'
    },
    {
      category: 'Frameworks',
      title: 'Needs-Based Segmentation',
      desc:
        'Grouping people by the problem they share rather than by age, location, or income. Often reveals segments that demographics completely hide.',
      application:
        'The "morning commuters who need a clean, one-handed breakfast" segment — invisible if you slice by age or income.'
    },
    {
      category: 'Quality Check',
      title: 'The MECE Rule',
      desc:
        'Mutually Exclusive, Collectively Exhaustive: segments should not overlap, and together they should cover everyone. Overlapping segments produce double-counted metrics and muddled strategy.',
      application:
        '"Active free users" vs. "active paying users" is MECE. "Young users" vs. "power users" is not — many people are both.'
    }
  ],

  vocab: [
    {
      term: "User",
      define: "The User: the person who actually operates the product day to day.",
      clues: [
        "I'm one half of a classic split.",
        "I use the product every day.",
        "In Spotify Kids, six-year-old Leo tapping the app is me."
      ]
    },
    {
      term: "Customer",
      define: "The Customer: the person or organization that pays and decides to buy.",
      clues: [
        "I'm the other half of that split.",
        "I hold the budget and make the buying call.",
        "Sarah entering her card for the family plan is me."
      ]
    },
    {
      term: "Demographics",
      define: "Demographic segmentation: grouping by fixed traits like age, income, or job title.",
      clues: [
        "I slice people by fixed facts.",
        "Age, income, job title, industry \u2014 that's my toolkit.",
        "Great for sizing markets, weak for guiding features."
      ]
    },
    {
      term: "Behavioral",
      define: "Behavioral segmentation: grouping by observed actions like usage frequency and features used.",
      clues: [
        "I slice by what people actually do.",
        "Usage frequency, features touched, purchase patterns \u2014 my signals.",
        "Product teams love me because actions don't lie."
      ]
    },
    {
      term: "Needs-Based",
      define: "Needs-based segmentation: grouping people by the shared problem they're trying to solve.",
      clues: [
        "I ignore who people are and focus on their problem.",
        "I group by a shared struggle, not shared traits.",
        "\"Commuters who need a clean one-handed breakfast\" is my kind of group."
      ]
    },
    {
      term: "MECE",
      define: "MECE: segments that don't overlap (mutually exclusive) and together cover everyone (collectively exhaustive).",
      clues: [
        "I'm a four-letter quality check for your groups.",
        "No overlaps, no gaps.",
        "Mutually Exclusive, Collectively Exhaustive \u2014 that's me."
      ]
    }
  ]
};
