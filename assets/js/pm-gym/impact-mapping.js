// PM Gym — Impact Mapping page data and Alignment Statement builder.

window.pmData = {
  flashcards: [
    {
      category: 'Mindset',
      title: "The Golden Rule of Impact Mapping",
      desc:
        "Never ask 'what should we build?' until you know whose behavior needs to change to reach the business goal. Features come last, not first.",
      application:
        "Watch for surprising competitors for attention. Duolingo isn't just competing with Rosetta Stone — it competes with mobile games and social feeds."
    },
    {
      category: 'Building Blocks',
      title: 'Goals vs. Deliverables',
      desc:
        "A goal is a business result you can measure. A deliverable is just the feature you ship. Mixing them up is the most common beginner mistake.",
      application:
        "Weak: 'Our goal is to build an onboarding flow.' Strong: 'Our goal is to increase day-7 retention by 15% by Q3.'"
    },
    {
      category: 'Actors',
      title: 'Actors Are More Than "the User"',
      desc:
        'Actors are everyone who can move your metric: customer segments, but also support agents, delivery partners, or sales teams.',
      application:
        "Uber: if drivers (actors) can't find parking, your pickup-time metric suffers — no matter how polished the rider app is."
    },
    {
      category: 'Impacts',
      title: 'What Counts as an Impact?',
      desc:
        'A real, observable change in human behavior — someone doing something more, less, or differently. Not a feature, not a click.',
      application:
        "Weak: 'Users utilize our notifications.' Strong: 'Casual joggers open the app daily to log their runs.'"
    },
    {
      category: 'Prioritization',
      title: 'The Alignment Guard',
      desc:
        "Every feature request must trace back through a behavior change to the goal. If the trace line breaks, the feature is probably waste.",
      application:
        'A polite way to challenge a pet feature: "Which behavior change does this drive, and which goal does that serve?"'
    },
    {
      category: 'Thinking Tool',
      title: 'The Right-to-Left Check',
      desc:
        "Read your map backwards to verify it: 'By building this feature, will this person change this behavior, and will that move this goal?'",
      application:
        'Walking through the map backwards out loud is a great way to pressure-test any product plan before committing to it.'
    }
  ],

  vocab: [
    {
      term: "Goal",
      define: "The Goal: the measurable business outcome the whole map traces back to.",
      clues: [
        "I'm the first link in the chain \u2014 the 'why'.",
        "I'm the measurable business result you're aiming for.",
        "\"Grow premium subscribers 15% by Q3\" is an example of me."
      ]
    },
    {
      term: "Actor",
      define: "Actors: the people whose behavior can make the goal succeed or fail.",
      clues: [
        "I'm the 'who' of the map.",
        "I'm a person whose behavior can move the goal.",
        "Not just 'the user' \u2014 I might be a support agent or a delivery driver."
      ]
    },
    {
      term: "Impact",
      define: "An Impact: the behavior change in an actor that moves the goal \u2014 never a feature or a click.",
      clues: [
        "I'm the 'how' \u2014 a change, not a thing.",
        "I'm a shift in what a person actually does.",
        "\"Drivers spend less time hunting for parking\" is me \u2014 a behavior change, not a feature."
      ]
    },
    {
      term: "Deliverable",
      define: "A Deliverable: the feature or tool you ship to trigger a behavior change.",
      clues: [
        "I'm the last link \u2014 the 'what'.",
        "I'm the thing engineers actually ship.",
        "\"A lyric-export button\" is me \u2014 built only to cause an impact."
      ]
    },
    {
      term: "Impact Map",
      define: "An Impact Map: a visual chain from goal to actor to impact to deliverable that keeps features tied to outcomes.",
      clues: [
        "I'm the whole picture, not one link.",
        "I connect business goals to features through people and behavior.",
        "My four levels are goal, actor, impact, deliverable."
      ]
    }
  ]
};
