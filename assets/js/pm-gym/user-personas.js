// PM Gym — User Personas page data and Persona Builder tool.

window.pmData = {
  flashcards: [
    {
      category: 'Anatomy',
      title: 'The Empathy Thread',
      desc:
        "The rule that a persona's parts must fit together logically: the goals should relieve the exact pains, and the behaviors should show the person already trying to cope with those pains.",
      application:
        "If Rohan's pain is a brutal work schedule, his goal must be about saving time — not about hunting bulk discounts."
    },
    {
      category: 'Methodology',
      title: 'The Traceability Audit',
      desc:
        'A quality check where every bullet on the persona is linked back to a real interview quote or usage metric. Anything without a source gets deleted.',
      application:
        'If no user ever mentioned visuals in interviews, remove the line claiming they "value high aesthetics."'
    },
    {
      category: 'Pitfalls',
      title: 'The Fiction Trap',
      desc:
        'Inventing an idealized user who has no real struggles and conveniently loves every feature your team wants to build. A persona like this validates everything and guides nothing.',
      application:
        'Anchor every persona to real friction found in research — that is what stops the team from building generic features that delight no one.'
    },
    {
      category: 'Application',
      title: 'The Roadmap Tiebreaker',
      desc:
        "Personas earn their keep in debates. When designers and engineers disagree on what to build next, check which option maps best to the persona's core pains and goals.",
      application:
        'For a time-starved professional: a one-tap reorder button beats a coupon scheduler with a 10-minute setup — every time.'
    }
  ],

  vocab: [
    {
      term: "Persona",
      define: "A Persona: a research-based profile of a target user \u2014 their identity, behaviors, pains, and goals.",
      clues: [
        "I turn research data into someone you can picture.",
        "I balance identity, behaviors, pains, and goals.",
        "\"Rohan, 34, time-starved architect\" is me."
      ]
    },
    {
      term: "Empathy Thread",
      define: "The Empathy Thread: the logical consistency where a persona's goals relieve its stated pains.",
      clues: [
        "I'm the logic that holds a persona together.",
        "My rule: the goals must relieve the listed pains.",
        "If Rohan's pain is no time but his goal is bargain-hunting, I'm broken."
      ]
    },
    {
      term: "Traceability Audit",
      define: "The Traceability Audit: linking every persona detail back to a real quote or metric, and cutting the rest.",
      clues: [
        "I'm a quality check on every persona line.",
        "I ask 'where's the evidence?' for each trait.",
        "No source quote or metric? The line gets deleted \u2014 that's me."
      ]
    },
    {
      term: "Fiction Trap",
      define: "The Fiction Trap: inventing an idealized user with no real struggles who conveniently validates everything.",
      clues: [
        "I'm a mistake, not a method.",
        "I'm inventing a user with no real pains.",
        "A made-up customer who loves every backlog idea is me at work."
      ]
    },
    {
      term: "Cosmetic Bloat",
      define: "Cosmetic Bloat: lifestyle details that make a persona vivid but never change a product decision.",
      clues: [
        "I'm clutter on a persona card.",
        "I'm a detail that changes no decision.",
        "\"Loves oat-milk lattes\" on a fintech persona is me."
      ]
    }
  ]
};
