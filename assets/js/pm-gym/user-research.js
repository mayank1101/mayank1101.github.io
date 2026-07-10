// PM Gym — 5-Step User Research page data and Research Plan builder.

window.pmData = {
  flashcards: [
    {
      category: 'Process',
      title: 'The 5 Steps',
      desc:
        'Define objectives → Choose the method → Recruit participants → Conduct the research → Synthesize and share. Skipping a step usually means redoing it later, more expensively.',
      application:
        'The order matters: the method depends on the objective, recruitment depends on the method, and synthesis is planned before the first session.'
    },
    {
      category: 'Step 1',
      title: 'Research Objective',
      desc:
        'One sharp question the research must answer, plus the decision that hangs on it. "Learn about users" is not an objective; "find out why trial users don\'t convert, to decide what onboarding fixes" is.',
      application:
        'Test an objective by asking: what will we decide differently based on the answer? No decision, no research.'
    },
    {
      category: 'Step 2',
      title: 'Matching Method to Question',
      desc:
        'Why/how questions → interviews and field studies. Can-they-use-it questions → usability tests. How-many questions → surveys and analytics. Which-is-better questions → A/B tests.',
      application:
        'A method chosen before the question is a hammer looking for nails. Write the question first, then pick the tool it needs.'
    },
    {
      category: 'Step 3',
      title: 'Recruiting the Right People',
      desc:
        'Five users from your actual target segment beat fifty random people. Screen for the behavior you\'re studying, not just demographics — and never test only on teammates or friends.',
      application:
        'For a study on churned users, recruit people who actually canceled — not current happy users guessing why others left.'
    },
    {
      category: 'Step 4',
      title: 'Asking Without Leading',
      desc:
        'Ask about past behavior, not future intent ("walk me through the last time you…"). Ask open questions. Never ask "would you use this?" — the answer is polite noise.',
      application:
        'Replace "would you pay for this feature?" with "tell me about the last time you paid for a tool — what convinced you?"'
    },
    {
      category: 'Step 5',
      title: 'Synthesize and Share',
      desc:
        'Research that stays in a folder changed nothing. Cluster observations into themes, write insight statements, and share them with the evidence (quotes, clips) attached.',
      application:
        'A 30-second clip of a real user failing at checkout persuades a room faster than a 30-page report.'
    }
  ],

  vocab: [
    {
      term: "Research Objective",
      define: "A Research Objective: one clear question plus the decision that depends on its answer.",
      clues: [
        "I'm step one of the process.",
        "I pair a sharp question with a decision.",
        "\"Why do trials cancel, to decide onboarding fixes\" is me."
      ]
    },
    {
      term: "Usability Test",
      define: "A Usability Test: watching real users attempt real tasks to find where they struggle.",
      clues: [
        "I'm a method for one kind of question.",
        "I answer 'can people actually do it?'",
        "About five users on a real task finds most of my issues."
      ]
    },
    {
      term: "Convenience Sample",
      define: "A Convenience Sample: participants chosen for ease of access, not fit \u2014 and usually biased.",
      clues: [
        "I'm a recruiting trap.",
        "I'm made of whoever is easiest to reach.",
        "Teammates, friends, and superfans are me \u2014 biased by definition."
      ]
    },
    {
      term: "Leading Question",
      define: "A Leading Question: one phrased to nudge the participant toward a particular answer.",
      clues: [
        "I quietly ruin interview data.",
        "I hint at the answer I want.",
        "\"Don't you think this is easier?\" is me."
      ]
    },
    {
      term: "Screening",
      define: "Screening: filtering recruits so participants actually match the behavior you're studying.",
      clues: [
        "I happen before the session.",
        "I make sure participants fit the study.",
        "I filter by real behavior, not just demographics."
      ]
    },
    {
      term: "Synthesis",
      define: "Synthesis: turning raw research into themes, insight statements, and shareable findings.",
      clues: [
        "I'm the final step of research.",
        "I turn raw sessions into themes and decisions.",
        "Cluster, write insight statements, share with clips \u2014 that's me."
      ]
    }
  ]
};
