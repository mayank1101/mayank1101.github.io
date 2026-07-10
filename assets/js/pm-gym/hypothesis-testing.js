// PM Gym — Hypothesis Testing page data.

window.pmData = {
  flashcards: [
    {
      category: "Foundation",
      title: "Hypothesis",
      desc: "A falsifiable statement about what will happen if you make a specific change \u2014 precise enough that a real result could prove it wrong.",
      application: "Before building a feature, write the belief behind it as a hypothesis so you can actually check if it holds."
    },
    {
      category: "Anatomy",
      title: "Hypothesis Template",
      desc: "\"We believe [X] for [group] will result in [outcome]. We'll know we're right if [measurable signal].\"",
      application: "Use this template to force vague ideas (\"this will help users\") into something testable."
    },
    {
      category: "Risk",
      title: "Leap of Faith Assumption",
      desc: "The assumption underneath an idea that is both highly uncertain and, if wrong, would sink the whole idea.",
      application: "Test your leap of faith assumptions first \u2014 they tell you fastest whether an idea is worth pursuing further."
    },
    {
      category: "Discipline",
      title: "Success Criteria",
      desc: "The specific threshold, decided before a test runs, that determines whether the result counts as a pass or a fail.",
      application: "Agree \"we need 8% conversion to proceed\" before launching the test, not after seeing the result."
    },
    {
      category: "Bias",
      title: "Confirmation Bias",
      desc: "The tendency to notice and favor results that support what you already wanted to believe.",
      application: "Pre-registering success criteria is one of the best defenses against reading a test result the way you want to."
    },
    {
      category: "Concept",
      title: "Falsifiable",
      desc: "Able to be proven wrong by a real result \u2014 a hallmark of a genuine hypothesis as opposed to a vague hope.",
      application: "\"This might help\" isn't falsifiable; \"this will raise completion by 10%\" is."
    }
  ],

  vocab: [
    {
      term: "Hypothesis",
      define: "A Hypothesis: a falsifiable statement about what will happen if you make a specific change.",
      clues: [
        "I'm a bet you can lose.",
        "A real result can prove me wrong \u2014 that's the whole point of me.",
        "\"We believe X will result in Y\" is roughly my shape."
      ]
    },
    {
      term: "Leap of Faith Assumption",
      define: "Leap of Faith Assumption: the assumption that is both most uncertain and most critical to an idea's success \u2014 test it first.",
      clues: [
        "I'm the riskiest belief underneath an idea.",
        "If I'm wrong, the whole idea collapses.",
        "Test me before anything safer."
      ]
    },
    {
      term: "Falsifiable",
      define: "Falsifiable: stated precisely enough that a real-world result could prove it wrong.",
      clues: [
        "A real result can knock me down.",
        "\"Might help somehow\" doesn't have me; a specific number does.",
        "I'm what separates a hypothesis from a vague hope."
      ]
    },
    {
      term: "Success Criteria",
      define: "Success Criteria: the threshold, set in advance, that determines whether a test result counts as pass or fail.",
      clues: [
        "I'm decided before the test runs, not after.",
        "Without me, any result can be spun as a win.",
        "\"We need 8% to proceed\" is an example of me."
      ]
    },
    {
      term: "Assumption Mapping",
      define: "Assumption Mapping: plotting the assumptions behind an idea by risk and uncertainty to find which to test first.",
      clues: [
        "I sort assumptions by risk and uncertainty.",
        "I help you find the leap of faith among many smaller bets.",
        "I usually get drawn as a 2x2 grid."
      ]
    },
    {
      term: "Confirmation Bias",
      define: "Confirmation Bias: the tendency to favor evidence that confirms an existing belief, distorting how test results are read.",
      clues: [
        "I make you see what you already believed.",
        "I quietly discount evidence you don't like.",
        "Pre-set success criteria help defend against me."
      ]
    }
  ]
};
