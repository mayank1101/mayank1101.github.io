// PM Gym — A/B Testing page data.

window.pmData = {
  flashcards: [
    {
      category: "Foundation",
      title: "A/B Testing",
      desc: "Randomly splitting traffic between a control (A) and a variant (B) to compare their effect on a chosen metric.",
      application: "Use it for isolated, well-defined changes with enough traffic to reach a reliable result."
    },
    {
      category: "Design",
      title: "Control Group",
      desc: "The unchanged version (A) that a variant is compared against, so any difference can be attributed to the change.",
      application: "Always keep a control group \u2014 without one, you can't tell if a result reflects the change or just normal variation over time."
    },
    {
      category: "Design",
      title: "Single Variable",
      desc: "Changing exactly one thing between control and variant, so any observed difference can be attributed to that one change.",
      application: "If you want to test a headline and a button color, run two separate tests, not one combined test."
    },
    {
      category: "Statistics",
      title: "Statistical Significance",
      desc: "Your confidence that an observed difference is real rather than random noise.",
      application: "A common threshold is p < 0.05: if there were truly no difference, you'd see a gap this big under 5% of the time by chance."
    },
    {
      category: "Pitfall",
      title: "Peeking",
      desc: "Checking a test's results early and repeatedly, then stopping as soon as it looks significant, which inflates false positives.",
      application: "Decide the sample size and duration in advance, and don't stop the test the moment it first looks like a win."
    },
    {
      category: "Pitfall",
      title: "Novelty Effect",
      desc: "A temporary lift caused by users reacting to anything new, which fades once the change stops feeling new.",
      application: "Run tests long enough to see past the initial spike before declaring a permanent winner."
    }
  ],

  vocab: [
    {
      term: "Control Group",
      define: "Control Group: the unchanged version (A) in an A/B test that a variant is compared against.",
      clues: [
        "I'm the unchanged version in a test.",
        "The variant gets compared against me.",
        "Without me, you can't tell if a result is real."
      ]
    },
    {
      term: "Statistical Significance",
      define: "Statistical Significance: a measure of how likely an observed difference reflects a real effect rather than random chance.",
      clues: [
        "I tell you how likely a result is due to chance.",
        "p < 0.05 is a common threshold for me.",
        "I don't tell you if an effect is big enough to matter."
      ]
    },
    {
      term: "Peeking",
      define: "Peeking: checking a test's results early and repeatedly, then stopping as soon as it appears significant \u2014 which inflates false positives.",
      clues: [
        "I happen when you check results too often.",
        "Stopping a test the moment I look good inflates false positives.",
        "Avoiding me means committing to a sample size and duration in advance."
      ]
    },
    {
      term: "Novelty Effect",
      define: "Novelty Effect: a temporary lift caused by users reacting to anything new, which fades as the change stops feeling new.",
      clues: [
        "I make any new thing look good, briefly.",
        "I fade once something stops feeling new.",
        "I can trick a team into declaring an early, temporary winner."
      ]
    },
    {
      term: "Sample Ratio Mismatch",
      define: "Sample Ratio Mismatch: when the actual traffic split between variants doesn't match what was intended, signaling a broken test setup.",
      clues: [
        "I show up when the split isn't actually what it was supposed to be.",
        "50/50 planned but 60/40 observed is a sign of me.",
        "I mean something in the test setup is broken."
      ]
    },
    {
      term: "Practical Significance",
      define: "Practical Significance: whether an effect, even if statistically real, is large enough to be worth the cost of acting on.",
      clues: [
        "I'm not the same thing as statistical significance.",
        "I ask if an effect is big enough to be worth acting on.",
        "A 'significant' result can still fail me if the effect is tiny."
      ]
    }
  ]
};
