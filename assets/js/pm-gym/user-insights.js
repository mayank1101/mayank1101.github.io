// PM Gym — Driving User Insights page data and Insight Statement builder.

window.pmData = {
  flashcards: [
    {
      category: 'Definition',
      title: 'Data vs. Insight',
      desc:
        'Data is a fact ("40% of users abandon checkout"). An insight explains the why behind the fact and points at what to do ("users abandon because the shipping cost appears only at the last step, feeling like a trap").',
      application:
        'Test any claimed "insight": does it explain a cause and suggest an action? If it just restates a number, it\'s still data.'
    },
    {
      category: 'Sources',
      title: 'Quantitative + Qualitative',
      desc:
        'Quantitative data (analytics, funnels, surveys at scale) tells you WHAT is happening and how much. Qualitative data (interviews, session recordings, support tickets) tells you WHY.',
      application:
        'Analytics show 40% drop at checkout; five user interviews reveal the surprise shipping fee. Neither alone is enough — pair them.'
    },
    {
      category: 'Method',
      title: 'Affinity Mapping',
      desc:
        'A synthesis technique: write every observation on a separate note, then cluster similar notes into groups. The clusters that emerge are your candidate themes.',
      application:
        'After ten interviews, cluster 200 sticky notes; if 30 notes land in a "distrust of automatic payments" cluster, you found a theme worth digging into.'
    },
    {
      category: 'Format',
      title: 'The Insight Statement',
      desc:
        '"[User group] does/feels [observation], because [underlying reason], which means [implication for the product]." Observation + cause + consequence, in one sentence.',
      application:
        '"New users skip the tutorial because it delays their first real task, which means the tutorial should teach through the task instead of before it."'
    },
    {
      category: 'Pitfalls',
      title: 'Confirmation Bias',
      desc:
        'The strongest gravitational pull in research: noticing only the evidence that supports what you already believe. Insights born this way just launder opinions into "findings."',
      application:
        'Actively hunt for disconfirming evidence — ask "what would prove me wrong?" and weight surprises more heavily than confirmations.'
    }
  ],

  vocab: [
    {
      term: "Data",
      define: "Data: a fact or measurement that describes what happened, without explaining why.",
      clues: [
        "I'm just a fact on my own.",
        "I describe what happened, not why.",
        "\"40% abandon checkout\" is me \u2014 no explanation attached."
      ]
    },
    {
      term: "Insight",
      define: "An Insight: an explanation of why something happens, sharp enough to suggest what to do.",
      clues: [
        "I go deeper than a fact.",
        "I explain the why and point to an action.",
        "\"They abandon because the shipping fee feels like a trap\" is me."
      ]
    },
    {
      term: "Quantitative",
      define: "Quantitative research: analytics and surveys that measure what is happening and at what scale.",
      clues: [
        "I'm one of two research flavors.",
        "I measure what and how many.",
        "Analytics, funnels, and big surveys are my tools."
      ]
    },
    {
      term: "Qualitative",
      define: "Qualitative research: interviews and observation that explain why users behave as they do.",
      clues: [
        "I'm the other research flavor.",
        "I explain the why behind the numbers.",
        "Interviews, session recordings, and support tickets are my tools."
      ]
    },
    {
      term: "Affinity Mapping",
      define: "Affinity Mapping: clustering individual observations into emergent themes during synthesis.",
      clues: [
        "I'm a synthesis technique.",
        "I turn scattered notes into themes.",
        "Cluster similar notes, then name the group after it forms \u2014 that's me."
      ]
    },
    {
      term: "Say/Do Gap",
      define: "The Say/Do Gap: stated intentions routinely overstate what people will actually do.",
      clues: [
        "I'm the reason surveys mislead.",
        "I'm the space between stated intent and real behavior.",
        "\"I'd definitely use this\" \u2014 then never clicking \u2014 is me."
      ]
    }
  ]
};
