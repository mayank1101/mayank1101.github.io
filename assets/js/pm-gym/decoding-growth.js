// PM Gym — Decoding Growth page data.

window.pmData = {
  flashcards: [
    {
      category: "Foundation",
      title: "Growth",
      desc: "The net effect of users coming in (acquisition) and users leaving (churn) \u2014 not just the count of new signups.",
      application: "When a growth number looks good, always ask what's happening to retention underneath it."
    },
    {
      category: "Model",
      title: "The Leaky Bucket",
      desc: "A metaphor: acquisition pours users into a bucket while churn leaks them out. If the holes are big, pouring faster barely raises the level.",
      application: "Before spending more on acquisition, check whether you're pouring into a leaky bucket \u2014 patch retention first."
    },
    {
      category: "Framework",
      title: "AARRR (Pirate Metrics)",
      desc: "Five growth stages: Acquisition, Activation, Retention, Referral, Revenue \u2014 each a place to measure and improve.",
      application: "Diagnose where growth breaks by finding which AARRR stage has the biggest drop-off."
    },
    {
      category: "Stage",
      title: "Activation",
      desc: "The user's first real value moment \u2014 the \"aha\" where the product clicks \u2014 distinct from merely signing up.",
      application: "A big gap between signups and activated users means people arrive but the product never clicks; fix the first-value experience."
    },
    {
      category: "Engine",
      title: "Retention",
      desc: "Whether users keep coming back over time. A flattening retention curve is a strong product-market-fit signal and the engine that makes growth compound.",
      application: "Plot retention by cohort; a curve that flattens (vs. sliding to zero) means each new cohort stacks onto a durable base."
    },
    {
      category: "Discipline",
      title: "Earned vs. Bought Growth",
      desc: "Earned growth (retention, referral, loops) keeps working after you stop pushing; bought growth (paid ads) stops the moment spend stops.",
      application: "Treat paid acquisition as fuel, not the engine \u2014 and only run it while a customer is worth more than they cost to acquire."
    }
  ],

  vocab: [
    {
      term: "Leaky Bucket",
      define: "The Leaky Bucket: a metaphor where acquisition pours users in while churn leaks them out, so growth stalls unless retention holds them.",
      clues: [
        "I'm a metaphor about pouring water and losing it.",
        "My holes are churn; my inflow is acquisition.",
        "Pour faster into me and the level barely rises."
      ]
    },
    {
      term: "AARRR",
      define: "AARRR (Pirate Metrics): five growth stages \u2014 Acquisition, Activation, Retention, Referral, Revenue \u2014 each a place to measure and improve.",
      clues: [
        "My nickname involves pirates.",
        "My five letters are stages of growth.",
        "Acquisition and Activation are my first two."
      ]
    },
    {
      term: "Activation",
      define: "Activation: the user's first real value moment, distinct from simply signing up or arriving.",
      clues: [
        "I'm a first, not a repeat.",
        "I'm the \"aha\" moment where the product clicks.",
        "Signing up isn't me \u2014 reaching real value is."
      ]
    },
    {
      term: "Retention Curve",
      define: "Retention Curve: the percentage of a signup cohort still active over time \u2014 a flattening curve signals a durable base and product-market fit.",
      clues: [
        "I'm a line that falls over time.",
        "If I flatten, you have a durable base; if I hit zero, you don't.",
        "My flattening is a product-market-fit signal."
      ]
    },
    {
      term: "Sustainable Growth",
      define: "Sustainable (Earned) Growth: growth driven by the product itself \u2014 retention, referral, loops \u2014 that persists after active spending stops.",
      clues: [
        "I keep working after you stop pushing.",
        "I come from retention, referral, and loops, not paid ads alone.",
        "I'm earned, not rented."
      ]
    },
    {
      term: "Churn",
      define: "Churn: the rate at which users stop using a product \u2014 the outflow that acquisition has to outrun for growth to happen.",
      clues: [
        "I'm the outflow, not the inflow.",
        "I'm the holes in the leaky bucket.",
        "Lower me and the same acquisition fills the bucket faster."
      ]
    }
  ]
};
