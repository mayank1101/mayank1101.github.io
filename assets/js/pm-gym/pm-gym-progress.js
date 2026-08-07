// PM Gym — progress storage (localStorage) + shared a11y touch-ups.
// Loaded on EVERY PM Gym page. This file owns the storage format only; the
// topic pages record through window.pmProgress and the hub reads the same API
// to render badges, bars and weak-topic hints.
//
// Shape of a stored topic:
//   { visited, completed, lesson, quiz: { q3: {correct, attempts} },
//     cards: { "2": 1 }, game: { plays, best, max } }
(function () {
  'use strict';

  var KEY = 'pmgym:progress:v2';
  var LEGACY_KEY = 'pmgym:progress:v1';
  var DAY = 24 * 60 * 60 * 1000;
  var BOX_DAYS = [0, 1, 3, 7, 21]; // Leitner intervals, box 1 through 5

  function blankTopic() {
    return {
      visited: 0,
      completed: 0,
      lesson: 1,
      quiz: {},
      cards: {},
      game: { plays: 0, best: 0, max: 0 }
    };
  }

  function normalise(raw) {
    var data = raw && typeof raw === 'object' ? raw : {};
    data.v = 2;
    data.topics = data.topics && typeof data.topics === 'object' ? data.topics : {};
    Object.keys(data.topics).forEach(function (slug) {
      var t = data.topics[slug] || {};
      var base = blankTopic();
      base.visited = t.visited || 0;
      base.completed = t.completed || 0;
      base.lesson = t.lesson || 1;
      base.quiz = t.quiz && typeof t.quiz === 'object' ? t.quiz : {};
      base.cards = t.cards && typeof t.cards === 'object' ? t.cards : {};
      // Early builds stored a bare review count per card; Leitner needs a box
      // and a due date, so lift those numbers into the richer shape.
      Object.keys(base.cards).forEach(function (i) {
        if (typeof base.cards[i] === 'number') {
          base.cards[i] = { seen: base.cards[i], box: 1, due: 0 };
        }
      });
      base.game = t.game && typeof t.game === 'object' ? t.game : base.game;
      data.topics[slug] = base;
    });
    return data;
  }

  // v1 stored only two flat maps of timestamps. Carry them across so returning
  // learners don't lose their completed badges.
  function migrateLegacy(data) {
    var raw;
    try {
      raw = JSON.parse(localStorage.getItem(LEGACY_KEY));
    } catch (e) {
      return data;
    }
    if (!raw) return data;
    ['visited', 'completed'].forEach(function (field) {
      var map = raw[field] || {};
      Object.keys(map).forEach(function (slug) {
        var t = data.topics[slug] || (data.topics[slug] = blankTopic());
        if (!t[field]) t[field] = map[slug];
      });
    });
    return data;
  }

  function load() {
    var data;
    try {
      data = normalise(JSON.parse(localStorage.getItem(KEY)));
    } catch (e) {
      data = normalise(null);
    }
    if (!localStorage.getItem(KEY)) data = migrateLegacy(data);
    return data;
  }

  function save(data) {
    try {
      localStorage.setItem(KEY, JSON.stringify(data));
    } catch (e) {
      /* storage unavailable (private mode / disabled) — degrade silently */
    }
    return data;
  }

  function update(slug, fn) {
    var data = load();
    var topic = data.topics[slug] || (data.topics[slug] = blankTopic());
    fn(topic);
    return save(data).topics[slug];
  }

  function currentSlug() {
    var m = location.pathname.match(/\/pm-gym\/([^\/]+)\.html?$/);
    return m ? m[1] : null;
  }

  function scored(topic) {
    var ids = Object.keys(topic.quiz);
    var correct = 0;
    ids.forEach(function (id) {
      if (topic.quiz[id].correct) correct++;
    });
    return { answered: ids.length, correct: correct };
  }

  window.pmProgress = {
    key: KEY,
    load: load,
    save: save,
    slug: currentSlug,
    blankTopic: blankTopic,

    topic: function (slug) {
      return load().topics[slug] || blankTopic();
    },

    markVisited: function (slug) {
      return update(slug, function (t) {
        if (!t.visited) t.visited = Date.now();
      });
    },

    // Only the FIRST attempt counts — re-answering a question you already got
    // wrong shouldn't repaint it as knowledge you had.
    recordAnswer: function (slug, qid, correct) {
      return update(slug, function (t) {
        var entry = t.quiz[qid];
        if (!entry) {
          t.quiz[qid] = { correct: !!correct, attempts: 1 };
        } else {
          entry.attempts++;
        }
      });
    },

    recordLesson: function (slug, lesson) {
      return update(slug, function (t) {
        t.lesson = Number(lesson) || 1;
      });
    },

    recordCard: function (slug, index) {
      return update(slug, function (t) {
        var card = t.cards[String(index)] || (t.cards[String(index)] = { seen: 0, box: 1, due: 0 });
        card.seen++;
      });
    },

    // Leitner boxes: recalling a card promotes it to a longer interval, missing
    // it drops back to box 1 so it comes round again in the same session.
    gradeCard: function (slug, index, remembered) {
      return update(slug, function (t) {
        var key = String(index);
        var card = t.cards[key] || (t.cards[key] = { seen: 0, box: 1, due: 0 });
        card.box = remembered ? Math.min(card.box + 1, BOX_DAYS.length) : 1;
        card.due = Date.now() + BOX_DAYS[card.box - 1] * DAY;
        card.graded = Date.now();
      });
    },

    // Cards whose interval has elapsed, plus any never reviewed.
    dueCards: function (slug, total) {
      var cards = (load().topics[slug] || blankTopic()).cards;
      var now = Date.now();
      var due = [];
      for (var i = 0; i < total; i++) {
        var card = cards[String(i)];
        if (!card || !card.due || card.due <= now) due.push(i);
      }
      return due;
    },

    cardState: function (slug, index) {
      var cards = (load().topics[slug] || blankTopic()).cards;
      return cards[String(index)] || { seen: 0, box: 1, due: 0 };
    },

    boxDays: function (box) {
      return BOX_DAYS[Math.min(Math.max(box, 1), BOX_DAYS.length) - 1];
    },

    recordGame: function (slug, score, max) {
      return update(slug, function (t) {
        t.game.plays++;
        t.game.max = max || t.game.max;
        if (score > t.game.best) t.game.best = score;
      });
    },

    markComplete: function (slug) {
      return update(slug, function (t) {
        if (!t.completed) t.completed = Date.now();
      });
    },

    // Answered/correct counts across every quiz recorded for a topic.
    score: function (slug) {
      return scored(load().topics[slug] || blankTopic());
    },

    exportJSON: function () {
      return JSON.stringify(load(), null, 2);
    },

    importJSON: function (text) {
      var incoming = normalise(JSON.parse(text));
      var data = load();
      Object.keys(incoming.topics).forEach(function (slug) {
        var mine = data.topics[slug];
        var theirs = incoming.topics[slug];
        if (!mine) {
          data.topics[slug] = theirs;
          return;
        }
        // Merge conservatively: keep whichever side knows more.
        mine.visited = mine.visited || theirs.visited;
        mine.completed = mine.completed || theirs.completed;
        mine.lesson = Math.max(mine.lesson, theirs.lesson);
        Object.keys(theirs.quiz).forEach(function (qid) {
          if (!mine.quiz[qid]) mine.quiz[qid] = theirs.quiz[qid];
        });
        Object.keys(theirs.cards).forEach(function (i) {
          mine.cards[i] = Math.max(mine.cards[i] || 0, theirs.cards[i]);
        });
        mine.game.plays += theirs.game.plays;
        mine.game.best = Math.max(mine.game.best, theirs.game.best);
        mine.game.max = mine.game.max || theirs.game.max;
      });
      return save(data);
    },

    reset: function () {
      try {
        localStorage.removeItem(KEY);
        localStorage.removeItem(LEGACY_KEY);
      } catch (e) {}
    }
  };

  // --- a11y: Font Awesome icons are decorative; hide from screen readers ---
  var icons = document.querySelectorAll('i.fas, i.far, i.fab, i.fa');
  for (var i = 0; i < icons.length; i++) {
    if (!icons[i].hasAttribute('aria-hidden')) {
      icons[i].setAttribute('aria-hidden', 'true');
    }
  }

  var slug = currentSlug();
  if (slug) window.pmProgress.markVisited(slug);
})();
