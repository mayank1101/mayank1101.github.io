// PM Gym — shared interactive engine for /pm-gym/ pages.
// Each page ships its deck as JSON in <script type="application/json" id="pm-data">,
// rendered from _data/pm-gym/<topic>.yml.

(function () {
  'use strict';

  // Deck source: the inline JSON block, with the legacy window.pmData object
  // still honoured so a page that hasn't been migrated keeps working.
  var deck = (function () {
    var el = document.getElementById('pm-data');
    if (el) {
      try {
        return JSON.parse(el.textContent);
      } catch (e) {
        /* fall through to window.pmData */
      }
    }
    return window.pmData || {};
  })();

  var slugMatch = location.pathname.match(/\/pm-gym\/([^\/]+)\.html?$/);
  var SLUG = slugMatch ? slugMatch[1] : null;

  // Progress storage lives in pm-gym-progress.js, which the layout loads after
  // this file. Every call below happens inside an event handler, by which time
  // the API exists — but guard anyway so a load failure can't break the page.
  function progress() {
    return window.pmProgress || null;
  }

  // Analytics: without per-question data there's no way to tell which questions
  // are too easy, too hard, or badly worded. No-ops when gtag isn't loaded.
  function track(name, params) {
    if (typeof window.gtag !== 'function') return;
    params = params || {};
    params.topic = SLUG;
    window.gtag('event', name, params);
  }

  // ---------- Tabs ----------
  function switchTab(tabId, focusTab) {
    document.querySelectorAll('.pm-pane').forEach(function (p) {
      var on = p.dataset.pane === tabId;
      p.classList.toggle('active', on);
      p.hidden = !on;
    });
    document.querySelectorAll('.pm-tab-btn').forEach(function (b) {
      var on = b.dataset.tab === tabId;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
      b.tabIndex = on ? 0 : -1;
      if (on && focusTab) b.focus();
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // Wires the tab strip up as a real ARIA tablist: without this the buttons are
  // announced as plain buttons and the panes as unrelated content.
  function initTabs() {
    var list = document.querySelector('.pm-tabs');
    if (!list) return;
    list.setAttribute('role', 'tablist');

    var buttons = Array.prototype.slice.call(list.querySelectorAll('.pm-tab-btn'));
    buttons.forEach(function (btn) {
      var name = btn.dataset.tab;
      var pane = document.querySelector('.pm-pane[data-pane="' + name + '"]');
      btn.setAttribute('role', 'tab');
      btn.id = 'pm-tab-' + name;
      btn.setAttribute('aria-selected', btn.classList.contains('active') ? 'true' : 'false');
      btn.tabIndex = btn.classList.contains('active') ? 0 : -1;
      if (!pane) return;
      pane.id = 'pm-pane-' + name;
      pane.setAttribute('role', 'tabpanel');
      pane.setAttribute('aria-labelledby', btn.id);
      pane.tabIndex = 0;
      pane.hidden = !pane.classList.contains('active');
      btn.setAttribute('aria-controls', pane.id);
    });

    // Left/right (and Home/End) move between tabs, as the tab pattern expects.
    list.addEventListener('keydown', function (e) {
      var i = buttons.indexOf(document.activeElement);
      if (i === -1) return;
      var next = null;
      if (e.key === 'ArrowRight') next = (i + 1) % buttons.length;
      else if (e.key === 'ArrowLeft') next = (i - 1 + buttons.length) % buttons.length;
      else if (e.key === 'Home') next = 0;
      else if (e.key === 'End') next = buttons.length - 1;
      if (next === null) return;
      e.preventDefault();
      switchTab(buttons[next].dataset.tab, true);
    });
  }

  // Quiz feedback is injected after a click; without a live region a screen
  // reader never hears whether the answer was right.
  function initLiveRegions() {
    var live = document.querySelectorAll(
      '.pm-quiz-feedback, #pm-game-reveal, #pm-game-clues, #pm-game-end'
    );
    live.forEach(function (el) {
      el.setAttribute('role', 'status');
      el.setAttribute('aria-live', 'polite');
    });
  }

  // ---------- Lesson steps ----------
  function showLesson(step) {
    document.querySelectorAll('.pm-lesson').forEach(function (l) {
      l.classList.toggle('active', l.dataset.lesson === String(step));
    });
    document.querySelectorAll('.pm-lesson-btn').forEach(function (b) {
      var on = b.dataset.lesson === String(step);
      b.classList.toggle('active', on);
      if (on) b.setAttribute('aria-current', 'step');
      else b.removeAttribute('aria-current');
    });
    var p = progress();
    if (p && SLUG) p.recordLesson(SLUG, step);
  }

  // ---------- Quiz checkpoints ----------
  // Markup: .pm-quiz > .pm-quiz-options > button.pm-quiz-option
  //   data-result="correct" on the right answer,
  //   data-msg="feedback text" per option,
  //   sibling .pm-quiz-feedback receives the message.

  // Options are authored in a fixed order, which let the correct answer sit in
  // the same slot every time. Shuffling on load makes position meaningless.
  function shuffleQuizOptions() {
    document.querySelectorAll('.pm-quiz-options').forEach(function (box) {
      var opts = Array.prototype.slice.call(box.querySelectorAll('.pm-quiz-option'));
      if (opts.length < 2) return;
      shuffle(opts).forEach(function (btn) {
        box.appendChild(btn);
      });
    });
  }
  // Each .pm-quiz gets a stable id from its document position. Option order is
  // shuffled, but the quizzes themselves never move, so the id survives reloads.
  function indexQuizzes() {
    document.querySelectorAll('.pm-quiz').forEach(function (quiz, i) {
      quiz.dataset.qid = 'q' + i;
    });
  }

  function handleQuizClick(btn) {
    if (btn.disabled) return;
    var quiz = btn.closest('.pm-quiz');
    var feedback = quiz.querySelector('.pm-quiz-feedback');
    if (!feedback) return;
    var correct = btn.dataset.result === 'correct';

    var p = progress();
    if (p && SLUG && quiz.dataset.qid) p.recordAnswer(SLUG, quiz.dataset.qid, correct);

    var scenario = btn.closest('.pm-scenario-card');
    track('pm_quiz_answer', {
      question_id: quiz.dataset.qid,
      correct: correct,
      difficulty: scenario ? scenario.dataset.difficulty : 'lesson',
      pane: btn.closest('.pm-pane') ? btn.closest('.pm-pane').dataset.pane : ''
    });

    if (correct) {
      // Lock the whole question once it's answered right.
      quiz.querySelectorAll('.pm-quiz-option').forEach(function (o) {
        o.disabled = true;
        o.classList.toggle('pm-option-correct', o === btn);
      });
      quiz.classList.add('pm-quiz-answered');
    } else {
      // Retire only the wrong pick, so the learner can try the rest.
      btn.disabled = true;
      btn.classList.add('pm-option-wrong');
    }

    feedback.classList.remove('pm-correct', 'pm-wrong');
    feedback.classList.add(correct ? 'pm-correct' : 'pm-wrong');
    feedback.innerHTML =
      '<strong><i class="fas ' +
      (correct ? 'fa-circle-check' : 'fa-circle-xmark') +
      '"></i> ' +
      (correct ? 'Correct!' : 'Not quite — try again.') +
      '</strong>' +
      (btn.dataset.msg || '');

    renderPracticeCounter();
    checkCompletion();
  }

  // ---------- Practice progress ----------
  function practiceQuizzes() {
    return document.querySelectorAll('[data-pane="practice"] .pm-quiz');
  }

  function practiceStats() {
    var p = progress();
    var topic = p && SLUG ? p.topic(SLUG) : { quiz: {} };
    var total = 0;
    var answered = 0;
    var correct = 0;
    practiceQuizzes().forEach(function (quiz) {
      total++;
      var entry = topic.quiz[quiz.dataset.qid];
      if (!entry) return;
      answered++;
      if (entry.correct) correct++;
    });
    return { total: total, answered: answered, correct: correct };
  }

  function renderPracticeCounter() {
    var box = document.getElementById('pm-practice-progress');
    if (!box) return;
    var s = practiceStats();
    var pct = s.total ? Math.round((s.answered / s.total) * 100) : 0;
    box.querySelector('.pm-practice-fill').style.width = pct + '%';
    box.querySelector('.pm-practice-label').textContent =
      s.answered + ' of ' + s.total + ' answered · ' +
      s.correct + ' right first try';
  }

  function mountPracticeCounter() {
    var intro = document.querySelector('[data-pane="practice"] .pm-section-intro');
    if (!intro || !practiceQuizzes().length) return;
    var box = document.createElement('div');
    box.className = 'pm-practice-progress';
    box.id = 'pm-practice-progress';
    box.innerHTML =
      '<span class="pm-practice-track"><span class="pm-practice-fill"></span></span>' +
      '<span class="pm-practice-label"></span>';
    intro.appendChild(box);
    renderPracticeCounter();
  }

  // ---------- Practice filters ----------
  function scenarioCards() {
    return Array.prototype.slice.call(document.querySelectorAll('.pm-scenario-card'));
  }

  function filterPractice(level) {
    document.querySelectorAll('.pm-filter').forEach(function (b) {
      b.classList.toggle('active', b.dataset.difficulty === level);
      b.setAttribute('aria-pressed', b.dataset.difficulty === level ? 'true' : 'false');
    });
    scenarioCards().forEach(function (card) {
      card.classList.toggle('pm-hidden', level !== 'all' && card.dataset.difficulty !== level);
    });
  }

  // Show a random handful instead of the full list — a shorter set people
  // actually finish, and a different one each time.
  function samplePractice(size) {
    var cards = scenarioCards();
    var keep = {};
    shuffle(cards.map(function (_, i) { return i; }))
      .slice(0, size)
      .forEach(function (i) { keep[i] = true; });
    cards.forEach(function (card, i) {
      card.classList.toggle('pm-hidden', !keep[i]);
    });
    document.querySelectorAll('.pm-filter').forEach(function (b) {
      b.classList.toggle('active', false);
      b.setAttribute('aria-pressed', 'false');
    });
  }

  // A topic counts as done once every practice scenario has been attempted —
  // not when the learner merely clicked through to another tab.
  function checkCompletion() {
    var p = progress();
    if (!p || !SLUG) return;
    var s = practiceStats();
    if (s.total && s.answered >= s.total) {
      var already = p.topic(SLUG).completed;
      p.markComplete(SLUG);
      if (!already) track('pm_topic_complete', { correct: s.correct, total: s.total });
    }
  }

  // Re-apply stored answers so a reload doesn't hand back free retries.
  function restoreAnswers() {
    var p = progress();
    if (!p || !SLUG) return;
    var topic = p.topic(SLUG);
    document.querySelectorAll('.pm-quiz').forEach(function (quiz) {
      var entry = topic.quiz[quiz.dataset.qid];
      if (!entry || !entry.correct) return;
      quiz.classList.add('pm-quiz-answered');
      quiz.querySelectorAll('.pm-quiz-option').forEach(function (o) {
        o.disabled = true;
        o.classList.toggle('pm-option-correct', o.dataset.result === 'correct');
      });
    });
  }

  // ---------- Flashcards ----------
  // `order` is the study queue: cards that are due (or never seen) come first,
  // so a returning learner meets the ones they were shakiest on.
  var cardIndex = 0;
  var order = [];

  function buildDeckOrder() {
    var cards = deck.flashcards || [];
    var p = progress();
    order = cards.map(function (_, i) { return i; });
    if (!p || !SLUG) return;
    var due = p.dueCards(SLUG, cards.length);
    var dueSet = {};
    due.forEach(function (i) { dueSet[i] = true; });
    order.sort(function (a, b) {
      return (dueSet[b] ? 1 : 0) - (dueSet[a] ? 1 : 0);
    });
  }

  function currentCard() {
    return order.length ? order[cardIndex] : cardIndex;
  }

  function describeDue(state) {
    var p = progress();
    if (!p || !state.graded) return 'New card';
    if (state.due <= Date.now()) return 'Due now · box ' + state.box;
    var days = Math.ceil((state.due - Date.now()) / (24 * 60 * 60 * 1000));
    return 'Box ' + state.box + ' · next in ' + days + (days === 1 ? ' day' : ' days');
  }

  function renderCard() {
    var cards = deck.flashcards || [];
    if (!cards.length) return;
    var card = cards[currentCard()];
    var inner = document.getElementById('pm-flashcard-inner');
    if (!inner) return;
    inner.classList.remove('flipped');
    var grade = document.getElementById('pm-card-grade');
    if (grade) grade.classList.add('pm-hidden');
    var p = progress();
    setTimeout(function () {
      setText('pm-card-category', card.category);
      setText('pm-card-progress', 'Card ' + (cardIndex + 1) + ' of ' + cards.length);
      setText('pm-card-title', card.title);
      setText('pm-card-back-text', card.desc);
      setText('pm-card-application', card.application);
      setText('pm-deck-tracker', (cardIndex + 1) + ' / ' + cards.length);
      if (p && SLUG) setText('pm-card-due', describeDue(p.cardState(SLUG, currentCard())));
    }, 150);
  }

  function gradeCard(remembered) {
    var p = progress();
    if (p && SLUG) p.gradeCard(SLUG, currentCard(), remembered);
    nextCard();
  }

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) el.innerText = value;
  }

  function flipCard() {
    var inner = document.getElementById('pm-flashcard-inner');
    if (!inner) return;
    inner.classList.toggle('flipped');
    var flipped = inner.classList.contains('flipped');
    // Count a card as reviewed only when its answer side is turned up, and
    // only then offer the recall buttons.
    var p = progress();
    if (flipped && p && SLUG) p.recordCard(SLUG, currentCard());
    var grade = document.getElementById('pm-card-grade');
    if (grade) grade.classList.toggle('pm-hidden', !flipped);
  }

  function nextCard() {
    var cards = deck.flashcards || [];
    if (!cards.length) return;
    cardIndex = (cardIndex + 1) % cards.length;
    renderCard();
  }

  function prevCard() {
    var cards = deck.flashcards || [];
    if (!cards.length) return;
    cardIndex = (cardIndex - 1 + cards.length) % cards.length;
    renderCard();
  }

  // ---------- Vocab guessing game ----------
  // Data: deck.vocab = [{ term, define, clues: [c1, c2, c3] }]
  // Read the clues, guess the term. Fewer clues used = more points.
  var game = { order: [], idx: 0, score: 0, streak: 0, cluesShown: 1, answered: false };

  function shuffle(arr) {
    var a = arr.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
    return a;
  }

  function vocabDeck() {
    return deck.vocab || [];
  }

  function currentTerm() {
    return vocabDeck()[game.order[game.idx]];
  }

  function startGame() {
    var deck = vocabDeck();
    if (!deck.length) return;
    game.order = shuffle(deck.map(function (_, i) { return i; }));
    game.idx = 0;
    game.score = 0;
    game.streak = 0;
    toggle('pm-game-end', false);
    toggle('pm-game-card', true);
    renderRound();
  }

  function renderRound() {
    game.cluesShown = 1;
    game.answered = false;
    var term = currentTerm();
    if (!term) return;

    setText('pm-game-progress', 'Round ' + (game.idx + 1) + ' / ' + game.order.length);
    setText('pm-game-score', 'Score ' + game.score);
    renderClues(term);

    // Options: correct term + up to 3 distractors from the deck
    var others = vocabDeck()
      .map(function (v) { return v.term; })
      .filter(function (t) { return t !== term.term; });
    var opts = shuffle(others).slice(0, 3);
    opts.push(term.term);
    renderOptions(shuffle(opts));

    var reveal = document.getElementById('pm-game-reveal');
    if (reveal) {
      reveal.className = 'pm-game-reveal';
      reveal.innerHTML = '';
    }
    var next = document.getElementById('pm-game-next');
    if (next) {
      next.classList.add('pm-hidden');
      next.innerHTML =
        game.idx + 1 >= game.order.length
          ? 'See results <i class="fas fa-flag-checkered"></i>'
          : 'Next term <i class="fas fa-chevron-right"></i>';
    }
  }

  function renderClues(term) {
    var box = document.getElementById('pm-game-clues');
    if (!box) return;
    var worth = 4 - game.cluesShown; // clue 1 → 3, clue 2 → 2, clue 3 → 1
    var html = '<div class="pm-game-worth">Worth ' + worth + ' point' + (worth === 1 ? '' : 's') + '</div>';
    for (var i = 0; i < game.cluesShown && i < term.clues.length; i++) {
      html +=
        '<p class="pm-game-clue"><span>Clue ' + (i + 1) + '</span>' + term.clues[i] + '</p>';
    }
    box.innerHTML = html;
  }

  function renderOptions(terms) {
    var box = document.getElementById('pm-game-options');
    if (!box) return;
    box.innerHTML = '';
    terms.forEach(function (t) {
      var btn = document.createElement('button');
      btn.className = 'pm-game-option';
      btn.dataset.term = t;
      btn.innerText = t;
      box.appendChild(btn);
    });
  }

  function pickOption(el) {
    if (game.answered) return;
    var term = currentTerm();
    var chosen = el.dataset.term;

    if (chosen === term.term) {
      var points = 4 - game.cluesShown;
      game.score += points;
      game.streak += 1;
      el.classList.add('pm-game-correct');
      finishRound(term, true, points);
    } else {
      el.classList.add('pm-game-wrong');
      el.disabled = true;
      if (game.cluesShown < term.clues.length) {
        game.cluesShown += 1;
        renderClues(term);
      } else {
        game.streak = 0;
        markCorrectOption(term.term);
        finishRound(term, false, 0);
      }
    }
  }

  function markCorrectOption(termText) {
    document.querySelectorAll('#pm-game-options .pm-game-option').forEach(function (b) {
      if (b.dataset.term === termText) b.classList.add('pm-game-correct');
    });
  }

  function finishRound(term, won, points) {
    game.answered = true;
    document.querySelectorAll('#pm-game-options .pm-game-option').forEach(function (b) {
      b.disabled = true;
    });
    var reveal = document.getElementById('pm-game-reveal');
    if (reveal) {
      reveal.className = 'pm-game-reveal ' + (won ? 'pm-correct' : 'pm-wrong');
      reveal.innerHTML =
        '<strong><i class="fas ' +
        (won ? 'fa-circle-check' : 'fa-circle-xmark') +
        '"></i> ' +
        (won ? '+' + points + ' — ' + term.term : 'The answer was ' + term.term) +
        '</strong>' +
        term.define;
    }
    setText('pm-game-score', 'Score ' + game.score);
    var next = document.getElementById('pm-game-next');
    if (next) next.classList.remove('pm-hidden');
  }

  function nextRound() {
    game.idx += 1;
    if (game.idx >= game.order.length) return endGame();
    renderRound();
  }

  function endGame() {
    toggle('pm-game-card', false);
    var max = game.order.length * 3;
    var pct = max ? Math.round((game.score / max) * 100) : 0;
    var msg =
      pct >= 80 ? 'Sharp — these terms are locked in.'
        : pct >= 50 ? 'Solid. A second run will cement the rest.'
          : 'Good start — replay to lock them in.';
    var end = document.getElementById('pm-game-end');
    if (end) {
      end.innerHTML =
        '<div class="pm-game-final">' +
        '<i class="fas fa-trophy"></i>' +
        '<p class="pm-game-final-score">' + game.score + ' / ' + max + '</p>' +
        '<p class="muted">' + msg + '</p>' +
        '<button class="cta" data-pm-action="game-replay"><i class="fas fa-rotate-right"></i> Play again</button>' +
        '</div>';
      toggle('pm-game-end', true);
    }
    var p = progress();
    if (p && SLUG) p.recordGame(SLUG, game.score, max);
    track('pm_game_complete', { score: game.score, max: max });
    document.dispatchEvent(new CustomEvent('pmgym:gamecomplete'));
  }

  function toggle(id, on) {
    var el = document.getElementById(id);
    if (el) el.classList.toggle('pm-hidden', !on);
  }

  // The card is a <div> with a click handler, so keyboard users couldn't reach
  // or flip it. Give it button semantics plus arrow-key deck navigation.
  function initFlashcard() {
    var card = document.querySelector('.pm-flashcard');
    if (!card) return;
    card.setAttribute('role', 'button');
    card.tabIndex = 0;
    card.setAttribute('aria-label', 'Flashcard. Press Enter to flip, arrow keys to change card.');

    var back = document.getElementById('pm-card-back-text');
    if (back) {
      back.setAttribute('role', 'status');
      back.setAttribute('aria-live', 'polite');
    }

    card.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') {
        e.preventDefault();
        flipCard();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        nextCard();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        prevCard();
      }
    });
  }

  // ---------- Modal ----------
  var lastFocused = null;

  function initModal() {
    var modal = document.getElementById('pm-modal');
    if (!modal) return;
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-labelledby', 'pm-modal-title');
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && modal.classList.contains('visible')) closeModal();
    });
  }

  function openModal(title, msg, iconClass) {
    setText('pm-modal-title', title);
    setText('pm-modal-msg', msg);
    var icon = document.getElementById('pm-modal-icon');
    if (icon) icon.className = 'fas ' + (iconClass || 'fa-info');
    var modal = document.getElementById('pm-modal');
    if (!modal) return;
    lastFocused = document.activeElement;
    modal.classList.add('visible');
    var close = modal.querySelector('[data-pm-action="close-modal"]');
    if (close) close.focus();
  }

  function closeModal() {
    var modal = document.getElementById('pm-modal');
    if (modal) modal.classList.remove('visible');
    if (lastFocused && lastFocused.focus) lastFocused.focus();
    lastFocused = null;
  }

  // ---------- Wiring ----------
  document.addEventListener('click', function (e) {
    var tab = e.target.closest('.pm-tab-btn');
    if (tab) return switchTab(tab.dataset.tab);

    var lessonBtn = e.target.closest('[data-goto-lesson]');
    if (lessonBtn) return showLesson(lessonBtn.dataset.gotoLesson);

    var lessonNav = e.target.closest('.pm-lesson-btn');
    if (lessonNav) return showLesson(lessonNav.dataset.lesson);

    var gotoTab = e.target.closest('[data-goto-tab]');
    if (gotoTab) return switchTab(gotoTab.dataset.gotoTab);

    var filter = e.target.closest('.pm-filter');
    if (filter) return filterPractice(filter.dataset.difficulty);

    var quizOption = e.target.closest('.pm-quiz-option');
    if (quizOption) return handleQuizClick(quizOption);

    var gameOption = e.target.closest('.pm-game-option');
    if (gameOption) return pickOption(gameOption);

    var action = e.target.closest('[data-pm-action]');
    if (action) {
      var actions = {
        'flip-card': flipCard,
        'next-card': nextCard,
        'prev-card': prevCard,
        'card-good': function () { gradeCard(true); },
        'card-again': function () { gradeCard(false); },
        'practice-sample': function () { samplePractice(8); },
        'game-next': nextRound,
        'game-replay': startGame,
        'close-modal': closeModal
      };
      var fn = actions[action.dataset.pmAction];
      if (fn) return fn();
    }

    if (e.target.id === 'pm-modal') closeModal();
  });

  document.addEventListener('DOMContentLoaded', function () {
    initTabs();
    initLiveRegions();
    initFlashcard();
    initModal();
    indexQuizzes();
    shuffleQuizOptions();
    restoreAnswers();
    mountPracticeCounter();
    checkCompletion();
    buildDeckOrder();
    renderCard();
    startGame();

    // Drop the learner back on the lesson they left off at.
    var p = progress();
    if (p && SLUG) {
      var last = p.topic(SLUG).lesson;
      if (last > 1 && document.querySelector('.pm-lesson[data-lesson="' + last + '"]')) {
        showLesson(last);
      }
    }
  });

  // Expose for page-specific scripts
  window.pmGuide = {
    switchTab: switchTab,
    showLesson: showLesson,
    openModal: openModal,
    closeModal: closeModal
  };
})();
