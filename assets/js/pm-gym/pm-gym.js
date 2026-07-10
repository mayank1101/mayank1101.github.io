// PM Gym — shared interactive engine for /pm-gym/ pages.
// Each page provides its data via window.pmData = { flashcards, vocab }.

(function () {
  'use strict';

  // ---------- Tabs ----------
  function switchTab(tabId) {
    document.querySelectorAll('.pm-pane').forEach(function (p) {
      p.classList.toggle('active', p.dataset.pane === tabId);
    });
    document.querySelectorAll('.pm-tab-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.tab === tabId);
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ---------- Lesson steps ----------
  function showLesson(step) {
    document.querySelectorAll('.pm-lesson').forEach(function (l) {
      l.classList.toggle('active', l.dataset.lesson === String(step));
    });
    document.querySelectorAll('.pm-lesson-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.lesson === String(step));
    });
  }

  // ---------- Quiz checkpoints ----------
  // Markup: .pm-quiz > .pm-quiz-options > button.pm-quiz-option
  //   data-result="correct" on the right answer,
  //   data-msg="feedback text" per option,
  //   sibling .pm-quiz-feedback receives the message.
  function handleQuizClick(btn) {
    var quiz = btn.closest('.pm-quiz');
    var feedback = quiz.querySelector('.pm-quiz-feedback');
    if (!feedback) return;
    var correct = btn.dataset.result === 'correct';
    feedback.classList.remove('pm-correct', 'pm-wrong');
    feedback.classList.add(correct ? 'pm-correct' : 'pm-wrong');
    feedback.innerHTML =
      '<strong><i class="fas ' +
      (correct ? 'fa-circle-check' : 'fa-circle-xmark') +
      '"></i> ' +
      (correct ? 'Correct!' : 'Not quite — try again.') +
      '</strong>' +
      (btn.dataset.msg || '');
  }

  // ---------- Flashcards ----------
  var cardIndex = 0;

  function renderCard() {
    var cards = (window.pmData || {}).flashcards || [];
    if (!cards.length) return;
    var card = cards[cardIndex];
    var inner = document.getElementById('pm-flashcard-inner');
    if (!inner) return;
    inner.classList.remove('flipped');
    setTimeout(function () {
      setText('pm-card-category', card.category);
      setText('pm-card-progress', 'Card ' + (cardIndex + 1) + ' of ' + cards.length);
      setText('pm-card-title', card.title);
      setText('pm-card-back-text', card.desc);
      setText('pm-card-application', card.application);
      setText('pm-deck-tracker', (cardIndex + 1) + ' / ' + cards.length);
    }, 150);
  }

  function setText(id, value) {
    var el = document.getElementById(id);
    if (el) el.innerText = value;
  }

  function flipCard() {
    var inner = document.getElementById('pm-flashcard-inner');
    if (inner) inner.classList.toggle('flipped');
  }

  function nextCard() {
    var cards = (window.pmData || {}).flashcards || [];
    cardIndex = (cardIndex + 1) % cards.length;
    renderCard();
  }

  function prevCard() {
    var cards = (window.pmData || {}).flashcards || [];
    cardIndex = (cardIndex - 1 + cards.length) % cards.length;
    renderCard();
  }

  // ---------- Vocab guessing game ----------
  // Data: window.pmData.vocab = [{ term, define, clues: [c1, c2, c3] }]
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
    return (window.pmData || {}).vocab || [];
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
    // Signal to the progress tracker that this topic's vocab game was finished.
    document.dispatchEvent(new CustomEvent('pmgym:gamecomplete'));
  }

  function toggle(id, on) {
    var el = document.getElementById(id);
    if (el) el.classList.toggle('pm-hidden', !on);
  }

  // ---------- Modal ----------
  function openModal(title, msg, iconClass) {
    setText('pm-modal-title', title);
    setText('pm-modal-msg', msg);
    var icon = document.getElementById('pm-modal-icon');
    if (icon) icon.className = 'fas ' + (iconClass || 'fa-info');
    var modal = document.getElementById('pm-modal');
    if (modal) modal.classList.add('visible');
  }

  function closeModal() {
    var modal = document.getElementById('pm-modal');
    if (modal) modal.classList.remove('visible');
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
    renderCard();
    startGame();
  });

  // Expose for page-specific scripts
  window.pmGuide = {
    switchTab: switchTab,
    showLesson: showLesson,
    openModal: openModal,
    closeModal: closeModal
  };
})();
