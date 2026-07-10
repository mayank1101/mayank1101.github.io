// PM Gym — progress tracking (localStorage) + shared a11y touch-ups.
// Loaded on EVERY PM Gym page. On a topic page it records visited/completed;
// the hub reads this data via window.pmProgress to render badges and bars.
(function () {
  'use strict';

  var KEY = 'pmgym:progress:v1';

  function load() {
    try {
      var raw = localStorage.getItem(KEY);
      var obj = raw ? JSON.parse(raw) : {};
      obj.visited = obj.visited || {};
      obj.completed = obj.completed || {};
      return obj;
    } catch (e) {
      return { visited: {}, completed: {} };
    }
  }

  function save(obj) {
    try {
      localStorage.setItem(KEY, JSON.stringify(obj));
    } catch (e) {
      /* storage unavailable (private mode / disabled) — degrade silently */
    }
  }

  function currentSlug() {
    var m = location.pathname.match(/\/pm-gym\/([^\/]+)\.html?$/);
    return m ? m[1] : null;
  }

  // Expose a tiny API for the hub script.
  window.pmProgress = {
    key: KEY,
    load: load,
    reset: function () {
      try { localStorage.removeItem(KEY); } catch (e) {}
    }
  };

  // --- a11y: Font Awesome icons are decorative; hide from screen readers ---
  var icons = document.querySelectorAll('i.fas, i.far, i.fab, i.fa');
  for (var i = 0; i < icons.length; i++) {
    if (!icons[i].hasAttribute('aria-hidden')) {
      icons[i].setAttribute('aria-hidden', 'true');
    }
  }

  // --- Record progress only on individual topic pages ---
  var slug = currentSlug();
  if (!slug) return;

  var data = load();
  if (!data.visited[slug]) {
    data.visited[slug] = Date.now();
    save(data);
  }

  function markComplete() {
    var d = load();
    if (!d.completed[slug]) {
      d.completed[slug] = Date.now();
      save(d);
    }
  }

  // Completion signal 1: user reached the final lesson and clicked through to
  // Flashcards (only the last lesson's button uses data-goto-tab="flashcards").
  document.addEventListener('click', function (e) {
    var t = e.target.closest && e.target.closest('[data-goto-tab="flashcards"]');
    if (t) markComplete();
  });

  // Completion signal 2: user finished the vocab game.
  document.addEventListener('pmgym:gamecomplete', markComplete);
})();
