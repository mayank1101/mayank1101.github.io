// PM Gym — hub-only enhancements: progress badges + per-category bars,
// jump navigation, live search/filter, and uniform per-card practice meta.
// Self-contained (reads localStorage directly), so load order doesn't matter.
(function () {
  'use strict';

  var categories = document.querySelectorAll('.pm-category');
  if (!categories.length) return; // not the hub

  var STORAGE_KEY = 'pmgym:progress:v1';

  function loadProgress() {
    try {
      var o = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
      o.visited = o.visited || {};
      o.completed = o.completed || {};
      return o;
    } catch (e) {
      return { visited: {}, completed: {} };
    }
  }

  function slugOf(card) {
    var href = card.getAttribute('href') || '';
    var m = href.match(/\/pm-gym\/([^\/]+)\.html?/);
    return m ? m[1] : null;
  }

  function icon(name) {
    var i = document.createElement('i');
    i.className = 'fas ' + name;
    i.setAttribute('aria-hidden', 'true');
    return i;
  }

  var progress = loadProgress();
  var allCards = Array.prototype.slice.call(document.querySelectorAll('.pm-category .card'));
  var totalCount = allCards.length;
  var totalDone = 0;

  // ---------- Per-card state + uniform practice meta ----------
  allCards.forEach(function (card) {
    var slug = slugOf(card);
    var state = 'new';
    if (slug && progress.completed[slug]) state = 'done';
    else if (slug && progress.visited[slug]) state = 'progress';

    if (state === 'done') {
      totalDone++;
      card.classList.add('pm-card-done');
    } else if (state === 'progress') {
      card.classList.add('pm-card-progress');
    }

    // Status badge (top-left) — only for started/completed, to avoid noise.
    if (state !== 'new') {
      var badge = document.createElement('span');
      badge.className = 'pm-card-status pm-card-status-' + state;
      if (state === 'done') {
        badge.appendChild(icon('fa-circle-check'));
        badge.appendChild(document.createTextNode(' Completed'));
      } else {
        badge.appendChild(icon('fa-circle-half-stroke'));
        badge.appendChild(document.createTextNode(' In progress'));
      }
      card.insertBefore(badge, card.firstChild);
    }

    // Uniform practice meta — true for every topic page.
    var tags = card.querySelector('.skill-tags');
    if (tags && !card.querySelector('.pm-card-meta')) {
      var meta = document.createElement('p');
      meta.className = 'pm-card-meta';
      meta.appendChild(icon('fa-dumbbell'));
      meta.appendChild(document.createTextNode(' Lessons · flashcards · 15 practice scenarios · vocab game'));
      tags.parentNode.insertBefore(meta, tags.nextSibling);
    }

    // Searchable text blob.
    var hay = (card.textContent || '').toLowerCase();
    card.setAttribute('data-search', hay);
  });

  // ---------- Per-category progress bars + ids ----------
  categories.forEach(function (cat, idx) {
    cat.id = 'cat-' + (idx + 1);
    var cards = cat.querySelectorAll('.card');
    var done = 0;
    cards.forEach(function (c) {
      var s = slugOf(c);
      if (s && progress.completed[s]) done++;
    });
    var head = cat.querySelector('.pm-category-head');
    if (!head) return;
    var wrap = document.createElement('div');
    wrap.className = 'pm-cat-progress';
    var pct = cards.length ? Math.round((done / cards.length) * 100) : 0;
    var label = document.createElement('span');
    label.className = 'pm-cat-progress-label';
    label.textContent = done + ' / ' + cards.length + ' completed';
    var track = document.createElement('span');
    track.className = 'pm-cat-progress-track';
    var fill = document.createElement('span');
    fill.className = 'pm-cat-progress-fill';
    fill.style.width = pct + '%';
    track.appendChild(fill);
    wrap.appendChild(track);
    wrap.appendChild(label);
    head.appendChild(wrap);
  });

  // ---------- Jump nav + search + overall summary ----------
  var container = categories[0].parentNode;

  var bar = document.createElement('div');
  bar.className = 'pm-hub-bar';

  // Overall summary + reset
  var summary = document.createElement('div');
  summary.className = 'pm-hub-summary';
  var sumText = document.createElement('span');
  sumText.className = 'pm-hub-summary-text';
  function summaryLabel() {
    return totalDone + ' of ' + totalCount + ' topics completed';
  }
  sumText.textContent = summaryLabel();
  summary.appendChild(icon('fa-trophy'));
  summary.appendChild(sumText);

  var resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.className = 'pm-hub-reset';
  resetBtn.textContent = 'Reset progress';
  resetBtn.addEventListener('click', function () {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    location.reload();
  });
  // Only show reset if there's anything to reset.
  if (totalDone > 0 || Object.keys(progress.visited).length > 0) {
    summary.appendChild(resetBtn);
  }

  // Jump links + search
  var tools = document.createElement('div');
  tools.className = 'pm-hub-tools';

  var nav = document.createElement('nav');
  nav.className = 'pm-hub-jump';
  nav.setAttribute('aria-label', 'Jump to category');
  categories.forEach(function (cat, idx) {
    var kicker = cat.querySelector('.pm-category-kicker');
    var a = document.createElement('a');
    a.href = '#cat-' + (idx + 1);
    a.textContent = kicker ? kicker.textContent : 'Category ' + (idx + 1);
    nav.appendChild(a);
  });

  var searchWrap = document.createElement('div');
  searchWrap.className = 'pm-hub-search';
  var sIcon = icon('fa-magnifying-glass');
  var input = document.createElement('input');
  input.type = 'search';
  input.placeholder = 'Search topics…';
  input.className = 'pm-hub-search-input';
  input.setAttribute('aria-label', 'Search topics');
  searchWrap.appendChild(sIcon);
  searchWrap.appendChild(input);

  tools.appendChild(nav);
  tools.appendChild(searchWrap);

  bar.appendChild(summary);
  bar.appendChild(tools);
  container.insertBefore(bar, categories[0]);

  // Sticky offset below the gym header.
  var header = document.querySelector('.gym-header');
  if (header) {
    bar.style.top = header.offsetHeight + 'px';
  }

  // ---------- Live search ----------
  var noResults = document.createElement('p');
  noResults.className = 'pm-hub-noresults pm-hidden';
  noResults.textContent = 'No topics match your search.';
  container.appendChild(noResults);

  function runSearch() {
    var q = input.value.trim().toLowerCase();
    var anyVisible = false;
    categories.forEach(function (cat) {
      var cards = cat.querySelectorAll('.card');
      var visibleInCat = 0;
      cards.forEach(function (c) {
        var match = !q || (c.getAttribute('data-search') || '').indexOf(q) !== -1;
        c.classList.toggle('pm-hidden', !match);
        if (match) visibleInCat++;
      });
      cat.classList.toggle('pm-hidden', visibleInCat === 0);
      if (visibleInCat > 0) anyVisible = true;
    });
    noResults.classList.toggle('pm-hidden', anyVisible);
  }

  input.addEventListener('input', runSearch);
})();
