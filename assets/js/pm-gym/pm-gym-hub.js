// PM Gym — hub-only enhancements: progress badges + per-category bars,
// jump navigation, live search/filter, weak-topic hints, and progress
// export/import. Reads through window.pmProgress (pm-gym-progress.js), which
// the layout loads after this file — hence the DOMContentLoaded wrapper.
(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    var categories = document.querySelectorAll('.pm-category');
    if (!categories.length) return; // not the hub

    var store = window.pmProgress;
    if (!store) return;

    var data = store.load();

    function topicOf(slug) {
      return data.topics[slug] || store.blankTopic();
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

    function scoreOf(topic) {
      var ids = Object.keys(topic.quiz);
      var correct = 0;
      ids.forEach(function (id) {
        if (topic.quiz[id].correct) correct++;
      });
      return { answered: ids.length, correct: correct };
    }

    var allCards = Array.prototype.slice.call(
      document.querySelectorAll('.pm-category .card')
    );
    var totalCount = allCards.length;
    var totalDone = 0;
    var weak = [];

    // ---------- Per-card state + progress meta ----------
    allCards.forEach(function (card) {
      var slug = slugOf(card);
      var topic = slug ? topicOf(slug) : store.blankTopic();
      var state = 'new';
      if (topic.completed) state = 'done';
      else if (topic.visited) state = 'progress';

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

      var s = scoreOf(topic);
      if (s.answered >= 3 && s.correct / s.answered < 0.7) {
        var title = card.querySelector('h3');
        weak.push({
          name: title ? title.textContent.trim() : slug,
          href: card.getAttribute('href'),
          correct: s.correct,
          answered: s.answered
        });
      }

      // Per-card meta line: real progress once there is any, generic before.
      var tags = card.querySelector('.skill-tags');
      if (tags && !card.querySelector('.pm-card-meta')) {
        var meta = document.createElement('p');
        meta.className = 'pm-card-meta';
        meta.appendChild(icon('fa-dumbbell'));
        var text = ' Lessons · flashcards · practice scenarios · vocab game';
        if (s.answered) {
          text = ' ' + s.correct + '/' + s.answered + ' questions right first try';
          if (topic.lesson > 1) text += ' · resume at lesson ' + topic.lesson;
        }
        meta.appendChild(document.createTextNode(text));
        tags.parentNode.insertBefore(meta, tags.nextSibling);
      }

      card.setAttribute('data-search', (card.textContent || '').toLowerCase());
    });

    // ---------- Per-category progress bars + ids ----------
    categories.forEach(function (cat, idx) {
      cat.id = 'cat-' + (idx + 1);
      var cards = cat.querySelectorAll('.card');
      var done = 0;
      cards.forEach(function (c) {
        var s = slugOf(c);
        if (s && topicOf(s).completed) done++;
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

    var summary = document.createElement('div');
    summary.className = 'pm-hub-summary';
    var sumText = document.createElement('span');
    sumText.className = 'pm-hub-summary-text';
    sumText.textContent = totalDone + ' of ' + totalCount + ' topics completed';
    summary.appendChild(icon('fa-trophy'));
    summary.appendChild(sumText);

    var hasProgress = totalDone > 0 || Object.keys(data.topics).length > 0;
    if (hasProgress) {
      summary.appendChild(makeButton('pm-hub-reset', 'Reset progress', function () {
        store.reset();
        location.reload();
      }));
      summary.appendChild(makeButton('pm-hub-export', 'Export', downloadProgress));
    }
    summary.appendChild(makeImport());

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
    var input = document.createElement('input');
    input.type = 'search';
    input.placeholder = 'Search topics…';
    input.className = 'pm-hub-search-input';
    input.setAttribute('aria-label', 'Search topics');
    searchWrap.appendChild(icon('fa-magnifying-glass'));
    searchWrap.appendChild(input);

    tools.appendChild(nav);
    tools.appendChild(searchWrap);
    bar.appendChild(summary);
    bar.appendChild(tools);
    container.insertBefore(bar, categories[0]);

    if (weak.length) container.insertBefore(buildWeakPanel(weak), categories[0]);

    // Sticky offset below the gym header.
    var header = document.querySelector('.gym-header');
    if (header) bar.style.top = header.offsetHeight + 'px';

    // ---------- Live search ----------
    var noResults = document.createElement('p');
    noResults.className = 'pm-hub-noresults pm-hidden';
    noResults.textContent = 'No topics match your search.';
    container.appendChild(noResults);

    input.addEventListener('input', function () {
      var q = input.value.trim().toLowerCase();
      var anyVisible = false;
      categories.forEach(function (cat) {
        var visibleInCat = 0;
        cat.querySelectorAll('.card').forEach(function (c) {
          var match = !q || (c.getAttribute('data-search') || '').indexOf(q) !== -1;
          c.classList.toggle('pm-hidden', !match);
          if (match) visibleInCat++;
        });
        cat.classList.toggle('pm-hidden', visibleInCat === 0);
        if (visibleInCat > 0) anyVisible = true;
      });
      noResults.classList.toggle('pm-hidden', anyVisible);
    });

    // ---------- helpers ----------
    function makeButton(cls, label, onClick) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = cls;
      b.textContent = label;
      b.addEventListener('click', onClick);
      return b;
    }

    function downloadProgress() {
      var blob = new Blob([store.exportJSON()], { type: 'application/json' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'pm-gym-progress.json';
      a.click();
      URL.revokeObjectURL(a.href);
    }

    // localStorage is per-browser, so an explicit file is the only way to carry
    // progress to another device.
    function makeImport() {
      var label = document.createElement('label');
      label.className = 'pm-hub-import';
      label.textContent = 'Import';
      var file = document.createElement('input');
      file.type = 'file';
      file.accept = 'application/json,.json';
      file.className = 'pm-hidden';
      file.addEventListener('change', function () {
        var f = file.files && file.files[0];
        if (!f) return;
        var reader = new FileReader();
        reader.onload = function () {
          try {
            store.importJSON(String(reader.result));
            location.reload();
          } catch (e) {
            label.textContent = "Couldn't read that file";
          }
        };
        reader.readAsText(f);
      });
      label.appendChild(file);
      return label;
    }

    function buildWeakPanel(list) {
      var panel = document.createElement('section');
      panel.className = 'pm-hub-weak';
      var h = document.createElement('h2');
      h.appendChild(icon('fa-arrow-rotate-left'));
      h.appendChild(document.createTextNode(' Worth revisiting'));
      panel.appendChild(h);
      var p = document.createElement('p');
      p.className = 'muted';
      p.textContent = 'You got under 70% right first try on these.';
      panel.appendChild(p);
      var ul = document.createElement('ul');
      list
        .sort(function (a, b) {
          return a.correct / a.answered - b.correct / b.answered;
        })
        .slice(0, 5)
        .forEach(function (t) {
          var li = document.createElement('li');
          var a = document.createElement('a');
          a.href = t.href;
          a.textContent = t.name;
          li.appendChild(a);
          li.appendChild(
            document.createTextNode(' — ' + t.correct + '/' + t.answered + ' right first try')
          );
          ul.appendChild(li);
        });
      panel.appendChild(ul);
      return panel;
    }
  });
})();
