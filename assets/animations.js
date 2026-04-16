// ===== Professional Modern Theme — Animations & Interactions =====

(function () {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isMobile = window.matchMedia('(max-width: 768px)').matches;

  // -----------------------------------------------------------
  // Scroll progress bar (injected if not already in markup)
  // -----------------------------------------------------------
  const ensureProgressBar = () => {
    let bar = document.querySelector('.scroll-progress');
    if (!bar) {
      bar = document.createElement('div');
      bar.className = 'scroll-progress';
      document.body.prepend(bar);
    }
    return bar;
  };

  // -----------------------------------------------------------
  // Auto-tag elements for scroll reveal
  // Any direct child of .container, .section, .cards, plus cards/badges,
  // gets the .scroll-animate class if it isn't already flagged.
  // -----------------------------------------------------------
  const autoTagReveal = () => {
    const selectors = [
      '.hero > *',
      '.section > *',
      '.container > *',
      '.cards > *',
      '.blog-card',
      '.blog-card-list',
      '.cert-card',
      '.card',
    ];

    const nodes = document.querySelectorAll(selectors.join(','));
    nodes.forEach((el) => {
      if (
        el.classList.contains('scroll-animate') ||
        el.classList.contains('slide-in-left') ||
        el.classList.contains('slide-in-right') ||
        el.classList.contains('section-reveal') ||
        el.classList.contains('no-animate') ||
        el.tagName === 'SCRIPT' ||
        el.tagName === 'STYLE'
      ) {
        return;
      }
      el.classList.add('scroll-animate');
    });
  };

  // -----------------------------------------------------------
  // Mobile menu toggle
  // -----------------------------------------------------------
  const initMobileMenu = () => {
    const hamburger = document.querySelector('.hamburger');
    const navCenter = document.querySelector('.nav-center');
    const navRight = document.querySelector('.nav-right');
    if (!hamburger || !navCenter || !navRight) return;

    const close = () => {
      hamburger.classList.remove('active');
      navCenter.classList.remove('active');
      navRight.classList.remove('active');
    };

    hamburger.addEventListener('click', (e) => {
      e.stopPropagation();
      hamburger.classList.toggle('active');
      navCenter.classList.toggle('active');
      navRight.classList.toggle('active');
    });

    [...navCenter.querySelectorAll('a'), ...navRight.querySelectorAll('a')].forEach((link) => {
      link.addEventListener('click', close);
    });

    document.addEventListener('click', (e) => {
      if (!hamburger.contains(e.target) && !navCenter.contains(e.target) && !navRight.contains(e.target)) {
        close();
      }
    });
  };

  // -----------------------------------------------------------
  // Wrap tables on mobile so they can scroll horizontally
  // -----------------------------------------------------------
  const wrapTables = () => {
    if (!isMobile) return;
    document.querySelectorAll('.container table, .section table').forEach((table) => {
      if (table.parentElement.classList.contains('table-wrapper')) return;
      const wrapper = document.createElement('div');
      wrapper.className = 'table-wrapper';
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);

      const syncShadow = () => {
        const atEnd = wrapper.scrollLeft + wrapper.clientWidth >= wrapper.scrollWidth - 5;
        wrapper.classList.toggle('scrolled-end', atEnd);
      };
      wrapper.addEventListener('scroll', syncShadow, { passive: true });
      syncShadow();
    });
  };

  // -----------------------------------------------------------
  // Scroll reveal via IntersectionObserver with stagger
  // -----------------------------------------------------------
  const initScrollReveal = () => {
    const targets = document.querySelectorAll(
      '.scroll-animate, .slide-in-left, .slide-in-right, .card-animate, .section-reveal'
    );

    if (prefersReducedMotion) {
      targets.forEach((el) => {
        el.classList.add('animate-in', 'visible', 'revealed');
      });
      return;
    }

    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;

          // Stagger siblings inside the same grid/container.
          const parent = el.parentElement;
          let delay = 0;
          if (parent && (parent.classList.contains('cards') || parent.classList.contains('badges'))) {
            const siblings = Array.from(parent.children).filter((c) =>
              c.classList.contains('scroll-animate') ||
              c.classList.contains('card-animate')
            );
            const idx = siblings.indexOf(el);
            delay = Math.min(idx, 8) * 60;
          }

          el.style.transitionDelay = `${delay}ms`;
          el.classList.add('animate-in', 'visible', 'revealed');
          obs.unobserve(el);
        });
      },
      {
        threshold: 0.08,
        rootMargin: isMobile ? '0px 0px -40px 0px' : '0px 0px -80px 0px',
      }
    );

    targets.forEach((el) => observer.observe(el));
  };

  // -----------------------------------------------------------
  // Scroll progress + nav elevation on scroll
  // -----------------------------------------------------------
  const initScrollEffects = () => {
    const bar = ensureProgressBar();
    const nav = document.querySelector('.nav');
    let ticking = false;

    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const scrollTop = window.scrollY || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const pct = height > 0 ? (scrollTop / height) * 100 : 0;
        bar.style.width = pct + '%';

        if (nav) {
          nav.classList.toggle('nav-scrolled', scrollTop > 8);
        }
        ticking = false;
      });
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  };

  // -----------------------------------------------------------
  // Smooth anchor scrolling
  // -----------------------------------------------------------
  const initSmoothAnchors = () => {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener('click', function (e) {
        const hash = this.getAttribute('href');
        if (!hash || hash === '#') return;
        const target = document.querySelector(hash);
        if (!target) return;
        e.preventDefault();
        target.scrollIntoView({
          behavior: prefersReducedMotion ? 'auto' : 'smooth',
          block: 'start',
        });
      });
    });
  };

  // -----------------------------------------------------------
  // Counter animation (for elements with .counter and data-target)
  // -----------------------------------------------------------
  const initCounters = () => {
    const counters = document.querySelectorAll('.counter');
    if (!counters.length) return;

    const animate = (el, target, duration = 1500) => {
      if (prefersReducedMotion) {
        el.textContent = String(target);
        return;
      }
      const start = performance.now();
      const step = (now) => {
        const p = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = String(Math.floor(eased * target));
        if (p < 1) requestAnimationFrame(step);
        else el.textContent = String(target);
      };
      requestAnimationFrame(step);
    };

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;
          const target = parseInt(el.dataset.target || el.textContent, 10);
          el.textContent = '0';
          animate(el, target);
          obs.unobserve(el);
        });
      },
      { threshold: 0.4 }
    );

    counters.forEach((el) => obs.observe(el));
  };

  // -----------------------------------------------------------
  // Lazy image fade-in
  // -----------------------------------------------------------
  const initLazyImages = () => {
    const imgs = document.querySelectorAll('img[data-src]');
    if (!imgs.length) return;

    const obs = new IntersectionObserver((entries, o) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const img = entry.target;
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.4s ease';
        img.src = img.dataset.src;
        img.onload = () => {
          img.style.opacity = '1';
          img.removeAttribute('data-src');
        };
        o.unobserve(img);
      });
    });

    imgs.forEach((img) => obs.observe(img));
  };

  // -----------------------------------------------------------
  // Theme toggle (light / dark) with persistence
  // -----------------------------------------------------------
  const initThemeToggle = () => {
    const updateMetaThemeColor = (theme) => {
      const metas = document.querySelectorAll('meta[name="theme-color"]');
      metas.forEach((m) => {
        const media = m.getAttribute('media') || '';
        if (theme === 'dark' && (media.includes('dark') || !media)) {
          m.setAttribute('content', '#1C1B20');
        } else if (theme === 'light' && (media.includes('light') || !media)) {
          m.setAttribute('content', '#FBF8F1');
        }
      });
    };

    const saved = localStorage.getItem('theme');
    if (saved === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
      updateMetaThemeColor('dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
      updateMetaThemeColor('light');
    }

    const toggle = document.querySelector('.theme-toggle');
    if (!toggle) return;

    toggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      if (next === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      localStorage.setItem('theme', next);
      updateMetaThemeColor(next);
    });

    // Respect system preference if user hasn't chosen yet
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    mql.addEventListener('change', (e) => {
      if (localStorage.getItem('theme')) return;
      if (e.matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
        updateMetaThemeColor('dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
        updateMetaThemeColor('light');
      }
    });
  };

  // -----------------------------------------------------------
  // Boot
  // -----------------------------------------------------------
  const boot = () => {
    initThemeToggle();
    initMobileMenu();
    wrapTables();
    autoTagReveal();
    initScrollReveal();
    initScrollEffects();
    initSmoothAnchors();
    initCounters();
    initLazyImages();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
