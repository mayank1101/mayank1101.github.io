// ===== Studio Ghibli Style Animations =====
// Intersection Observer for scroll-triggered animations

document.addEventListener('DOMContentLoaded', () => {

  // Mobile Menu Toggle
  const hamburger = document.querySelector('.hamburger');
  const navCenter = document.querySelector('.nav-center');
  const navRight = document.querySelector('.nav-right');

  if (hamburger && navCenter && navRight) {
    hamburger.addEventListener('click', () => {
      hamburger.classList.toggle('active');
      navCenter.classList.toggle('active');
      navRight.classList.toggle('active');
    });

    // Close menu when clicking on a link
    const allNavLinks = [...navCenter.querySelectorAll('a'), ...navRight.querySelectorAll('a')];
    allNavLinks.forEach(link => {
      link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navCenter.classList.remove('active');
        navRight.classList.remove('active');
      });
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
      if (!hamburger.contains(e.target) && !navCenter.contains(e.target) && !navRight.contains(e.target)) {
        hamburger.classList.remove('active');
        navCenter.classList.remove('active');
        navRight.classList.remove('active');
      }
    });
  }

  // Check if device is mobile
  const isMobile = window.innerWidth <= 768;
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ===== Scroll Animations =====
  if (!prefersReducedMotion) {
    const observerOptions = {
      threshold: isMobile ? 0.05 : 0.1,
      rootMargin: isMobile ? '0px 0px -40px 0px' : '0px 0px -80px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-in');
          // Add slight random delay for organic feel
          const randomDelay = Math.random() * 100;
          entry.target.style.transitionDelay = `${randomDelay}ms`;
        }
      });
    }, observerOptions);

    // Observe all scroll-animate elements
    document.querySelectorAll('.scroll-animate').forEach(el => {
      observer.observe(el);
    });

    // Also observe slide-in elements
    document.querySelectorAll('.slide-in-left, .slide-in-right').forEach(el => {
      observer.observe(el);
    });

    // Section reveal observer
    const sectionObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.section-reveal').forEach(el => {
      sectionObserver.observe(el);
    });

    // Card animation observer
    const cardObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry, index) => {
        if (entry.isIntersecting) {
          setTimeout(() => {
            entry.target.classList.add('visible');
          }, index * 100); // Stagger effect
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.card-animate').forEach(el => {
      cardObserver.observe(el);
    });

  } else {
    // If reduced motion is preferred, show all elements immediately
    document.querySelectorAll('.scroll-animate, .slide-in-left, .slide-in-right').forEach(el => {
      el.classList.add('animate-in');
    });
    document.querySelectorAll('.section-reveal').forEach(el => {
      el.classList.add('revealed');
    });
    document.querySelectorAll('.card-animate').forEach(el => {
      el.classList.add('visible');
    });
  }

  // ===== Parallax Effects (Desktop Only) =====
  if (!isMobile && !prefersReducedMotion) {
    let ticking = false;

    window.addEventListener('scroll', () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          const scrolled = window.pageYOffset;
          
          // Parallax for glow orbs
          const parallaxElements = document.querySelectorAll('.parallax');
          parallaxElements.forEach(el => {
            const speed = parseFloat(el.dataset.speed) || 0.5;
            const yPos = -(scrolled * speed);
            el.style.transform = `translate3d(0, ${yPos}px, 0)`;
          });

          // Subtle parallax for clouds/orbs
          const glowOrbs = document.querySelectorAll('.glow-orb');
          glowOrbs.forEach((orb, index) => {
            const speed = 0.1 + (index * 0.05);
            const yPos = scrolled * speed;
            orb.style.transform = `translateY(${yPos}px)`;
          });

          ticking = false;
        });
        ticking = true;
      }
    });
  }

  // ===== Scroll Progress Indicator =====
  const progressBar = document.querySelector('.scroll-progress');
  if (progressBar) {
    window.addEventListener('scroll', () => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      progressBar.style.width = scrolled + '%';
    });
  }

  // ===== Profile Photo Animations =====
  const profilePhoto = document.querySelector('.profile-photo');
  if (profilePhoto && !prefersReducedMotion) {
    // Gentle floating effect instead of scale
    profilePhoto.classList.add('floating');
    
    // Remove floating on scroll to prevent conflict
    let scrollTimeout;
    window.addEventListener('scroll', () => {
      profilePhoto.classList.remove('floating');
      clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        profilePhoto.classList.add('floating');
      }, 150);
    });
  }

  // ===== Enhanced Ripple Effect for Buttons =====
  document.querySelectorAll('.cta, .cta-secondary, .pagination-btn').forEach(button => {
    button.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const ripple = document.createElement('span');
      
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        left: ${x}px;
        top: ${y}px;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        transform: scale(0);
        animation: rippleEffect 0.6s ease-out;
        pointer-events: none;
      `;
      
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      this.appendChild(ripple);

      setTimeout(() => ripple.remove(), 600);
    });
  });

  // Add ripple animation to stylesheet
  if (!document.querySelector('#ripple-styles')) {
    const style = document.createElement('style');
    style.id = 'ripple-styles';
    style.textContent = `
      @keyframes rippleEffect {
        to {
          transform: scale(4);
          opacity: 0;
        }
      }
    `;
    document.head.appendChild(style);
  }

  // ===== Card Hover Tilt Effect (Desktop Only) =====
  if (!isMobile && !prefersReducedMotion) {
    document.querySelectorAll('.card, .blog-card, .blog-card-list').forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        const rotateX = (y - centerY) / 20;
        const rotateY = (centerX - x) / 20;
        
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
      });

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0)';
      });
    });
  }

  // ===== Magnetic Button Effect =====
  if (!isMobile && !prefersReducedMotion) {
    document.querySelectorAll('.cta, .social-icon').forEach(btn => {
      btn.addEventListener('mousemove', (e) => {
        const rect = btn.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;
        
        btn.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px)`;
      });

      btn.addEventListener('mouseleave', () => {
        btn.style.transform = 'translate(0, 0)';
      });
    });
  }

  // ===== Text Reveal Animation =====
  const revealText = (element) => {
    const text = element.textContent;
    element.innerHTML = '';
    
    text.split('').forEach((char, i) => {
      const span = document.createElement('span');
      span.textContent = char === ' ' ? '\u00A0' : char;
      span.style.opacity = '0';
      span.style.transform = 'translateY(20px)';
      span.style.display = 'inline-block';
      span.style.transition = `all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) ${i * 30}ms`;
      element.appendChild(span);
      
      setTimeout(() => {
        span.style.opacity = '1';
        span.style.transform = 'translateY(0)';
      }, 100);
    });
  };

  // Apply to elements with text-reveal class
  document.querySelectorAll('.text-reveal').forEach(el => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          revealText(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });
    
    observer.observe(el);
  });

  // ===== Sparkle Effect on Hover =====
  const createSparkle = (e, element) => {
    const sparkle = document.createElement('span');
    sparkle.innerHTML = '✨';
    sparkle.style.cssText = `
      position: absolute;
      left: ${e.clientX - element.getBoundingClientRect().left}px;
      top: ${e.clientY - element.getBoundingClientRect().top}px;
      font-size: 12px;
      pointer-events: none;
      animation: sparkleFloat 1s ease-out forwards;
    `;
    element.appendChild(sparkle);
    setTimeout(() => sparkle.remove(), 1000);
  };

  // Add sparkle animation
  if (!document.querySelector('#sparkle-styles')) {
    const style = document.createElement('style');
    style.id = 'sparkle-styles';
    style.textContent = `
      @keyframes sparkleFloat {
        0% {
          opacity: 1;
          transform: translateY(0) scale(1);
        }
        100% {
          opacity: 0;
          transform: translateY(-30px) scale(0.5);
        }
      }
    `;
    document.head.appendChild(style);
  }

  // Add sparkle to badges
  document.querySelectorAll('.badges span').forEach(badge => {
    badge.style.position = 'relative';
    badge.style.overflow = 'visible';
    badge.addEventListener('mouseenter', (e) => {
      if (!prefersReducedMotion) {
        createSparkle(e, badge);
      }
    });
  });

  // ===== Smooth Counter Animation =====
  const animateCounter = (element, target, duration = 2000) => {
    let start = 0;
    const startTime = performance.now();
    
    const updateCounter = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Ease out quad
      const easeProgress = 1 - (1 - progress) * (1 - progress);
      const current = Math.floor(easeProgress * target);
      
      element.textContent = current;
      
      if (progress < 1) {
        requestAnimationFrame(updateCounter);
      } else {
        element.textContent = target;
      }
    };
    
    requestAnimationFrame(updateCounter);
  };

  // Apply to counter elements
  document.querySelectorAll('.counter').forEach(el => {
    const target = parseInt(el.dataset.target) || parseInt(el.textContent);
    el.textContent = '0';
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(el, target);
          observer.unobserve(el);
        }
      });
    }, { threshold: 0.5 });
    
    observer.observe(el);
  });

  // ===== Cursor Trail Effect (Optional - Desktop Only) =====
  if (!isMobile && !prefersReducedMotion && document.querySelector('.enable-cursor-trail')) {
    const trail = [];
    const trailLength = 10;
    
    for (let i = 0; i < trailLength; i++) {
      const dot = document.createElement('div');
      dot.className = 'cursor-trail-dot';
      dot.style.cssText = `
        position: fixed;
        width: ${10 - i}px;
        height: ${10 - i}px;
        background: linear-gradient(135deg, #90C695, #FFB6C1);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        opacity: ${1 - (i / trailLength)};
        transition: transform 0.1s ease;
      `;
      document.body.appendChild(dot);
      trail.push(dot);
    }
    
    let mouseX = 0, mouseY = 0;
    
    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    });
    
    const animateTrail = () => {
      let x = mouseX, y = mouseY;
      
      trail.forEach((dot, index) => {
        const nextDot = trail[index + 1] || trail[0];
        
        dot.style.left = x + 'px';
        dot.style.top = y + 'px';
        
        x += (parseFloat(nextDot.style.left) - x) * 0.35;
        y += (parseFloat(nextDot.style.top) - y) * 0.35;
      });
      
      requestAnimationFrame(animateTrail);
    };
    
    animateTrail();
  }

  // ===== Typing Effect =====
  const heroHeading = document.querySelector('.hero h1');
  if (heroHeading && heroHeading.dataset.typing && !prefersReducedMotion) {
    const text = heroHeading.textContent;
    heroHeading.textContent = '';
    heroHeading.style.borderRight = '2px solid var(--accent-green)';
    let i = 0;

    const typeWriter = () => {
      if (i < text.length) {
        heroHeading.textContent += text.charAt(i);
        i++;
        setTimeout(typeWriter, 50);
      } else {
        heroHeading.style.borderRight = 'none';
      }
    };

    setTimeout(typeWriter, 300);
  }

  // ===== Image Lazy Loading with Fade =====
  document.querySelectorAll('img[data-src]').forEach(img => {
    img.style.opacity = '0';
    img.style.transition = 'opacity 0.5s ease';
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          img.src = img.dataset.src;
          img.onload = () => {
            img.style.opacity = '1';
          };
          observer.unobserve(img);
        }
      });
    });
    
    observer.observe(img);
  });


  // ===== HOWL'S MOVING CASTLE THEME - Sparkles, Steam & Shimmer =====
  if (!prefersReducedMotion) {

    // === 1. Floating Sparkles Following Cursor ===
    const sparkleContainer = document.createElement('div');
    sparkleContainer.className = 'sparkle-cursor-container';
    sparkleContainer.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
      overflow: hidden;
    `;
    document.body.appendChild(sparkleContainer);

    let lastSparkleTime = 0;
    const sparkleThrottle = 50; // Create sparkle every 50ms

    const createCursorSparkle = (x, y) => {
      const now = Date.now();
      if (now - lastSparkleTime < sparkleThrottle) return;
      lastSparkleTime = now;

      const sparkle = document.createElement('div');
      const size = Math.random() * 8 + 4; // 4-12px
      const duration = Math.random() * 1000 + 800; // 800-1800ms
      const colors = [
        'rgba(255, 215, 0, 0.9)',     // Gold
        'rgba(255, 182, 193, 0.9)',   // Light pink
        'rgba(147, 112, 219, 0.9)',   // Purple
        'rgba(135, 206, 250, 0.9)',   // Sky blue
        'rgba(255, 255, 255, 0.9)'    // White
      ];
      const color = colors[Math.floor(Math.random() * colors.length)];

      // Random offset from cursor
      const offsetX = (Math.random() - 0.5) * 30;
      const offsetY = (Math.random() - 0.5) * 30;

      sparkle.className = 'cursor-sparkle';
      sparkle.style.cssText = `
        position: absolute;
        left: ${x + offsetX}px;
        top: ${y + offsetY}px;
        width: ${size}px;
        height: ${size}px;
        background: ${color};
        border-radius: 50%;
        pointer-events: none;
        animation: cursorSparkleFloat ${duration}ms ease-out forwards;
        box-shadow: 0 0 ${size * 2}px ${color}, 0 0 ${size * 4}px ${color.replace('0.9', '0.4')};
      `;

      sparkleContainer.appendChild(sparkle);

      setTimeout(() => sparkle.remove(), duration);
    };

    // Track cursor movement
    if (!isMobile) {
      document.addEventListener('mousemove', (e) => {
        createCursorSparkle(e.clientX, e.clientY);
      });
    }

    // === 2. Rising Steam/Smoke Wisps ===
    const steamContainer = document.createElement('div');
    steamContainer.className = 'steam-container';
    steamContainer.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    `;
    document.body.appendChild(steamContainer);

    const createSteamWisp = () => {
      const wisp = document.createElement('div');
      const width = Math.random() * 80 + 40; // 40-120px
      const height = Math.random() * 150 + 100; // 100-250px
      const startX = Math.random() * window.innerWidth;
      const duration = Math.random() * 15 + 15; // 15-30s
      const swayDuration = Math.random() * 4 + 3; // 3-7s
      const delay = Math.random() * 5;
      const opacity = Math.random() * 0.15 + 0.05; // 0.05-0.2

      wisp.className = 'steam-wisp';
      wisp.innerHTML = `
        <svg width="${width}" height="${height}" viewBox="0 0 100 150" style="filter: blur(20px);">
          <defs>
            <radialGradient id="steamGradient${Date.now()}" cx="50%" cy="70%" r="60%">
              <stop offset="0%" style="stop-color:rgba(255, 255, 255, ${opacity * 2})"/>
              <stop offset="50%" style="stop-color:rgba(200, 180, 255, ${opacity * 1.5})"/>
              <stop offset="100%" style="stop-color:rgba(150, 130, 200, 0)"/>
            </radialGradient>
          </defs>
          <ellipse cx="50" cy="75" rx="40" ry="60" fill="url(#steamGradient${Date.now()})" opacity="${opacity}"/>
        </svg>
      `;

      wisp.style.cssText = `
        position: absolute;
        left: ${startX}px;
        bottom: -${height}px;
        animation: steamRise ${duration}s ease-out ${delay}s infinite, steamSway ${swayDuration}s ease-in-out infinite;
        opacity: 0;
      `;

      steamContainer.appendChild(wisp);

      setTimeout(() => {
        wisp.remove();
        createSteamWisp(); // Create new wisp to replace this one
      }, (duration + delay) * 1000);
    };

    // === 3. Magical Shimmer Borders on Cards ===
    const addShimmerBorders = () => {
      document.querySelectorAll('.card, .blog-card, .blog-card-list').forEach(card => {
        // Create shimmer border element
        const shimmer = document.createElement('div');
        shimmer.className = 'card-shimmer-border';
        shimmer.style.cssText = `
          position: absolute;
          inset: -2px;
          border-radius: inherit;
          padding: 2px;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(255, 215, 0, 0.4) 25%,
            rgba(147, 112, 219, 0.4) 50%,
            rgba(255, 182, 193, 0.4) 75%,
            transparent 100%
          );
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          opacity: 0;
          pointer-events: none;
          z-index: 10;
          animation: shimmerMove 3s linear infinite;
        `;

        // Ensure card has proper positioning
        if (getComputedStyle(card).position === 'static') {
          card.style.position = 'relative';
        }

        card.appendChild(shimmer);

        // Show shimmer on hover
        card.addEventListener('mouseenter', () => {
          shimmer.style.opacity = '1';
        });

        card.addEventListener('mouseleave', () => {
          shimmer.style.opacity = '0';
        });
      });
    };

    // Add Howl's Moving Castle animations to stylesheet
    if (!document.querySelector('#howls-castle-styles')) {
      const style = document.createElement('style');
      style.id = 'howls-castle-styles';
      style.textContent = `
        /* Cursor sparkle float animation */
        @keyframes cursorSparkleFloat {
          0% {
            opacity: 1;
            transform: translate(0, 0) scale(1) rotate(0deg);
          }
          50% {
            opacity: 0.8;
            transform: translate(${(Math.random() - 0.5) * 40}px, ${-Math.random() * 40 - 20}px) scale(1.2) rotate(180deg);
          }
          100% {
            opacity: 0;
            transform: translate(${(Math.random() - 0.5) * 60}px, ${-Math.random() * 80 - 40}px) scale(0.5) rotate(360deg);
          }
        }

        /* Steam wisp rising */
        @keyframes steamRise {
          0% {
            bottom: -200px;
            opacity: 0;
          }
          10% {
            opacity: 0.8;
          }
          80% {
            opacity: 0.6;
          }
          100% {
            bottom: 110vh;
            opacity: 0;
          }
        }

        /* Steam wisp swaying */
        @keyframes steamSway {
          0%, 100% {
            transform: translateX(-20px) scale(1);
          }
          25% {
            transform: translateX(30px) scale(1.1);
          }
          50% {
            transform: translateX(-10px) scale(0.95);
          }
          75% {
            transform: translateX(20px) scale(1.05);
          }
        }

        /* Shimmer border animation */
        @keyframes shimmerMove {
          0% {
            background-position: -200% 0;
          }
          100% {
            background-position: 200% 0;
          }
        }

        /* Enhanced shimmer with rotation */
        .card-shimmer-border {
          transition: opacity 0.5s ease;
          background-size: 200% 100%;
        }

        /* Magical glow pulse on hover */
        @keyframes magicalGlow {
          0%, 100% {
            filter: drop-shadow(0 0 5px rgba(255, 215, 0, 0.3))
                    drop-shadow(0 0 10px rgba(147, 112, 219, 0.2));
          }
          50% {
            filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.5))
                    drop-shadow(0 0 20px rgba(147, 112, 219, 0.3));
          }
        }

        .card:hover,
        .blog-card:hover,
        .blog-card-list:hover {
          animation: magicalGlow 2s ease-in-out infinite;
        }

        /* Star sparkle variant for cursor */
        .cursor-sparkle:nth-child(3n) {
          clip-path: polygon(
            50% 0%, 61% 35%, 98% 35%, 68% 57%,
            79% 91%, 50% 70%, 21% 91%, 32% 57%,
            2% 35%, 39% 35%
          );
        }

        /* Diamond sparkle variant */
        .cursor-sparkle:nth-child(3n+1) {
          clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
        }
      `;
      document.head.appendChild(style);
    }

    // Initialize steam wisps
    const createInitialSteamWisps = () => {
      const wispCount = isMobile ? 3 : 6;
      for (let i = 0; i < wispCount; i++) {
        setTimeout(createSteamWisp, i * 1500);
      }
    };

    // Initialize shimmer borders
    addShimmerBorders();

    // Re-add shimmer borders when new cards are added dynamically
    const cardObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1 &&
              (node.classList.contains('card') ||
               node.classList.contains('blog-card') ||
               node.classList.contains('blog-card-list'))) {
            addShimmerBorders();
          }
        });
      });
    });

    cardObserver.observe(document.body, {
      childList: true,
      subtree: true
    });

    // Start creating steam wisps
    createInitialSteamWisps();
  }
});

// ===== Navbar Scroll Effect =====
let lastScroll = 0;
const nav = document.querySelector('.nav');

// Helper function to get theme-aware nav colors
const getNavColors = (isScrolled) => {
  const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
  
  if (isDarkMode) {
    return isScrolled ? {
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
      background: 'rgba(26, 34, 52, 0.98)',
      backdropFilter: 'blur(12px)',
      borderBottom: '3px solid rgba(58, 74, 96, 0.8)'
    } : {
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
      background: 'rgba(26, 34, 52, 0.95)',
      backdropFilter: 'blur(10px)',
      borderBottom: '3px solid var(--border)'
    };
  } else {
    return isScrolled ? {
      boxShadow: '0 8px 32px rgba(139, 126, 106, 0.2)',
      background: 'rgba(255, 248, 240, 0.98)',
      backdropFilter: 'blur(12px)',
      borderBottom: '3px solid rgba(232, 220, 200, 0.8)'
    } : {
      boxShadow: '0 4px 12px rgba(139, 126, 106, 0.12)',
      background: 'rgba(255, 248, 240, 0.95)',
      backdropFilter: 'blur(10px)',
      borderBottom: '3px solid var(--border)'
    };
  }
};

// Apply nav styles based on scroll and theme
const updateNavStyles = () => {
  if (!nav) return;
  const currentScroll = window.pageYOffset;
  const colors = getNavColors(currentScroll > 100);
  
  nav.style.boxShadow = colors.boxShadow;
  nav.style.background = colors.background;
  nav.style.backdropFilter = colors.backdropFilter;
  nav.style.borderBottom = colors.borderBottom;
  
  lastScroll = currentScroll;
};

if (nav) {
  window.addEventListener('scroll', updateNavStyles);
  
  // Also update when theme changes
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.attributeName === 'data-theme') {
        updateNavStyles();
      }
    });
  });
  
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
  });
}

// ===== Smooth Scroll Enhancement =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  });
});

// ===== Dark Mode Toggle (Day/Night Forest Theme) =====
const initThemeToggle = () => {
  // Helper function to update theme-color meta tag for mobile browsers
  const updateThemeColor = (theme) => {
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      // Use dark background color for dark mode, light for light mode
      metaThemeColor.setAttribute('content', theme === 'dark' ? '#1A2234' : '#FFF8F0');
    }
  };

  // Check for saved theme preference or system preference
  const savedTheme = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const currentHour = new Date().getHours();
  const isNightTime = currentHour >= 18 || currentHour < 6;

  // Set initial theme based on saved preference only (not auto dark)
  if (savedTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    updateThemeColor('dark');
  } else if (savedTheme === 'light') {
    document.documentElement.removeAttribute('data-theme');
    updateThemeColor('light');
  }
  // Don't auto-enable dark mode - let user choose

  // Apply nav styles immediately after setting theme
  if (typeof updateNavStyles === 'function') {
    updateNavStyles();
  }
  
  // Create theme toggle button if it doesn't exist
  const navSocial = document.querySelector('.nav-social');
  const nav = document.querySelector('.nav');
  
  if (nav && !document.querySelector('.theme-toggle')) {
    const themeToggle = document.createElement('button');
    themeToggle.className = 'theme-toggle';
    themeToggle.setAttribute('aria-label', 'Toggle dark mode');
    themeToggle.setAttribute('title', 'Toggle light/dark mode');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i><i class="fas fa-sun"></i>';
    
    // Insert into nav-social if it exists, or create container in nav-right
    if (navSocial) {
      navSocial.appendChild(themeToggle);
    } else {
      // Find nav-right or create position after nav-center
      const navRight = document.querySelector('.nav-right');
      if (navRight) {
        navRight.insertBefore(themeToggle, navRight.firstChild);
      } else {
        // Fallback: append to nav
        nav.appendChild(themeToggle);
      }
    }
    
    // Toggle theme on click
    themeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

      if (newTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      localStorage.setItem('theme', newTheme);

      // Update theme-color meta tag for mobile browsers
      updateThemeColor(newTheme);
      
      // Update firefly visibility
      const fireflyContainer = document.querySelector('.firefly-container');
      if (fireflyContainer) {
        if (newTheme === 'dark') {
          fireflyContainer.style.display = 'block';
          // Create fireflies if switching to dark mode
          const existingFireflies = fireflyContainer.querySelectorAll('.firefly');
          if (existingFireflies.length === 0) {
            // Trigger firefly creation - dispatch custom event
            document.dispatchEvent(new CustomEvent('createFireflies'));
          }
        } else {
          fireflyContainer.style.display = 'none';
        }
      }
      
      // Update leaf shadow opacity for dark mode
      const leafShadowContainer = document.querySelector('.leaf-shadow-container');
      if (leafShadowContainer) {
        leafShadowContainer.style.opacity = newTheme === 'dark' ? '0.08' : '0.15';
      }
      
      // Animate the toggle
      themeToggle.style.transform = 'rotate(360deg) scale(1.2)';
      setTimeout(() => {
        themeToggle.style.transform = '';
      }, 400);
    });
  }
  
  // Listen for system theme changes (only if no saved preference)
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
      if (e.matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
        updateThemeColor('dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
        updateThemeColor('light');
      }
    }
  });
};

// Initialize theme toggle when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initThemeToggle);
} else {
  initThemeToggle();
}

// Listen for firefly creation event (from theme toggle)
document.addEventListener('createFireflies', () => {
  const fireflyContainer = document.querySelector('.firefly-container');
  if (fireflyContainer && fireflyContainer.querySelectorAll('.firefly').length === 0) {
    // Create fireflies dynamically
    for (let i = 0; i < 15; i++) {
      setTimeout(() => {
        const firefly = document.createElement('div');
        const size = Math.random() * 4 + 2;
        const startX = Math.random() * window.innerWidth;
        const startY = Math.random() * window.innerHeight;
        const duration = Math.random() * 8 + 12;
        const glowDuration = Math.random() * 2 + 1.5;
        
        firefly.className = 'firefly';
        firefly.style.cssText = `
          position: absolute;
          width: ${size}px;
          height: ${size}px;
          background: radial-gradient(circle, rgba(255, 255, 180, 1) 0%, rgba(180, 255, 100, 0.8) 40%, transparent 70%);
          border-radius: 50%;
          left: ${startX}px;
          top: ${startY}px;
          box-shadow: 0 0 ${size * 3}px rgba(200, 255, 100, 0.8), 0 0 ${size * 6}px rgba(180, 255, 100, 0.4);
          animation: fireflyFloat${Math.floor(Math.random() * 3)} ${duration}s ease-in-out infinite, fireflyGlow ${glowDuration}s ease-in-out infinite;
          opacity: 0;
        `;
        
        fireflyContainer.appendChild(firefly);
        
        // Remove after duration
        setTimeout(() => firefly.remove(), duration * 1000);
      }, i * 200);
    }
  }
});
