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

  // ===== SPIRITED AWAY THEME - Lanterns, Water Ripples & Paper Charms =====
  if (!prefersReducedMotion) {
    
    // === 1. Floating Lanterns Container ===
    const lanternContainer = document.createElement('div');
    lanternContainer.className = 'lantern-container';
    lanternContainer.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
      overflow: hidden;
    `;
    document.body.appendChild(lanternContainer);

    // Create floating lanterns
    const createLantern = () => {
      const lantern = document.createElement('div');
      const size = Math.random() * 20 + 15; // 15-35px
      const startX = Math.random() * window.innerWidth;
      const duration = Math.random() * 15 + 20; // 20-35s to float up
      const swayDuration = Math.random() * 3 + 2; // 2-5s sway
      const delay = Math.random() * 10;
      const hue = Math.random() > 0.5 ?
        `rgba(255, 180, 100, ${Math.random() * 0.3 + 0.5})` : // warm orange
        `rgba(255, 220, 150, ${Math.random() * 0.3 + 0.5})`; // warm yellow
      
      lantern.className = 'spirit-lantern';
      lantern.innerHTML = `
        <svg viewBox="0 0 40 60" width="${size}" height="${size * 1.5}">
          <defs>
            <radialGradient id="lanternGlow${Date.now()}" cx="50%" cy="50%" r="50%">
              <stop offset="0%" style="stop-color:rgba(255, 250, 200, 0.9)"/>
              <stop offset="50%" style="stop-color:${hue}"/>
              <stop offset="100%" style="stop-color:rgba(255, 150, 50, 0.2)"/>
            </radialGradient>
          </defs>
          <!-- Lantern body -->
          <ellipse cx="20" cy="30" rx="15" ry="20" fill="url(#lanternGlow${Date.now()})" opacity="0.8"/>
          <!-- Lantern frame lines -->
          <path d="M8 15 Q20 5 32 15" stroke="rgba(180, 100, 50, 0.4)" stroke-width="1" fill="none"/>
          <path d="M8 45 Q20 55 32 45" stroke="rgba(180, 100, 50, 0.4)" stroke-width="1" fill="none"/>
          <line x1="8" y1="15" x2="8" y2="45" stroke="rgba(180, 100, 50, 0.3)" stroke-width="0.5"/>
          <line x1="32" y1="15" x2="32" y2="45" stroke="rgba(180, 100, 50, 0.3)" stroke-width="0.5"/>
          <!-- Inner glow -->
          <ellipse cx="20" cy="30" rx="8" ry="12" fill="rgba(255, 255, 220, 0.6)"/>
        </svg>
      `;
      lantern.style.cssText = `
        position: absolute;
        left: ${startX}px;
        bottom: -80px;
        filter: drop-shadow(0 0 ${size/2}px rgba(255, 200, 100, 0.6));
        animation: lanternFloat ${duration}s ease-in-out ${delay}s infinite, lanternSway ${swayDuration}s ease-in-out infinite;
        opacity: 0;
      `;
      
      lanternContainer.appendChild(lantern);
      
      // Remove and recreate after animation
      setTimeout(() => {
        lantern.remove();
        createLantern();
      }, (duration + delay) * 1000);
    };

    // === 2. Paper Charms (Ofuda) Container ===
    const charmContainer = document.createElement('div');
    charmContainer.className = 'charm-container';
    charmContainer.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
    `;
    document.body.appendChild(charmContainer);

    // Create paper charms
    const createPaperCharm = () => {
      const charm = document.createElement('div');
      const width = Math.random() * 15 + 10; // 10-25px wide
      const height = width * 2.5; // Paper ratio
      const startX = Math.random() * window.innerWidth;
      const duration = Math.random() * 20 + 15;
      const flutterDuration = Math.random() * 2 + 1;
      const delay = Math.random() * 8;
      
      charm.className = 'paper-charm';
      charm.style.cssText = `
        position: absolute;
        width: ${width}px;
        height: ${height}px;
        left: ${startX}px;
        top: -${height + 20}px;
        background: linear-gradient(180deg,
          rgba(255, 250, 240, 0.9) 0%,
          rgba(255, 245, 230, 0.8) 50%,
          rgba(245, 235, 220, 0.7) 100%);
        border-radius: 2px;
        box-shadow: 0 2px 4px rgba(139, 126, 106, 0.2);
        animation: charmFall ${duration}s ease-in-out ${delay}s infinite, charmFlutter ${flutterDuration}s ease-in-out infinite;
        opacity: 0;
      `;
      
      // Add kanji-like decoration
      const decoration = document.createElement('div');
      decoration.style.cssText = `
        position: absolute;
        top: 20%;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        height: 50%;
        background: linear-gradient(180deg,
          rgba(200, 80, 80, 0.3) 0%,
          rgba(200, 80, 80, 0.1) 100%);
        border-radius: 1px;
      `;
      charm.appendChild(decoration);
      
      charmContainer.appendChild(charm);
      
      setTimeout(() => {
        charm.remove();
        createPaperCharm();
      }, (duration + delay) * 1000);
    };

    // Add Spirited Away theme animations
    if (!document.querySelector('#spirited-away-styles')) {
      const style = document.createElement('style');
      style.id = 'spirited-away-styles';
      style.textContent = `
        /* Lantern floating up */
        @keyframes lanternFloat {
          0% {
            bottom: -80px;
            opacity: 0;
            transform: translateX(0);
          }
          5% {
            opacity: 0.8;
          }
          95% {
            opacity: 0.8;
          }
          100% {
            bottom: 110vh;
            opacity: 0;
            transform: translateX(${Math.random() > 0.5 ? '' : '-'}${Math.random() * 50}px);
          }
        }
        
        /* Lantern gentle sway */
        @keyframes lanternSway {
          0%, 100% {
            transform: translateX(0) rotate(-3deg);
          }
          50% {
            transform: translateX(15px) rotate(3deg);
          }
        }
        
        /* Paper charm falling */
        @keyframes charmFall {
          0% {
            top: -60px;
            opacity: 0;
          }
          5% {
            opacity: 0.7;
          }
          90% {
            opacity: 0.5;
          }
          100% {
            top: 110vh;
            opacity: 0;
          }
        }
        
        /* Paper charm flutter */
        @keyframes charmFlutter {
          0%, 100% {
            transform: rotateY(0deg) rotateZ(-10deg);
          }
          25% {
            transform: rotateY(30deg) rotateZ(5deg);
          }
          50% {
            transform: rotateY(0deg) rotateZ(10deg);
          }
          75% {
            transform: rotateY(-30deg) rotateZ(-5deg);
          }
        }
        
        /* Water ripple effect on card hover */
        .card::after,
        .blog-card::after,
        .blog-card-list::after {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 0;
          height: 0;
          border-radius: 50%;
          background: radial-gradient(circle,
            rgba(135, 206, 235, 0.3) 0%,
            rgba(180, 215, 233, 0.1) 50%,
            transparent 70%);
          transform: translate(-50%, -50%);
          transition: width 0.6s ease, height 0.6s ease, opacity 0.6s ease;
          opacity: 0;
          pointer-events: none;
          z-index: -1;
        }
        
        .card:hover::after,
        .blog-card:hover::after,
        .blog-card-list:hover::after {
          width: 300%;
          height: 300%;
          opacity: 1;
        }
        
        /* Spirit orb glow for badges */
        .badges span::before {
          content: '';
          position: absolute;
          inset: -3px;
          border-radius: 25px;
          background: linear-gradient(45deg,
            rgba(255, 200, 100, 0.3),
            rgba(255, 150, 100, 0.2),
            rgba(212, 181, 232, 0.3));
          opacity: 0;
          transition: opacity 0.4s ease;
          z-index: -1;
        }
        
        .badges span:hover::before {
          opacity: 1;
          animation: spiritGlow 2s ease-in-out infinite;
        }
        
        @keyframes spiritGlow {
          0%, 100% { filter: blur(8px); }
          50% { filter: blur(12px); }
        }
        
        /* Bathhouse lantern glow for section headers */
        h2 i {
          position: relative;
        }
        
        h2 i::after {
          content: '';
          position: absolute;
          inset: -10px;
          border-radius: 50%;
          background: radial-gradient(circle,
            rgba(255, 200, 100, 0.4) 0%,
            transparent 70%);
          animation: iconLanternGlow 3s ease-in-out infinite;
          z-index: -1;
        }
        
        @keyframes iconLanternGlow {
          0%, 100% {
            opacity: 0.5;
            transform: scale(1);
          }
          50% {
            opacity: 0.8;
            transform: scale(1.1);
          }
        }
      `;
      document.head.appendChild(style);
    }

    // Create initial lanterns (more visible in dark mode)
    const createInitialLanterns = () => {
      const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
      const lanternCount = isDarkMode ? 8 : 5;
      for (let i = 0; i < lanternCount; i++) {
        setTimeout(createLantern, i * 800);
      }
    };

    // Create initial paper charms
    const createInitialCharms = () => {
      for (let i = 0; i < 4; i++) {
        setTimeout(createPaperCharm, i * 2000);
      }
    };

    // Initialize
    createInitialLanterns();
    createInitialCharms();

    // Listen for theme changes to adjust lantern count
    const themeObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === 'data-theme') {
          // Add more lanterns in dark mode
          const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
          if (isDarkMode) {
            for (let i = 0; i < 3; i++) {
              setTimeout(createLantern, i * 500);
            }
          }
        }
      });
    });
    
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    // === 3. Add water ripple effect on interactive elements ===
    document.querySelectorAll('.card, .blog-card, .blog-card-list').forEach(card => {
      card.style.position = 'relative';
      card.style.overflow = 'hidden';
    });
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
