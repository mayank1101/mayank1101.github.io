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

  // ===== Add floating particles (Ghibli dust motes) =====
  if (!isMobile && !prefersReducedMotion) {
    const particleContainer = document.createElement('div');
    particleContainer.className = 'particle-container';
    particleContainer.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: -1;
      overflow: hidden;
    `;
    document.body.appendChild(particleContainer);

    const createParticle = () => {
      const particle = document.createElement('div');
      const size = Math.random() * 6 + 2;
      const startX = Math.random() * window.innerWidth;
      const duration = Math.random() * 10 + 15;
      
      particle.style.cssText = `
        position: absolute;
        width: ${size}px;
        height: ${size}px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 50%;
        left: ${startX}px;
        top: -10px;
        animation: particleFall ${duration}s linear forwards;
        opacity: ${Math.random() * 0.5 + 0.3};
      `;
      
      particleContainer.appendChild(particle);
      
      setTimeout(() => particle.remove(), duration * 1000);
    };

    // Add particle fall animation
    if (!document.querySelector('#particle-styles')) {
      const style = document.createElement('style');
      style.id = 'particle-styles';
      style.textContent = `
        @keyframes particleFall {
          0% {
            transform: translateY(0) translateX(0) rotate(0deg);
          }
          100% {
            transform: translateY(100vh) translateX(${Math.random() * 200 - 100}px) rotate(360deg);
          }
        }
      `;
      document.head.appendChild(style);
    }

    // Create particles periodically
    setInterval(createParticle, 3000);
    
    // Create initial particles
    for (let i = 0; i < 5; i++) {
      setTimeout(createParticle, i * 500);
    }
  }
});

// ===== Navbar Scroll Effect =====
let lastScroll = 0;
const nav = document.querySelector('.nav');

if (nav) {
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 100) {
      nav.style.boxShadow = '0 8px 32px rgba(139, 126, 106, 0.2)';
      nav.style.background = 'rgba(255, 248, 240, 0.98)';
      nav.style.backdropFilter = 'blur(12px)';
      nav.style.borderBottom = '3px solid rgba(232, 220, 200, 0.8)';
    } else {
      nav.style.boxShadow = '0 4px 12px rgba(139, 126, 106, 0.12)';
      nav.style.background = 'rgba(255, 248, 240, 0.95)';
      nav.style.backdropFilter = 'blur(10px)';
      nav.style.borderBottom = '3px solid var(--border)';
    }

    lastScroll = currentScroll;
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
