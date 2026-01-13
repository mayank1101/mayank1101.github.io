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

  // Scroll animations (enabled on all devices unless reduced motion is preferred)
  if (!prefersReducedMotion) {
    const observerOptions = {
      threshold: isMobile ? 0.05 : 0.1,
      rootMargin: isMobile ? '0px 0px -40px 0px' : '0px 0px -80px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('animate-in');
        }
      });
    }, observerOptions);

    // Observe all scroll-animate elements
    document.querySelectorAll('.scroll-animate').forEach(el => {
      observer.observe(el);
    });
  } else {
    // If reduced motion is preferred, show all elements immediately
    document.querySelectorAll('.scroll-animate').forEach(el => {
      el.classList.add('animate-in');
    });
  }

  // Parallax effect for glow orbs (desktop only)
  if (!isMobile) {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      const parallaxElements = document.querySelectorAll('.parallax');

      parallaxElements.forEach(el => {
        const speed = parseFloat(el.dataset.speed) || 0.5;
        const yPos = -(scrolled * speed);
        el.style.transform = `translate3d(0, ${yPos}px, 0)`;
      });
    });
  }

  // Scroll progress indicator (for blog posts)
  const progressBar = document.querySelector('.scroll-progress');
  if (progressBar) {
    window.addEventListener('scroll', () => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      progressBar.style.width = scrolled + '%';
    });
  }

  // Profile photo zoom on scroll (subtle)
  const profilePhoto = document.querySelector('.profile-photo');
  if (profilePhoto) {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      const scale = Math.min(1 + (scrolled / 2000), 1.05);
      profilePhoto.style.transform = `scale(${scale})`;
    });
  }

  // Add ripple effect to CTA buttons
  document.querySelectorAll('.cta').forEach(button => {
    button.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.left = e.clientX - rect.left + 'px';
      ripple.style.top = e.clientY - rect.top + 'px';
      this.appendChild(ripple);

      setTimeout(() => ripple.remove(), 600);
    });
  });

  // Typing effect for hero heading (optional - subtle)
  const heroHeading = document.querySelector('.hero h1');
  if (heroHeading && heroHeading.dataset.typing) {
    const text = heroHeading.textContent;
    heroHeading.textContent = '';
    let i = 0;

    const typeWriter = () => {
      if (i < text.length) {
        heroHeading.textContent += text.charAt(i);
        i++;
        setTimeout(typeWriter, 50);
      }
    };

    setTimeout(typeWriter, 300);
  }
});

// Navbar scroll effect
let lastScroll = 0;
const nav = document.querySelector('.nav');

if (nav) {
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 100) {
      nav.style.boxShadow = '0 8px 32px rgba(139, 126, 106, 0.2)';
      nav.style.background = 'rgba(255, 248, 240, 0.98)';
      nav.style.backdropFilter = 'blur(12px)';
    } else {
      nav.style.boxShadow = '0 4px 12px rgba(139, 126, 106, 0.12)';
      nav.style.background = 'rgba(255, 248, 240, 0.95)';
      nav.style.backdropFilter = 'blur(10px)';
    }

    lastScroll = currentScroll;
  });
}
