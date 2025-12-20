// Intersection Observer for scroll-triggered animations
document.addEventListener('DOMContentLoaded', () => {

  // Check if device is mobile
  const isMobile = window.innerWidth <= 768;
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Scroll animations (only on desktop or if user allows motion)
  if (!isMobile && !prefersReducedMotion) {
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -80px 0px'
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
    // On mobile, show all elements immediately
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
      nav.style.boxShadow = '0 5px 20px rgba(0, 0, 0, 0.5)';
      nav.style.background = 'rgba(2, 6, 23, 0.95)';
    } else {
      nav.style.boxShadow = 'none';
      nav.style.background = 'rgba(2, 6, 23, 0.8)';
    }

    lastScroll = currentScroll;
  });
}
