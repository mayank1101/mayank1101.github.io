# Mayank Sharma - Personal Portfolio & Blog

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-success)](https://mayank1101.github.io)
[![Jekyll](https://img.shields.io/badge/Built%20with-Jekyll-red)](https://jekyllrb.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A modern, responsive portfolio website and technical blog showcasing my transition from AI & Analytics into Product Management. Built with Jekyll and GitHub Pages, featuring a bold design with vibrant animations and a mobile-first approach. Highlighting how technical depth in ML, NLP, and data systems informs better product decisions.

## Live Website

Visit: [https://hellomayank.in](https://hellomayank.in)

## About Me

I'm **Mayank Sharma**, an AI & Analytics professional transitioning into Product Management. With 5+ years building production ML/NLP systems and analytics platforms, I'm making an intentional move into product leadership. My technical depth in data science, AI, and analytics is my competitive advantage — I understand engineering constraints, can credibly partner with technical teams, and make smarter product decisions grounded in data. This rare combination of technical credibility and product thinking is what I bring to PM roles.

### Highlights
- **Education**: M.Tech in Data Science from IIT Jammu (CGPA: 8.70/10)
- **Research**: Co-author of AAAI 2024 (OpenMedLM, NEJM AI)
- **Recognition**: Bravo Award at nference for 9x speedup and 8x model compression
- **Impact**: Built systems serving 10,000+ daily users with 99% efficiency gains
- **Global Scale**: Deployed AI Analytics Platform covering information across 18+ countries in 9+ languages

### Current Focus (Transitioning to PM)
- Product Strategy & Roadmapping (with technical depth)
- Data-Driven Product Decisions & Evaluation Rigor
- User Research & Problem Discovery
- Cross-Functional Team Leadership (with engineering credibility)
- Building Products that Matter at Scale
- Bridging AI/Analytics and Product Leadership

## Tech Stack

### Core Technologies
- **Jekyll**: Static site generator
- **GitHub Pages**: Hosting and deployment
- **SCSS/Sass**: Modular CSS preprocessing
- **Liquid**: Templating engine
- **Kramdown**: Markdown processor with Rouge syntax highlighting

### Frontend
- **HTML5**: Semantic structure with Jekyll front matter
- **CSS3**: Custom properties, Grid/Flexbox, animations, responsive design
- **JavaScript**: Intersection Observer API, hamburger menu, scroll effects

### Design
- **Google Fonts**: Nunito, Quicksand, JetBrains Mono
- **Font Awesome 6**: Icon library
- **CSS Animations**: Keyframes, transitions, transforms
- **Dark/Light Mode**: Theme toggle with localStorage persistence

## Project Structure

```
mayank1101.github.io/
├── _includes/                    # Reusable components
│   ├── navbar.html               # Navigation with dynamic active states
│   ├── footer.html               # Footer with dynamic year
│   ├── head.html                 # SEO meta tags, fonts, CSS
│   └── theme-script.html         # Theme initialization
├── _layouts/
│   ├── default.html              # Base layout
│   ├── page.html                 # Standard page layout
│   ├── post.html                 # Blog post layout with series support
│   └── series.html               # Series listing layout
├── _sass/                        # Modular SCSS
│   ├── _variables.scss           # CSS custom properties & SCSS variables
│   ├── _base.scss                # Reset, typography, utilities
│   ├── _navigation.scss          # Navbar & hamburger menu
│   ├── _hero.scss                # Hero section & profile
│   ├── _cards.scss               # Card components
│   ├── _buttons.scss             # CTAs & buttons
│   ├── _blog.scss                # Blog & series styles
│   ├── _manifesto.scss           # Token by Token section
│   ├── _footer.scss              # Footer styles
│   ├── _animations.scss          # Keyframes & transitions
│   ├── _code.scss                # Code blocks & syntax
│   ├── _tables.scss              # Table styles
│   ├── _about.scss               # About page specific
│   ├── _dark-mode.scss           # Dark theme overrides
│   └── _responsive.scss          # Media queries
├── _posts/                       # Blog posts
├── _data/
│   └── series.yml                # Blog series definitions
├── series/                       # Series landing pages
├── assets/
│   ├── css/
│   │   └── main.scss             # Main SCSS entry point
│   ├── animations.js             # Scroll animations, theme toggle
│   └── profile.jpeg              # Profile photo
├── index.html                    # Homepage
├── about.html                    # About page
├── blog.html                     # Blog listing
├── _config.yml                   # Jekyll configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Local Development

### Prerequisites
- Ruby (2.7+)
- Bundler
- Jekyll

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mayank1101/mayank1101.github.io.git
   cd mayank1101.github.io
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run locally**
   ```bash
   bundle exec jekyll serve
   ```

4. **View in browser**
   ```
   http://localhost:4000
   ```

### Live Reload
Jekyll watches for file changes and rebuilds automatically. Refresh your browser to see updates.

## Deployment

### GitHub Pages (Automatic)

1. Push changes to `main` branch:
   ```bash
   git add .
   git commit -m "Update content"
   git push origin main
   ```

2. GitHub Pages automatically builds and deploys

3. Wait 2-3 minutes for changes to appear

### Manual Build
```bash
bundle exec jekyll build
```
Output in `_site/` directory

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- **Jekyll** and **GitHub Pages** for the platform
- **Font Awesome** for beautiful icons
- **Google Fonts** for Nunito, Quicksand, and JetBrains Mono

---

**Built by Mayank Sharma**
*AI & Analytics professional transitioning into Product Management. Shipping products powered by technical depth and user-centric thinking.*
