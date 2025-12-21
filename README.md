# Mayank Sharma - Personal Portfolio & Blog

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-success)](https://mayank1101.github.io)
[![Jekyll](https://img.shields.io/badge/Built%20with-Jekyll-red)](https://jekyllrb.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A modern, responsive portfolio website and technical blog showcasing my journey in AI/ML research and engineering. Built with Jekyll and GitHub Pages, featuring a bold design with vibrant animations and a mobile-first approach.

## 🌐 Live Website

Visit: [https://mayank1101.github.io](https://mayank1101.github.io)

## 👨‍💻 About Me

I'm **Mayank Sharma**, an AI/ML Engineer & Researcher with 5+ years of experience building trustworthy Generative AI systems. My work bridges the gap between cutting-edge research and production-scale AI applications.

### Highlights
- 🎓 **Education**: M.Tech in Data Science from IIT Jammu (CGPA: 8.70/10)
- 🏆 **Research**: Co-author of AAAI 2024 (OpenMedLM, NEJM AI)
- 🏅 **Recognition**: Bravo Award at nference for 9× speedup and 8× model compression
- 👥 **Impact**: Built systems serving 10,000+ daily users with 99% efficiency gains
- 🌍 **Global Scale**: Deployed AI across 18+ countries in 9+ languages

### Current Focus
- Multimodal RAG systems
- Retrieval evaluation & hallucination reduction
- Medical NLP & clinical AI
- Production-scale LLM deployment

## 🛠️ Tech Stack

### Core Technologies
- **Jekyll**: Static site generator
- **GitHub Pages**: Hosting and deployment
- **Liquid**: Templating engine
- **Kramdown**: Markdown processor with Rouge syntax highlighting

### Frontend
- **HTML5**: Semantic structure with Jekyll front matter
- **CSS3**: Custom properties, Grid/Flexbox, animations, responsive design
- **JavaScript**: Intersection Observer API, hamburger menu, scroll effects

### Design Tools
- **Google Fonts**: Inter (300-900), JetBrains Mono
- **Font Awesome 6**: Icon library
- **CSS Animations**: Keyframes, transitions, transforms

### Styling
- **Color Palette**:
  - Primary: Cyan (#38bdf8), Purple (#a855f7)
  - Secondary: Pink (#ec4899), Yellow (#fbbf24)
  - Background: Dark slate (#020617, #0f172a)
- **Typography**: Inter for text, JetBrains Mono for code
- **Animations**: fade-in, slide-in, parallax, hover effects

## 📂 Project Structure

```
mayank1101.github.io/
├── _layouts/
│   ├── default.html          # Base layout with CDN links, glow orbs
│   └── post.html              # Blog post layout with series support
├── _posts/
│   ├── 2025-12-15-pytorch-foundation-part1.md
│   └── 2025-12-20-pytorch-foundation-part2.md
├── assets/
│   ├── style.css              # Main stylesheet (1500+ lines)
│   ├── animations.js          # Scroll animations, hamburger menu
│   ├── profile.jpg            # Profile photo
│   └── blog_images/           # Blog series images
│       ├── pytorch-foundation-part1.png
│       └── pytorch-foundation-part2.png
├── index.html                 # Homepage with all sections
├── blog.html                  # Blog listing page
├── _config.yml                # Jekyll configuration
└── README.md                  # This file
```

## 🚀 Local Development

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
   Or simply:
   ```bash
   jekyll serve
   ```

4. **View in browser**
   ```
   http://localhost:4000
   ```

### Live Reload
Jekyll watches for file changes and rebuilds automatically. Refresh your browser to see updates.

## 🌐 Deployment

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

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Jekyll** and **GitHub Pages** for the platform
- **Font Awesome** for beautiful icons
- **Google Fonts** for Inter and JetBrains Mono

---

**Built with ❤️ by Mayank Sharma**
*Bridging research and real-world AI systems*
