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

## ✨ Features

### 🎨 Design & UX
- **Bold Visual Design**: Vibrant color gradients (cyan, purple, pink, yellow)
- **Smooth Animations**: Scroll-triggered fade-ins, parallax effects, hover interactions
- **Mobile-First**: Fully responsive with collapsible navigation
- **Dark Theme**: Professional dark background with glowing accents
- **Accessibility**: Touch-friendly targets, reduced motion support

### 📱 Responsive Navigation
- Desktop: Horizontal navigation bar with sticky positioning
- Mobile: Collapsible hamburger menu with smooth slide animations
- Auto-close on link click or outside click
- Active page highlighting

### 📝 Blog System
- **Tensor Tales**: Technical blog covering AI/ML, NLP, LLMs, and RAG systems
- **Series Support**: NLP Mastery Series with custom branding
- **Background Images**: Blog cards with full background images and overlays
- **Syntax Highlighting**: Rouge-based code highlighting with JetBrains Mono font
- **Series Metadata**: Author attribution, series badges, custom images

### 🎯 Key Sections
1. **Hero**: Research & production identity with IIT Jammu credentials
2. **Research & Industry**: Applied research, production systems, teaching
3. **Impact & Achievements**: AAAI 2024, user metrics, awards, global reach
4. **My Journey**: Academic foundation → Medical NLP → Production scaling
5. **Tensor Tales**: Technical blog with NLP Mastery series
6. **Resume**: Downloadable PDF with experience summary
7. **Collaboration**: Multiple CTAs for research, industry, mentoring

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
├── Mayank_Sharma_Resume.pdf   # Resume PDF
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

## 📝 Adding Blog Posts

### Create New Post

1. Create a new file in `_posts/` with format: `YYYY-MM-DD-title.md`

2. Add front matter:
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: 2025-12-21
   series: "NLP Mastery Series"
   series_author: "Tensor Tales"
   series_image: "/assets/blog_images/your-image.png"
   excerpt: "Brief description for preview cards..."
   ---
   ```

3. Write your content in Markdown

4. Add code blocks with syntax highlighting:
   ````markdown
   ```python
   def hello_world():
       print("Hello, World!")
   ```
   ````

### Blog Series

The website supports blog series with:
- **Series name**: Displayed as a badge on cards
- **Series author**: Author attribution (e.g., "Tensor Tales")
- **Series image**: Background image for blog cards
- **Series banner**: Shown at top of blog posts
- **Series footer**: Navigation to all articles

## 🎨 Customization

### Colors
Edit CSS variables in `assets/style.css`:
```css
:root {
  --accent-cyan: #38bdf8;
  --accent-purple: #a855f7;
  --accent-pink: #ec4899;
  --accent-yellow: #fbbf24;
}
```

### Animations
Modify animation settings in `assets/animations.js`:
- Scroll thresholds
- Parallax speeds
- Mobile behavior

### Content
Update sections in `index.html`:
- Hero text
- Badges
- Achievement cards
- Journey narrative

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

## 📊 Features Breakdown

### Mobile Optimizations
- ✅ Collapsible hamburger navigation
- ✅ Touch-friendly 48px minimum targets
- ✅ Responsive typography (clamp)
- ✅ Optimized image sizes
- ✅ Disabled resource-intensive effects (glow orbs, parallax)
- ✅ Faster animations (0.5s vs 0.8s)

### Performance
- ✅ Lazy loading with Intersection Observer
- ✅ GPU-accelerated animations (transform3d)
- ✅ Efficient CSS (custom properties)
- ✅ Minimal JavaScript
- ✅ Static site (fast loading)

### SEO
- ✅ Semantic HTML5
- ✅ Meta descriptions
- ✅ Proper heading hierarchy
- ✅ Alt text for images
- ✅ Clean URLs

## 📚 Blog Content

### Current Series: NLP Mastery
Technical deep-dives into NLP fundamentals and advanced techniques:
- PyTorch Foundation (Part 1 & 2)
- Neural Network Architectures
- Transformer Models
- Fine-tuning Techniques
- RAG Systems

### Writing Style
- Research-backed explanations
- Production-oriented examples
- Code snippets with syntax highlighting
- Visual diagrams and architecture illustrations
- Real-world case studies

## 🔗 Connect

- **Email**: sharma.mayank.iitjammu@gmail.com
- **LinkedIn**: [sharma-mayank-m11](https://www.linkedin.com/in/sharma-mayank-m11)
- **GitHub**: [mayank1101](https://github.com/mayank1101)
- **Medium**: [tensor.tales](https://medium.com/@tensor.tales)

## 📄 Resume

Download my full resume: [Mayank_Sharma_Resume.pdf](../Mayank_Sharma_Resume.pdf)

**Experience Highlights**:
- **Lamipak** (Nov 2024 - Present): Data Scientist, Hong Kong
- **LG Soft India** (Aug 2024 - Oct 2024): Research Engineer
- **NeurIOT Labs** (Mar 2024 - May 2024): Lead Data Scientist
- **SFO Technologies** (Mar 2023 - Aug 2023): Lead Software Engineer
- **nference.ai** (Jul 2021 - Mar 2023): Research Data Scientist
- **IIT Jammu** (Jul 2019 - Jun 2021): Teaching Assistant

## 🎓 Education

- **M.Tech in Computer Science - Data Science**
  Indian Institute of Technology (IIT) Jammu
  CGPA: 8.70/10 (Jul 2019 - Jun 2021)

- **B.Tech in Computer Science**
  University of Petroleum and Energy Studies (UPES)
  CGPA: 81.2% (Jul 2013 - May 2017)

## 🏆 Notable Achievements

- **AAAI 2024 Co-Author**: OpenMedLM (NEJM AI)
- **Bravo Award**: nference (9× speedup, 8× compression)
- **GATE CSE 2019**: AIR 1406, Top 98.9% nationally
- **Production Impact**: 10,000+ daily users, 99% efficiency gain
- **Global Deployment**: 18+ countries, 9+ languages, 200+ data sources

## 📖 Publications

**OpenMedLM** - AAAI 2024, NEJM AI
Demonstrated that prompt engineering can outperform fine-tuning in medical question answering using open-source LLMs.

## 💡 Philosophy

> "My humility isn't flattery — it's a value shaped by a humble upbringing."

I believe in building AI systems that bridge research and reality — moving beyond demos to create trustworthy, scalable solutions that serve real people in real environments.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Jekyll** and **GitHub Pages** for the platform
- **Font Awesome** for beautiful icons
- **Google Fonts** for Inter and JetBrains Mono
- **Tensor Tales** community for inspiration

---

**Built with ❤️ by Mayank Sharma**
*Bridging research and real-world AI systems*
