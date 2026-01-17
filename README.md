**Multi-model AI orchestration with intelligent routing and answer validation. 100% local via Ollama.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-18.2-61dafb.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/ericvarney87-collab/Orchestra-Multi-Model-AI-System?style=social)](https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System)

---

## Why Orchestra?

Large language models fail predictably. They invoke cosmology for mechanics problems, claim instant responses over light-year distances, and mix quantum with classical physics.

Orchestra fixes this by:
- Routing queries to specialized expert models
- Validating answers for common reasoning failures
- Synthesizing responses from multiple perspectives

**Example:** Ask about a light-year-long spring, most LLMs say "dark energy prevents it returning." Orchestra catches this, routes to physics-aware models, and gives you the correct answer.

---

## Features

### Intelligent Expert Routing
Automatically selects 3 most relevant experts from 18+ specialized models:
- STEM, Math, Reasoning experts
- Medical, Legal, Psychology specialists
- Code, Data Science, DevOps experts
- Creative Writing, History, Philosophy

### Smart Context Management
- Handles 100+ message conversations without breaking
- Uses only 44% of context window (14K/32K tokens)
- Compresses old messages intelligently
- RAG semantic search for long-term memory

### Physics Answer Validation
Detects and penalizes common LLM failures:
- Cosmology invoked for mechanics (e.g., "dark energy affects springs")
- Causality violations (e.g., "instantaneous response over light-years")
- Missing key concepts (e.g., no mention of wave propagation)

**Result:** 75-85% accuracy on physics questions vs 10-15% without validation

### Integrated Secure Browser
- Certificate validation with visual confirmation
- Phishing detection (warns about lookalike domains)
- Download security (blocks dangerous file types)
- 30-minute auto-logout
- Auto-indexes pages into RAG memory

### Document Creation
Professional document generation:
- DOCX (Word documents)
- PPTX (PowerPoint presentations)
- XLSX (Excel spreadsheets)
- PDF files

### Privacy First
- 100% local operation via Ollama
- No cloud dependencies
- Your data never leaves your machine
- Open source (MIT license)

---

## Quick Start

### Prerequisites

```bash
- Python 3.10+
- Node.js 18+
- Ollama installed and running
- 16GB RAM minimum (32GB recommended)
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System.git
cd Orchestra-Multi-Model-AI-System

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Node dependencies
npm install

# 4. Pull Ollama models (examples - see docs for full list)
ollama pull deepseek-r1:8b
ollama pull granite3.3:8b

# 5. Build and run
npm run build
npm run electron:build:appimage

# 6. Launch
cd dist
./Orchestra-2.9.0.AppImage
```

## Performance

| Metric | Value |
|--------|-------|
| Max conversation length | 100+ exchanges |
| Context efficiency | 44% (14K/32K tokens) |
| Expert routing time | 15-20 seconds |
| Blank response rate | <1% |
| Physics accuracy (with validation) | 75-85% |
| Memory usage | ~2GB (base models loaded) |

---

## Architecture

```
User Interface (Electron + React)
        ↓
    Conductor
   (Smart Router)
        ↓
   ┌────┴────┐
   ↓    ↓    ↓
Expert Expert Expert
Model  Model  Model
   (Ollama)
        ↓
  Validation Layer
        ↓
   Synthesis → Answer
```

## Reporting Issues

Found a bug? Have a feature request?

1. Check [existing issues](https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System/issues)
2. If not found, [open a new issue](https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System/issues/new)
3. Include:
   - OS and version
   - Python version
   - Steps to reproduce
   - Expected vs actual behavior

---

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local model hosting
- Electron and React communities
- All contributors and testers

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System/discussions)
- **Email:** eric.varney@yahoo.com
---

**Built with ❤️ for privacy, intelligence, and open source.**

**[⭐ Star this repo](https://github.com/ericvarney87-collab/Orchestra-Multi-Model-AI-System)** if you find it useful!
