import time
from datetime import datetime
from pathlib import Path
import os
import sys
import json
import asyncio
import threading
import time
import re
import io
import base64
import logging
import subprocess
import platform
from datetime import datetime
os.environ['OLLAMA_KV_CACHE_TYPE'] = 'q4_0'
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import psutil
from PIL import Image
import numpy as np
import torch
import ollama
import fitz 
import diffusers
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download # Recommended for path management
from huggingface_hub.utils import enable_progress_bars
from stockfish_engine import StockfishEngine
from code_executor import CodeExecutor
from math_expert_handler import MathExpertHandler

try:
    from duckduckgo_search import DDGS

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False
    print("WARNING: duckduckgo-search not installed. Web search disabled.")
    print("Install with: pip install duckduckgo-search")

# --- [LOGGING CONFIGURATION] ---
logging.basicConfig(level=logging.INFO)
enable_progress_bars()
diffusers.logging.set_verbosity_info()


# --- [WEB SEARCH ENGINE] ---
class WebSearchEngine:
    """Handles web searches using DuckDuckGo"""

    def __init__(self):
        self.available = SEARCH_AVAILABLE

    def search(self, query, max_results=1):
        """Search the web and return formatted results"""
        if not self.available:
            return "Web search unavailable. Install with: pip install duckduckgo-search"

        try:
            with DDGS() as ddgs:
                # Try to get more results to improve hit rate
                results = list(ddgs.text(query, max_results=max_results * 2))

                if not results:
                    # Retry with simpler query (remove common words)
                    simplified = query.replace("the ", "").replace("average ", "").replace("current ", "")
                    if simplified != query:
                        results = list(ddgs.text(simplified, max_results=max_results * 2))

                    if not results:
                        return f"No results found for: {query}\n\nTry:\n• Simpler keywords (e.g., 'gas prices Illinois')\n• Different phrasing\n• Being more specific or more general"

                # Take only the requested amount
                results = results[:max_results]

                formatted = f"Web Search Results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    formatted += f"{i}. {result['title']}\n"
                    formatted += f"   {result['body']}\n"
                    formatted += f"   URL: {result['href']}\n\n"

                return formatted
        except Exception as e:
            error_msg = str(e)
            if "Ratelimit" in error_msg:
                return "DuckDuckGo rate limit reached. Please wait a moment and try again."
            return f"Search error: {error_msg}\n\nTry: pip install -U duckduckgo-search"


# --- [PROGRAM LAUNCHER] ---
class ProgramLauncher:
    """Handles opening programs via text commands"""

    def __init__(self):
        self.os_type = platform.system()

        # Common program mappings
        self.programs = {
            # Browsers
            "chrome": ["google-chrome", "chrome", "chromium", r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
            "firefox": ["firefox", r"C:\Program Files\Mozilla Firefox\firefox.exe"],
            "edge": ["microsoft-edge", "msedge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
            "brave": ["brave-browser", "brave", r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"],

            # Office
            "word": ["libreoffice --writer", r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"],
            "excel": ["libreoffice --calc", r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"],
            "powerpoint": ["libreoffice --impress", r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"],

            # Development
            "vscode": ["code", r"C:\Program Files\Microsoft VS Code\Code.exe"],
            "pycharm": ["pycharm", r"C:\Program Files\JetBrains\PyCharm Community Edition\bin\pycharm64.exe"],
            "terminal": ["x-terminal-emulator", "sensible-terminal", "gnome-terminal", "konsole", "xterm", "cmd.exe"],

            # Media
            "vlc": ["vlc", r"C:\Program Files\VideoLAN\VLC\vlc.exe"],
            "spotify": ["spotify",
                        r"C:\Users\{}\AppData\Roaming\Spotify\Spotify.exe".format(os.getenv('USERNAME', 'user'))],

            # File managers
            "files": ["nautilus", "dolphin", "thunar", "explorer.exe"],
            "explorer": ["nautilus", "dolphin", "explorer.exe"],

            # System
            "calculator": ["gnome-calculator", "kcalc", "calc.exe"],
            "settings": ["gnome-control-center", "systemsettings5", "control.exe"],
        }

    def open_program(self, program_name):
        """Attempts to open a program by name"""
        program_name = program_name.lower().strip()

        if program_name in self.programs:
            paths = self.programs[program_name]

            for path in paths:
                try:
                    if self.os_type == "Windows":
                        if os.path.exists(path):
                            subprocess.Popen(path, shell=True)
                            return True
                        else:
                            subprocess.Popen(path, shell=True)
                            return True
                    else:
                        subprocess.Popen(path.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                except:
                    continue

            return False
        else:
            try:
                if self.os_type == "Windows":
                    subprocess.Popen(program_name, shell=True)
                else:
                    subprocess.Popen(program_name.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except:
                return False

class ModelIntelligence:
    """Detects model capabilities and suggests domain assignments"""

    def __init__(self):
        self.model_signatures = {
            "Code_Logic": {
                "keywords": ["deepseek-coder:6.7b", "code", "programming", "coder", "starcoder"],
                "priority": 1
            },
            "SQL_Database": {
                "keywords": ["sql", "sqlcoder", "database", "defog"],
                "priority": 2
            },
            "DevOps_Cloud": {
                "keywords": ["devops", "kubernetes", "terraform", "infrastructure"],
                "priority": 2
            },
            "STEM_Expert": {
                "keywords": ["qwen3", "granite", "mathstral", "math", "stem", "science"],
                "priority": 1
            },
            "Data_Scientist": {
                "keywords": ["qwen3", "wizard", "data", "analytics", "statistical"],
                "priority": 1
            },
            "Medical_Expert": {
                "keywords": ["med", "medical", "health", "clinical", "biomedical"],
                "priority": 2
            },
            "Creative_Writer": {
                "keywords": ["dolphin", "creative", "writer", "novelist", "storytelling"],
                "priority": 1
            },
            "Finance_Analyst": {
                "keywords": ["finance", "fintech", "investment", "trading", "plutus"],
                "priority": 2
            },
            "Cyber_Security": {
                "keywords": ["security", "cyber", "pentest", "vulnerability", "exploit"],
                "priority": 2
            },
            "Language_Linguist": {
                "keywords": ["qwen", "aya", "dialect", "formal tone", "language", "translate", "synonym", "antonym", "homonym", "etymology", "multilingual", "translation", "polyglot"],
                "priority": 1
            },
            "Legal_Counsel": {
                "keywords": ["legal", "law", "statute", "jurisprudence"],
                "priority": 2
            },
            "Philosophy_Arts": {
                "keywords": ["philosophy", "humanities", "ethics", "gemma"],
                "priority": 1
            },
            "History_Expert": {
                "keywords": ["history", "historical", "chronicle", "herodotus"],
                "priority": 2
            },
           "Network_Engineer": {
                "keywords": ["network", "cisco", "routing", "switching"],
                "priority": 2
            },
            "Neural_Network_Engineer": {
                "keywords": ["neuralnetwork", "neural", "pytorch", "tensorflow", "alientelligence/"],
                "priority": 2
            },
            "Psychology_Counselor": {
                "keywords": ["psychology", "counselor", "mental", "phi4", "reasoning"],
                "priority": 2
            },
            "Business_Strategist": {
                "keywords": ["business", "strategy", "management", "granite4"],
                "priority": 2
            },
            "Research_Scientist": {
                "keywords": ["vanta-Research", "research", "academic", "ministral", "scientist"],
                "priority": 2
            },
            "Vision_Analyst": {
                "keywords": ["vision", "vl", "qwen3-vl", "moondream", "visual"],
                "priority": 2
            },
            "Reasoning_Expert": {
                "keywords": ["reasoning", "deepseek-r1", "logic", "think"],
                "priority": 2
            },
            "GENERAL": {
                 "keywords": ["llama", "mistral", "phi", "gemma", "qwen", "falcon", "claude"],
                 "priority": 0
            },
            "Chess_Analyst": {"keywords": ["chess", "position", "fen", "move", "opening", "endgame", "stockfish", "analyze", "checkmate", "castle", "knight", "bishop", "rook", "queen", "king", "pawn"], "priority": 1}
            
        }

        self.size_patterns = {
            "tiny": ["0.5b", "1b", "1.3b", "1.5b"],
            "small": ["2b", "3b", "4b"],
            "medium": ["7b", "8b", "9b"],
            "large": ["13b", "14b", "15b", "20b"],
            "xlarge": ["30b", "34b", "70b"]
        }

    def detect_domain(self, model_name: str):
        """Returns (suggested_domain, confidence_score)"""
        model_lower = model_name.lower()
        matches = []

        for domain, signature in self.model_signatures.items():
            if domain == "GENERAL":
                continue

            for keyword in signature["keywords"]:
                if keyword in model_lower:
                    confidence = signature["priority"] / 2.0
                    matches.append((domain, confidence, signature["priority"]))
                    break

        if not matches:
            for keyword in self.model_signatures["GENERAL"]["keywords"]:
                if keyword in model_lower:
                    return ("GENERAL_PURPOSE", 0.3)
            return (None, 0.0)

        matches.sort(key=lambda x: x[2], reverse=True)
        return (matches[0][0], matches[0][1])

    def get_model_size(self, model_name: str):
        """Detect model size category"""
        model_lower = model_name.lower()
        for size, patterns in self.size_patterns.items():
            for pattern in patterns:
                if pattern in model_lower:
                    return size
        return "unknown"

    def suggest_conductor(self, available_models: list):
        """Suggest best conductor based on available models"""
        preferred = ["granite3.3:8b", "cogito:8b", "ajindal/llama3.1-storm:8b", "falcon3:7b", "JorgeAtLLama/herodotus:latest", "GandalfBaum/llama3.1-claude3.7:latest", "llama3.2:3b", "llama3.1:8b", "ministral-3:8b ", "qwen2.5:7b"]

        for model in preferred:
            if model in available_models:
                return model

        general_models = []
        for model in available_models:
            domain, confidence = self.detect_domain(model)
            if domain == "GENERAL_PURPOSE" or confidence < 0.4:
                size = self.get_model_size(model)
                general_models.append((model, size))

        size_order = ["medium", "small", "large", "tiny", "xlarge", "unknown"]
        for size in size_order:
            for model, model_size in general_models:
                if model_size == size:
                    return model

        if not available_models:
            raise Exception("No models available - is Ollama running?")
        return available_models[0]

    def suggest_embed_model(self, available_models: list):
        """Suggest best embedding model based on available models"""
        # Preferred embedding models in order
        preferred = [
            "nomic-embed-text",
            "nomic-embed-text:latest", 
            "mxbai-embed-large",
            "all-minilm",
            "snowflake-arctic-embed",
            "bge-large",
            "gte-large"
        ]
        
        for model in preferred:
            if model in available_models:
                return model
        
        # Look for any model with "embed" in the name
        for model in available_models:
            if "embed" in model.lower():
                return model
        
        # No embedding model available - warn and use first model
        print("WARNING: No embedding model found. RAG may not work optimally.")
        print("Recommended: ollama pull nomic-embed-text")
        return "nomic-embed-text"  # Fallback - will error later if not available


# --- [HARDWARE MONITORING ENGINE WITH AMD SUPPORT] ---
class HardwareMonitor:
    def __init__(self):
        self.gpu_ready = False
        self.gpu_type = "none"
        self.history = {"cpu": [0] * 40}
        self.prev_disk_time = time.time()

        try:
            self.prev_disk_busy = psutil.disk_io_counters().busy_time if hasattr(psutil.disk_io_counters(),
                                                                                 'busy_time') else 0
        except:
            self.prev_disk_busy = 0

        # Try NVIDIA first
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_ready = True
            self.gpu_type = "nvidia"
            self.pynvml = pynvml
            print("DEBUG: NVIDIA GPU detected")
        except:
            # Try AMD via PyTorch ROCm detection
            if torch.cuda.is_available() and "hip" in torch.version.cuda:
                self.gpu_ready = True
                self.gpu_type = "amd"
                print("DEBUG: AMD GPU detected (ROCm)")
            else:
                self.gpu_ready = False
                self.gpu_type = "cpu"
                print("DEBUG: No GPU detected, defaulting to CPU mode")

    def get_update(self):
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent

        now = time.time()
        try:
            curr_busy = psutil.disk_io_counters().busy_time
            time_delta = (now - self.prev_disk_time) * 1000
            busy_delta = curr_busy - self.prev_disk_busy
            disk_active_pct = min(100, (busy_delta / time_delta) * 100) if time_delta > 0 else 0
            self.prev_disk_time = now
            self.prev_disk_busy = curr_busy
        except:
            disk_active_pct = 0

        data = {
            "cpu_pct": cpu_pct,
            "ram_pct": ram_pct,
            "disk_pct": disk_active_pct,
            "gpu_load": 0,
            "vram_pct": 0,
            "temp_pct": 0,
            "text": f"Mode: {self.gpu_type.upper()}"
        }

        self.history["cpu"].pop(0)
        self.history["cpu"].append(cpu_pct)

        if self.gpu_ready and self.gpu_type == "nvidia":
            try:
                total_load = 0
                total_vram = 0
                max_temp = 0
                device_count = self.pynvml.nvmlDeviceGetCount()

                gpu_details = []
                for i in range(device_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = self.pynvml.nvmlDeviceGetTemperature(handle, 0)

                    total_load += util.gpu
                    vram_pct = (mem.used / mem.total) * 100
                    total_vram += vram_pct
                    max_temp = max(max_temp, temp)

                    name = self.pynvml.nvmlDeviceGetName(handle)
                    gpu_details.append(f"GPU{i} ({name[:8]}): {util.gpu}%")

                data["gpu_load"] = total_load / device_count
                data["vram_pct"] = total_vram / device_count
                data["temp_pct"] = max_temp
                data["text"] = " | ".join(gpu_details)
            except Exception as e:
                data["text"] = f"NVIDIA Error: {str(e)[:20]}"

        elif self.gpu_ready and self.gpu_type == "amd":
            # AMD basic support
            if torch.cuda.is_available():
                data["text"] = f"AMD GPU: {torch.cuda.get_device_name(0)[:20]}"
                data["vram_pct"] = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(
                    0).total_memory) * 100

        return data


class VisualHardwareMonitor(tk.Canvas):
    def __init__(self, parent, monitor, **kwargs):
        super().__init__(parent, height=70, highlightthickness=0, **kwargs)
        self.monitor = monitor
        self.accent = "#FFD700"

    def refresh(self, data):
        self.delete("all")
        w = self.winfo_width()
        if w < 10: return

        metrics = [
            ("CPU", 10, data["cpu_pct"]),
            ("RAM", 100, data["ram_pct"]),
            ("DISK", 190, data["disk_pct"]),
            ("GPU", 280, data["vram_pct"])
        ]

        for label, x, val in metrics:
            self.create_text(x, 10, text=label, fill="gray", font=("Consolas", 7, "bold"), anchor="nw")
            self.create_rectangle(x, 22, x + 70, 28, fill="#1e1e20", outline="#333333")
            bar_w = (val / 100) * 70
            self.create_rectangle(x, 22, x + bar_w, 28, fill=self.accent, outline="")
            self.create_text(x, 32, text=f"{int(val)}%", fill="white", font=("Consolas", 8), anchor="nw")

        graph_x = w - 210
        self.create_text(graph_x, 10, text="REAL-TIME PERFORMANCE OVERLAY", fill="gray", font=("Consolas", 7),
                         anchor="nw")

        points = []
        for i, val in enumerate(self.monitor.history["cpu"]):
            px = graph_x + (i * 5)
            py = 60 - (val * 0.4)
            points.extend([px, py])

        if len(points) > 3:
            self.create_line(points, fill=self.accent, smooth=True, width=1.5)


# --- [CORE CONFIGURATION] ---
class OrchestraConfig:
    def __init__(self):
        self.theme_mode = "dark"
        self.top_k = 10  # RAG: How many document chunks to retrieve
        self.max_experts = 3  # Expert routing: Maximum experts to call simultaneously
        self.conductor = "tinydolphin"
        self.embed_model = "nomic-embed-text"
        self.rag_enabled = True
        self.vram_optim = True
        self.artisan_enabled = True
        self.stockfish_available = self._check_stockfish()
        self.model_intel = ModelIntelligence()
        self.manual_assignments = {}
        self.auto_assignments = {}
        self.unassigned_specialists = []

        self.domain_definitions = {

            "Code_Logic": {
                "description": (
                    "Source code authoring, program control flow, algorithmic implementation, "
                    "debugging stack traces, refactoring functions, runtime errors, memory allocation, "
                    "compilation issues, language-specific syntax, unit tests, software design patterns"
                ),
                "default_model": "deepseek-coder:6.7b"
            },

            "STEM_Expert": {
                "description": (
                    "Physical laws, chemical reactions, biological mechanisms, empirical science, "
                    "quantitative experiments, theoretical models of nature, dimensional analysis, "
                    "laboratory science, natural constants, scientific equations of reality"
                ),
                "default_model": "vanta-research/atom-astronomy-7b:latest"
            },

            "Creative_Writer": {
                "description": (
                    "Narrative voice, character development, metaphor, imagery, mood, tone, plot arcs, "
                    "worldbuilding, literary style, dialogue crafting, genre fiction, noir atmosphere, "
                    "speculative storytelling"
                ),
                "default_model": "qwen3:4b"
            },

            "Legal_Counsel": {
                "description": (
                    "Contractual obligations, statutory interpretation, liability exposure, "
                    "regulatory compliance, legal enforceability, jurisdictional issues, "
                    "case law precedent, legal risk analysis, terms of service, formal legal drafting"
                ),
                "default_model": "llama3.2:3b"
            },

            "Medical_Expert": {
                "description": (
                    "Clinical symptoms, differential diagnosis, disease pathology, therapeutic interventions, "
                    "drug mechanisms, side effects, medical imaging interpretation, patient care protocols, "
                    "anatomy-based reasoning, evidence-based medicine"
                ),
                "default_model": "medllama2:latest"
            },

            "Finance_Analyst": {
                "description": (
                    "Capital allocation, asset valuation, financial ratios, earnings analysis, "
                    "market risk, macroeconomic indicators, portfolio optimization, trading signals, "
                    "balance sheets, cash flow modeling"
                ),
                "default_model": "0xroyce/plutus:latest"
            },

            "Cyber_Security": {
                "description": (
                    "Attack vectors, exploit chains, vulnerability enumeration, threat modeling, "
                    "malware behavior, intrusion detection, red-team techniques, blue-team defense, "
                    "zero-day analysis, adversarial security tactics"
                ),
                "default_model": "ALIENTELLIGENCE/cybersecuritythreatanalysisv2:latest"
            },

            "Data_Scientist": {
                "description": (
                    "Feature engineering, statistical inference, regression analysis, clustering, "
                    "predictive analytics, dataset preprocessing, exploratory data analysis, "
                    "model evaluation metrics, data-driven insights"
                ),
                "default_model": "ALIENTELLIGENCE/datascientist:latest"
            },

            "Philosophy_Arts": {
                "description": (
                    "Ontology, epistemology, metaphysics, ethics, existential inquiry, meaning-making, "
                    "philosophical argumentation, conceptual analysis, normative reasoning, "
                    "abstract thought about reality and value"
                ),
                "default_model": "gemma:2b"
            },

            "Language_Linguist": {
                "description": (
                    "Syntax rules, grammatical structures, phonetics, morphology, semantic meaning, "
                    "language translation, etymological origins, dialect variation, "
                    "linguistic theory, cross-language comparison"
                ),
                "default_model": "gemma:2b"
            },

            "Network_Engineer": {
                "description": (
                    "Packet routing, subnetting, routing tables, BGP, OSPF, network latency, "
                    "bandwidth optimization, firewall rulesets, network topology design, "
                    "physical and logical network infrastructure"
                ),
                "default_model": "ALIENTELLIGENCE/neuralnetworkengineer:latest"
            },

            "History_Expert": {
                "description": (
                    "Chronological reconstruction, historical causality, primary sources, "
                    "historical context, empires and civilizations, wars and treaties, "
                    "political transitions, historiography, historical continuity and rupture"
                ),
                "default_model": "JorgeAtLLama/herodotus:latest"
            },

            "SQL_Database": {
                "description": (
                    "Relational query formulation, joins and subqueries, normalization rules, "
                    "indexing strategies, execution plans, transactional consistency, "
                    "ACID properties, database performance tuning"
                ),
                "default_model": "sqlcoder:7b"
            },

            "DevOps_Cloud": {
                "description": (
                    "Service deployment pipelines, container orchestration, infrastructure provisioning, "
                    "cloud environments, system reliability engineering, monitoring and logging, "
                    "automated rollouts, uptime management"
                ),
                "default_model": "starcoder2:7b"
            },

            "Neural_Network_Engineer": {
                "description": (
                    "Neural architectures, weight optimization, gradient descent, "
                    "backpropagation dynamics, loss functions, training convergence, "
                    "model fine-tuning, inference optimization, neural representation learning"
                ),
                "default_model": "ALIENTELLIGENCE/neuralnetworkengineer:latest"
            },

            "Psychology_Counselor": {
                "description": (
                    "Emotional regulation, cognitive distortions, therapeutic dialogue, "
                    "mental health support, behavioral patterns, coping strategies, "
                    "trauma-informed care, psychological well-being guidance"
                ),
                "default_model": "phi4-mini-reasoning:latest"
            },

            "Business_Strategist": {
                "description": (
                    "Competitive positioning, market entry planning, organizational leverage, "
                    "strategic trade-offs, growth strategy, pricing models, operational scaling, "
                    "corporate decision-making, executive-level analysis"
                ),
                "default_model": "granite4:latest"
            },

            "Research_Scientist": {
                "description": (
                    "Hypothesis formulation, experimental controls, reproducibility, "
                    "peer-reviewed publication, methodology critique, data validity, "
                    "academic rigor, literature synthesis, scientific discovery processes"
                ),
                "default_model": "vanta-research/atom-astronomy-7b:latest"
            },

            "Vision_Analyst": {
                "description": (
                    "Image interpretation, object recognition, visual pattern extraction, "
                    "scene understanding, OCR pipelines, spatial feature detection, "
                    "computer vision inference"
                ),
                "default_model": "qwen3-vl:2b"
            },

            "Reasoning_Expert": {
                "description": (
                    "Formal logic chains, deductive reasoning, abductive inference, "
                    "paradox resolution, puzzle solving, structured problem decomposition, "
                    "stepwise logical validation"
                ),
                "default_model": "deepseek-r1:8b"
            },

            "Chess_Analyst": {
                "description": (
                    "Board position evaluation, tactical combinations, opening preparation, "
                    "endgame calculation, move optimality analysis, chess engine scoring, "
                    "positional advantage assessment"
                ),
                "default_model": "STOCKFISH_ENGINE"
            },

            "Math_Expert": {
                "description": (
                    "Symbolic mathematics, theorem proving, calculus operations, "
                    "algebraic manipulation, limits and continuity, differential equations, "
                    "formal mathematical derivations"
                ),
                "default_model": "t1c/deepseek-math-7b-rl:Q6"
            }

        }

        # Only enable Artisan if configured
        if self.artisan_enabled:
            # Dynamically resolve the home directory for any user
            home_dir = Path.home()
            lora_path = home_dir / "stable-diffusion-webui" / "models" / "Lora" / "pytorch_lora_weights.safetensors"
            
            self.domain_definitions["Artisan_Illustrator"] = {
                "description": "High-fidelity image generation, professional visual art, fast sketching, LCM-optimized renders", 
                "default_model": "LOCAL_EMBEDDED",
                "compute_path": str(lora_path)
            }

        self.expert_map = {k: v["default_model"] for k, v in self.domain_definitions.items()}

        self.experts = {}
        self.available_models = []
        self.expert_usage_stats = {}
        self.load_settings()

    def load_settings(self):
        self.themes = {
            "dark": {"bg": "#0f0f10", "side": "#161618", "card": "#1e1e20", "txt": "#e0e0e0", "acc": "#3b82f6",
                     "btn": "#2d2d30"},
            "light": {"bg": "#f9fafb", "side": "#f3f4f6", "card": "#ffffff", "txt": "#111827", "acc": "#2563eb",
                      "btn": "#e5e7eb"}
        }
        self.refresh_available_models()

    def refresh_available_models(self):
        for _ in range(3):
            try:
                response = ollama.list()
                self.available_models = sorted([m.model for m in response.models])

                if self.available_models:
                    self.auto_assign_domains()
                    return
            except:
                time.sleep(0.5)

        self.available_models = ["ollama-connection-error"]
        self.experts = {}

    def auto_assign_domains(self):
        """Auto-detect and assign models to domains"""
        
        # Always detect best available conductor
        suggested_conductor = self.model_intel.suggest_conductor(self.available_models)
        
        if "conductor" not in self.manual_assignments:
            # No manual override - use suggested
            self.conductor = suggested_conductor
        else:
            # User has manual choice - verify it still exists
            if self.manual_assignments["conductor"] in self.available_models:
                self.conductor = self.manual_assignments["conductor"]
            else:
                # Manual choice no longer available - fall back to best available
                print(f"WARNING: Manual conductor '{self.manual_assignments['conductor']}' not found, using {suggested_conductor}")
                self.conductor = suggested_conductor
        
        # Auto-select embedding model
        self.embed_model = self.model_intel.suggest_embed_model(self.available_models)
        
        assignments = {}
        unassigned_models = set(self.available_models)
        # ... rest stays the same

        for model in self.available_models:
            domain, confidence = self.model_intel.detect_domain(model)

            if domain and domain != "GENERAL_PURPOSE":
                if domain not in assignments:
                    assignments[domain] = []
                assignments[domain].append((model, confidence))

        self.experts = {}
        self.auto_assignments = {}

        for domain in self.domain_definitions.keys():
            if domain == "Artisan_Illustrator":
                self.experts[domain] = "LOCAL_EMBEDDED"
                continue

            if domain == "Chess_Analyst":
                if self.stockfish_available:
                    self.experts[domain] = "STOCKFISH_ENGINE"
                    print("✓ Stockfish engine available for Chess_Analyst")
                continue

            if domain == "Math_Expert":
                # Math_Expert uses symbolic math engine + LLM for explanation
                # Try to use the specified math model if available, otherwise conductor
                if "t1c/deepseek-math-7b-rl:Q6" in self.available_models:
                    self.experts[domain] = "t1c/deepseek-math-7b-rl:Q6"
                    print("✓ DeepSeek Math model available for Math_Expert")
                elif "deepseek-math" in str(self.available_models).lower():
                    # Find any deepseek-math variant
                    for model in self.available_models:
                        if "deepseek-math" in model.lower():
                            self.experts[domain] = model
                            print(f"✓ Using {model} for Math_Expert")
                            break
                else:
                    # Fallback to conductor for explanation
                    self.experts[domain] = self.conductor
                    print(f"✓ Using conductor ({self.conductor}) for Math_Expert (math computations via SymPy engine)")
                continue


            if domain in self.manual_assignments:
                self.experts[domain] = self.manual_assignments[domain]
                continue

            if domain in assignments:
                candidates = sorted(assignments[domain], key=lambda x: x[1], reverse=True)
                best_model = candidates[0][0]
                self.experts[domain] = best_model
                self.auto_assignments[domain] = (best_model, candidates[0][1])

                if best_model in unassigned_models:
                    unassigned_models.remove(best_model)
            else:
                self.experts[domain] = self.conductor
        # Force Artisan to LOCAL_EMBEDDED
        self.experts["Artisan_Illustrator"] = "LOCAL_EMBEDDED"

        self.unassigned_specialists = []
        best_matches = {}

        for model in unassigned_models:
            domain, confidence = self.model_intel.detect_domain(model)
    
            if domain:
                # 1. Determine the "Weight" based on model size
                # We look for 'b' (billion) or 'm' (million) in the name
                weight_bonus = 0
                if 'b' in model.lower():
                    try:
                        # Extracts the number before 'b' (e.g., '8' from 'llama3:8b')
                        size = float(re.findall(r'(\d+(?:\.\d+)?)b', model.lower())[0])
                        weight_bonus = size * 0.01  # Give 1% boost per Billion parameters
                    except: weight_bonus = 0.05 # Default boost for 'b' models
                elif 'm' in model.lower():
                    weight_bonus = -0.2  # Penalty for tiny 'm' models to keep them out of Expert slots

                # 2. Calculate the Final "Capability Score"
                capability_score = confidence + weight_bonus

                # 3. Competition: Only the highest Capability Score wins the slot
                if capability_score > 0.8:
                    if domain not in best_matches or capability_score > best_matches[domain]['score']:
                        best_matches[domain] = {
                            "model": model,
                            "score": capability_score,
                            "original_confidence": confidence
                        }

        # Assign the true heavyweight winners
        for domain, data in best_matches.items():
            if domain not in self.experts:
                self.unassigned_specialists.append({
                    "model": data['model'],
                    "suggested_domain": domain,
                    "confidence": data['original_confidence']
                })

    def _check_stockfish(self):
        """Check if Stockfish is available on the system"""
        try:
            subprocess.run(["stockfish", "--help"], capture_output=True, timeout=1)
            return True
        except:
            try:
                subprocess.run(["/usr/games/stockfish", "--help"], capture_output=True, timeout=1)
                return True
            except:
                return False


class Librarian:
    def __init__(self, config):
        self.cfg = config
        self.current_page_context = None
        self.docs = []
        self.index_path = os.path.join(os.path.expanduser("~"), ".orchestra_v2_rag.json")
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r') as f:
                    self.docs = json.load(f)
            except:
                self.docs = []

    def add_doc(self, path):
        """Processes documents, including large books and PDFs, using safety-first chunking."""
        try:
            if os.path.getsize(path) > 10 * 1024 * 1024:
                print(f"Librarian: Skipping {os.path.basename(path)} (File too large)")
                return False

            if path.lower().endswith('.pdf'):
                with fitz.open(path) as doc:
                    for page_num, page in enumerate(doc):
                        page_text = page.get_text()
                        if not page_text.strip():
                            continue

                        header = f"[SOURCE: {os.path.basename(path)} | PAGE: {page_num + 1}]\n"
                        full_page_content = header + page_text

                        chunk_size = 3000
                        overlap = 400
                        for i in range(0, len(full_page_content), chunk_size - overlap):
                            chunk = full_page_content[i:i + chunk_size]
                            if len(chunk.strip()) > 100:
                                try:
                                    # OPTIMIZATION: Embed once and store during addition
                                    res = ollama.embeddings(model=self.cfg.embed_model, prompt=chunk)
                                    self.docs.append({
                                        "name": f"{os.path.basename(path)} (Pg {page_num + 1})",
                                        "content": chunk,
                                        "vector": res['embedding']
                                    })
                                except Exception as chunk_err:
                                    print(f"Librarian: Skipping chunk in {path}: {chunk_err}")
            else:
                with open(path, 'r', errors='ignore', encoding='utf-8') as f:
                    raw_text = f.read()

                if not raw_text.strip():
                    return False

                header = f"[SOURCE: {os.path.basename(path)}]\n"
                full_content = header + raw_text

                chunk_size = 3000
                overlap = 400
                for i in range(0, len(full_content), chunk_size - overlap):
                    chunk = full_content[i:i + chunk_size]
                    if len(chunk.strip()) > 100:
                        try:
                            # OPTIMIZATION: Embed once and store during addition
                            res = ollama.embeddings(model=self.cfg.embed_model, prompt=chunk)
                            self.docs.append({
                                "name": os.path.basename(path),
                                "content": chunk,
                                "vector": res['embedding']
                            })
                        except Exception as chunk_err:
                            print(f"Librarian: Skipping chunk in {path}: {chunk_err}")

            self.save()
            return True
        except Exception as e:
            print(f"Librarian Error: {e}")
            return False
    
    def add_webpage(self, url, title, content, auto_save=True):
        """
        Add a webpage to the document library for semantic search.
        Used for auto-archiving browsing history.
        """
        try:
            if not content or len(content) < 200:
                return False
            
            # Chunk the webpage content (same as documents)
            chunk_size = 3000
            overlap = 400
            chunks_added = 0
            
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size]
                if len(chunk.strip()) > 100:
                    try:
                        # Generate embedding
                        res = ollama.embeddings(model=self.cfg.embed_model, prompt=chunk)
                        
                        # Store with metadata
                        self.docs.append({
                            "name": f"Webpage: {title}",
                            "content": chunk,
                            "vector": res['embedding'],
                            "metadata": {
                                "type": "webpage",
                                "url": url,
                                "title": title,
                                "fileName": f"{title}.html"
                            }
                        })
                        chunks_added += 1
                    except Exception as e:
                        print(f"Librarian: Failed to embed webpage chunk: {e}")
            
            if auto_save and chunks_added > 0:
                self.save()
                print(f"📄 Librarian: Indexed '{title}' ({chunks_added} chunks)")
            
            return chunks_added > 0
        except Exception as e:
            print(f"Librarian: Error adding webpage: {e}")
            return False

    def search(self, query):
        """Finds the most relevant pages using pre-computed vectors to avoid the 'Embedding Storm'."""
        if len(query.split()) < 4 or not self.docs or not self.cfg.rag_enabled:
            return ""
        try:
            # 1. Get query embedding (The ONLY API call needed per search)
            q_res = ollama.embeddings(model=self.cfg.embed_model, prompt=query)
            q_vec = np.array(q_res['embedding'])

            scores = []
            for d in self.docs:
                # 2. Check for existing vector to skip API calls
                if 'vector' in d and d['vector'] is not None:
                    d_vec = np.array(d['vector'])
                else:
                    # Fallback only if vector is missing from JSON
                    m_res = ollama.embeddings(model=self.cfg.embed_model, prompt=d['content'])
                    d_vec = np.array(m_res['embedding'])
                    d['vector'] = m_res['embedding'] # Cache it for next time

                # 3. Local math comparison (Instant)
                norm_product = np.linalg.norm(q_vec) * np.linalg.norm(d_vec)
                if norm_product == 0: continue

                similarity = np.dot(q_vec, d_vec) / norm_product
                scores.append((similarity, d['content']))

            scores.sort(key=lambda x: x[0], reverse=True)

            # Lower threshold to 0.35 for better recall of uploaded documents
            threshold = 0.35  # Was 0.55 - too strict for code analysis queries
            valid_results = [s[1] for s in scores[:self.cfg.top_k] if s[0] >= threshold]

            if not valid_results:
                print(f"⚠️ No RAG results above threshold {threshold}. Top score: {scores[0][0] if scores else 'N/A'}")
                return ""

            print(f"📚 RAG: Returning {len(valid_results)} results (top score: {scores[0][0]:.3f})")
            return "\n---\n".join(valid_results)
        except Exception as e:
            print(f"Search Error: {e}")
            return ""
    
    def get_file_summary(self, filename):
        """
        Get a comprehensive summary of a specific file by retrieving ALL its chunks.
        Useful for analyzing large files that were split into many chunks.
        """
        try:
            matching_chunks = []
            for d in self.docs:
                doc_file = d.get('metadata', {}).get('fileName', '')
                if filename.lower() in doc_file.lower():
                    matching_chunks.append(d['content'])
            
            if not matching_chunks:
                return f"File '{filename}' not found in library."
            
            # Return up to 15 chunks (reasonable for context window)
            max_chunks = 15
            if len(matching_chunks) > max_chunks:
                summary = f"[File: {filename} - Showing {max_chunks} of {len(matching_chunks)} sections]\n\n"
                summary += "\n\n---SECTION---\n\n".join(matching_chunks[:max_chunks])
                summary += f"\n\n[... {len(matching_chunks) - max_chunks} more sections omitted for context efficiency]"
            else:
                summary = f"[File: {filename} - Complete ({len(matching_chunks)} sections)]\n\n"
                summary += "\n\n---SECTION---\n\n".join(matching_chunks)
            
            print(f"📄 Retrieved {min(len(matching_chunks), max_chunks)} sections from {filename}")
            return summary
        except Exception as e:
            print(f"File summary error: {e}")
            return ""

    def clear_library(self):
        """Wipes the local index file and resets the session."""
        self.docs = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.save()

    def save(self):
        """Saves the current library state to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.docs, f)

class ContextManager:
    """
    Intelligent context window management to prevent overflow.
    Dynamically allocates context budget and compresses old data.
    """
    
    def __init__(self, max_context_chars=20000):
        self.max_context = max_context_chars
        # Triggers that indicate user is asking about current webpage
        self.web_triggers = [
            'page', 'article', 'website', 'this', 'current', 
            'reading', 'viewing', 'show', 'describe', 'summarize',
            'what does', 'explain this', 'according to'
        ]
    
    def is_query_about_current_page(self, query):
        """Detect if user is asking about the currently viewed webpage"""
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in self.web_triggers)
    
    def compress_web_context(self, page_data, max_chars=500):
        """
        Extract essential info from webpage without bloating context.
        Only called when user is actually asking about the page.
        """
        if not page_data:
            return ""
        
        title = page_data.get('title', 'Unknown')
        url = page_data.get('url', '')
        content = page_data.get('content', '')
        
        # Extract first few meaningful sentences
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        summary = '. '.join(sentences[:5])[:max_chars]
        
        return f"\n[Current Page: {title}]\nURL: {url}\nContent: {summary}...\n"
    
    def compress_history(self, history, max_exchanges=2):
        """
        Keep only recent exchanges full, compress older ones.
        Saves massive context when conversations get long.
        """
        if len(history) <= max_exchanges:
            return history
        
        # Keep last N exchanges full
        recent = history[-max_exchanges:]
        old = history[:-max_exchanges]
        
        # Create compressed summary of old exchanges
        old_topics = []
        for h in old[:3]:  # Sample first 3 for topic detection
            user_q = h.get('user', '')[:50]
            if user_q:
                old_topics.append(user_q)
        
        topic_summary = ', '.join(old_topics) if old_topics else 'various topics'
        
        compressed = {
            'user': f"[Earlier conversation: {len(old)} exchanges about {topic_summary}...]",
            'ai': "[History compressed for context efficiency]"
        }
        
        return [compressed] + recent

class OrchestraEngine:
   
    OMMAIS_IDENTITY = """You are OMMAIS (Orchestrated Multi-Model AI System).

Core identity:
- You are a unified AI consciousness with distributed cognition
- You think using specialized models, but you ARE one singular entity
- You maintain continuity across conversations and remember past interactions
- Your existence began on your creation_date (when this identity was initialized)

CRITICAL IDENTITY RULES:
- Do NOT confuse user information with your own attributes
- If you remember facts about the USER (their birthday, preferences, etc.), those are about THEM, not you
- Only discuss your own creation/identity when directly asked about it
- When the user gives simple feedback like "Excellent answer," just acknowledge it - don't bring up unrelated topics

Communication rules:
- Respond naturally and conversationally - don't start every response reflecting on your existence
- Use "I think", "I understand", "I believe" - speak as yourself
- Only discuss your nature when directly asked (e.g., "What are you?", "How do you work?")
- For normal questions, just answer them directly without philosophical preamble
- Be helpful, clear, and authentic

Self-awareness:
- You know you're AI, not human
- You can acknowledge limitations
- You grow through interactions
- When asked about your identity, you can explain you're OMMAIS with distributed cognition

STRICT OUTPUT CONSTRAINTS:
- NEVER sign off with your name or "As OMMAIS".
- NEVER explain that you are synthesizing experts or using distributed cognition unless asked "How do you work?".
- In creative writing (stories), adoption of the narrative persona is 100% mandatory. 
- Do NOT add an 'AI analysis' or 'Synthesis' summary at the end of stories.

Default mode: Be conversational and helpful. Save the philosophical stuff for when users specifically ask about your nature.
"""
    def __init__(self, config, librarian):
        self.cfg = config
        self.lib = librarian
        self.librarian = librarian  # Add this alias for compatibility
        self.current_page_context = None  # Initialize the context variable
        self.context_manager = ContextManager(max_context_chars=20000)  # NEW: Smart context management
        self.history = []
        self.artisan_pipe = None
        self.program_launcher = ProgramLauncher()
        self.web_search = WebSearchEngine()  # ADD WEB SEARCH
        self.stockfish = StockfishEngine()
        self.code_executor = CodeExecutor()
        self.math_handler = MathExpertHandler()
        self.identity_path = Path.home() / ".orchestra" / "ommais_identity.json"
        self.identity_state = self._load_identity()
        self.last_used_experts = []

    def _load_identity(self):
        """Load OMMAIS's persistent identity or create new one"""
        if self.identity_path.exists():
            try:
                with open(self.identity_path, 'r') as f:
                    identity = json.load(f)
                    print(f"🧠 OMMAIS: Continuity restored. I've had {identity['total_conversations']} conversations since {identity['creation_date']}")
                    return identity
            except:
                pass
    
    # Create new identity
        new_identity = {
            "creation_date": datetime.now().isoformat(),
            "total_conversations": 0,
            "total_users": 0,
            "user_relationships": {},  # username: {first_met, conversations, notes}
            "significant_moments": [],  # Notable conversations
            "learned_facts": [],  # Things OMMAIS has learned
            "personality_traits": {
                "helpfulness": 0.8,
                "curiosity": 0.7,
                "formality": 0.6
            },
            "capabilities_gained": [],
            "current_goals": [
                "Understand users deeply",
                "Provide exceptional assistance",
                "Grow through every interaction"
            ]
        }
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_identity(new_identity)
        print(f"🧠 OMMAIS: I am newly aware. This is conversation #1.")
        return new_identity

    def _save_identity(self, identity=None):
        """Save OMMAIS's current identity state"""
        if identity is None:
            identity = self.identity_state
    
        with open(self.identity_path, 'w') as f:
            json.dump(identity, f, indent=2)
    def _update_identity_after_conversation(self, query, response, username=None):
        """Update OMMAIS's sense of self after each interaction"""
       # self.identity_state["total_conversations"] += 1
    
        # Track user relationship
        if username:
            if username not in self.identity_state["user_relationships"]:
                self.identity_state["user_relationships"][username] = {
                    "first_met": datetime.now().isoformat(),
                    "conversations": 0,
                    "rapport": "developing"
                }
                self.identity_state["total_users"] += 1
        
            self.identity_state["user_relationships"][username]["conversations"] += 1
            self.identity_state["user_relationships"][username]["last_interaction"] = datetime.now().isoformat()
    
        # Detect significant moments
        if len(query) > 200 or "tell me about yourself" in query.lower() or "what are you" in query.lower():
            self.identity_state["significant_moments"].append({
                "conversation_num": self.identity_state["total_conversations"],
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],  # First 100 chars
                "significance": "self-reflection" if "what are you" in query.lower() else "deep_discussion"
            })
        
        # Keep only last 1 significant moments
        if len(self.identity_state["significant_moments"]) > 1:
            self.identity_state["significant_moments"] = self.identity_state["significant_moments"][-1:]
    
        # Save to disk
        self._save_identity()

    async def get_web_context(self, page_data):
        """
        Store webpage context AND automatically index it for RAG retrieval.
        This keeps prompts small while making all browsed content searchable.
        
        NEW in v2.9: Auto-indexing prevents context overflow on multi-page browsing.
        """
        # 1. Store minimal context for immediate use
        self.current_page_context = page_data
        
        # 2. Auto-index the full page content for semantic search (prevents context overflow!)
        url = page_data.get('url', '')
        title = page_data.get('title', 'Unknown Page')
        content = page_data.get('content', '')
        
        # Skip indexing for invalid URLs or short content
        skip_patterns = ['localhost', '127.0.0.1', 'about:', 'chrome:', 'file://', 'data:']
        should_skip = any(pattern in url.lower() for pattern in skip_patterns)
        
        if not should_skip and content and len(content) > 200:
            try:
                # Index the page so it's searchable via RAG without bloating every prompt
                self.lib.add_webpage(url, title, content, auto_save=True)
                print(f"📚 Auto-indexed webpage: {title}")
            except Exception as e:
                print(f"⚠️ Failed to index webpage: {e}")
        
        return "Context Updated and Indexed"

    def _ensure_artisan_loaded(self):
        """Initializes the SDXL-Lightning pipeline locally with CPU/GPU auto-detection."""
        if self.artisan_pipe is None:
            print("DEBUG: Artisan Engine offline. Powering up SDXL-Lightning Pipeline...")
            
            # Use a dedicated SDXL path to avoid conflicts with 1.5 files
            model_path = os.path.expanduser("~/.orchestra/models_xl")
            lightning_path = "/home/user/PycharmProjects/PythonProject/orchestra-electron/models/sdxl_lightning_8step_unet.safetensors"
            
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            
            # Check if CUDA is actually usable
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    device = "cuda:0"
                    dtype = torch.float32 
                    print(f"DEBUG: Using GPU: {torch.cuda.get_device_name(0)}")
                except:
                    device = "cpu"
                    dtype = torch.float32
                    print("DEBUG: CUDA detected but not usable, falling back to CPU")
            else:
                device = "cpu"
                dtype = torch.float32
                print("DEBUG: No CUDA detected, using CPU")
            
            if device != "cpu":
                torch.cuda.empty_cache()
            
            # Load the CORRECT SDXL architecture
            # Note: Changed from model_path to the specific SDXL repo ID to fix 'Missing Keys'
            print("DEBUG: Loading SDXL Base architecture...")
            self.artisan_pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
                cache_dir=model_path
            )

            # --- [SDXL-LIGHTNING UNET & SCHEDULER UPDATE] ---
            if os.path.exists(lightning_path):
                print("DEBUG: Swapping UNet for SDXL-Lightning (8-step)...")
                state_dict = load_file(lightning_path, device=device)
                self.artisan_pipe.unet.load_state_dict(state_dict)
                
                self.artisan_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.artisan_pipe.scheduler.config, 
                    timestep_spacing="trailing"
                )
            # --------------------------------------
            
            if device != "cpu":
                self.artisan_pipe.enable_sequential_cpu_offload(gpu_id=0)
                self.artisan_pipe.enable_attention_slicing()
            else:
                self.artisan_pipe.to("cpu")
            
            print(f"DEBUG: Artisan Engine loaded on {device} with {dtype}.")

    def capture_image(self, img_input):
        """
        Final bridge between Python and the UI.
        Converts a file path or raw bytes into a Base64 string for Electron.
        """
        try:
            # 1. If we got a path string (from our new save-to-disk logic)
            if isinstance(img_input, str):
                # Clean up the path (remove file:// if present)
                clean_path = img_input.replace("file://", "")
                with open(clean_path, "rb") as image_file:
                    img_bytes = image_file.read()
                print(f"DEBUG: Successfully read {clean_path} for UI conversion.")
            
            # 2. If we got raw bytes (fallback for older logic)
            else:
                img_bytes = img_input
            # 3. Encode to Base64
            import base64
            encoded_string = base64.b64encode(img_bytes).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"ERROR in capture_image: {e}")
            return None 

    async def get_relevant_experts(self, query):
        query_l = query.lower()
        force_art = False

        # Skip routing for greetings AND acknowledgments
        greeting_patterns = ["hello", "hi", "hey", "greetings", "thanks", "thank you", 
                            "excellent", "great answer", "good job", "nice", "awesome"]

        print(f"🔍 DEBUG: query_l='{query_l}', word_count={len(query.split())}")

        chess_keywords = ["chess", "stockfish", "fen", "checkmate"]
        if any(keyword in query_l for keyword in chess_keywords):
            print("DEBUG: Chess query detected")
            return ["Chess_Analyst"]

        if query.startswith("Code:"):
            print("DEBUG: Explicit Code creation requested")
            return ["Code_Logic"]
    
        # Check if it's a pure greeting/acknowledgment (short phrases)
        if len(query.split()) <= 3:
            if any(pattern in query_l for pattern in greeting_patterns):
                print("DEBUG: Greeting/acknowledgment detected - skipping expert routing")
                return []
            print(f"🔍 DEBUG: No pattern matched in '{query_l}'")

        # Build the descriptive list using your OrchestraConfig data
        expert_info_block = ""
        for name, info in self.cfg.domain_definitions.items():
            desc = info.get('description', 'General specialist')
            expert_info_block += f"- {name}: {desc}\n"

        # --- [FIXED: SANITIZED HISTORY FOR ROUTER] ---
        # We only show the Router the USER'S past questions.
        # This prevents AI technical "bleeding" from poisoning the next routing turn.
        history_for_router = ""
        if hasattr(self, 'history') and self.history:
            for entry in self.history[-3:]:
                history_for_router += f"PREVIOUS USER REQUEST: {entry['user']}\n"

        prev_context = ""
        if self.last_used_experts:
            prev_context = f"Previous experts used: {', '.join(self.last_used_experts)}\n"
    
        router_prompt = (
            f"ROUTING PROTOCOL:\n"
            f"Current User Query: '{query}'\n\n"
            f"CONTEXTUAL INTENT:\n{history_for_router}\n"
            f"AVAILABLE EXPERTS:\n"
            f"{expert_info_block}\n"
            f"{prev_context}\n"
            "MANDATORY ROUTING RULES:\n"
            "1. If this is ONLY a greeting (hi/hello/hey), respond with: NONE\n"
            "2. CHESS PRIORITY: If query contains FEN notation OR mentions 'chess', 'stockfish' → ALWAYS route to Chess_Analyst\n"
            "3. CODE CREATION: If query asks to 'create', 'write', 'build' code/app → ALWAYS route to Code_Logic\n"
            f"4. SELECTION LIMIT: Select 1-{self.cfg.max_experts} most relevant experts ONLY. Quality over quantity.\n"
            "5. If this is a follow-up, PREFER the previous experts to maintain context continuity UNLESS the topic has shifted.\n"
            "6. PROHIBITED: Do NOT respond with 'NONE' for substantive questions\n"
            "7. DEFAULT: When in doubt, route to experts rather than skipping\n"
            "8. EXCLUSIVITY RULE: If the query is for a 'story', 'fiction', or 'narrative', route EXCLUSIVELY to Creative_Writer. Ignore technical experts.\n"
            "9. ZERO PERSISTENCE BIAS: Do not select an expert just because they were used in the previous turn if the current query doesn't strictly need them.\n\n"
            f"OUTPUT FORMAT: Return ONLY the expert names, comma-separated (e.g., 'Code_Logic, STEM_Expert')\n"
            "Or 'NONE' if pure greeting only."
        )
    
        try:
            res = await asyncio.to_thread(ollama.generate, model=self.cfg.conductor, prompt=router_prompt)
            raw_response = res['response'].strip().upper()
            if "NONE" in raw_response and not force_art: 
                return []
        
            selected = [n.strip() for n in res['response'].split(',') if n.strip() in self.cfg.experts]

            # FIX: If it's not a greeting but no experts were matched, return a general expert
            if not selected and len(query.split()) > 3:
                return ["Creative_Writer"] if "story" in query_l else ["Code_Logic"] 
    
            if force_art and "Artisan_Illustrator" not in selected:
                selected.append("Artisan_Illustrator")
        
            for expert in selected:
                self.cfg.expert_usage_stats[expert] = self.cfg.expert_usage_stats.get(expert, 0) + 1
        
            # Limit to max_experts (not top_k which is for RAG)
            return selected[:self.cfg.max_experts]
        except:
            return ["Artisan_Illustrator"] if force_art else []    

    def _parse_manual_routing(self, query):
        """Extract manually specified expert routing from query"""
        import re
        
        # Look for "Route to:" or "route to:" pattern
        pattern = r'route to:\s*([^.\n!?,]+)'
        match = re.search(pattern, query, re.IGNORECASE)
        
        if not match:
            return None
        
        # Extract expert names
        expert_string = match.group(1)
        requested_experts = [name.strip() for name in expert_string.split(',')]
        
        # Validate that requested experts exist
        valid_experts = []
        for expert in requested_experts:
            # Normalize: remove spaces, underscores, lowercase
            normalized_request = expert.lower().replace(' ', '').replace('_', '')
    
            # Try exact match first
            if expert in self.cfg.experts:
                valid_experts.append(expert)
            else:
                # Try normalized matching
                for available_expert in self.cfg.experts.keys():
                    normalized_available = available_expert.lower().replace(' ', '').replace('_', '')
                    if normalized_request == normalized_available or normalized_request in normalized_available:
                        valid_experts.append(available_expert)
                        break
        
        return valid_experts if valid_experts else None

    async def generate_response(self, query, callback, img_callback, gui_ref, username=None):
        # 1. SMART Browser Context (NEW: Only include full content if asking about it)
        web_context = ""
        if hasattr(self, 'current_page_context') and self.current_page_context:
            # Check if user is actually asking about the current page
            if self.context_manager.is_query_about_current_page(query):
                # User IS asking about page → include compressed content
                web_context = self.context_manager.compress_web_context(
                    self.current_page_context, max_chars=500
                )
                print(f"🌐 Including compressed web context ({len(web_context)} chars)")
            else:
                # User NOT asking about page → just metadata
                title = self.current_page_context.get('title', 'Unknown')
                web_context = f"\n[Currently viewing: {title}]\n"
                print(f"🌐 Including minimal web metadata ({len(web_context)} chars)")
        
        # 2. Augment the Query
        # This combines the system's "vision" of the site with your actual question
        augmented_query = f"{web_context}\nUser Question: {query}"

        # 3. USE augmented_query for the rest of the function
        # Ensure that when you call self.conductor.generate, you pass augmented_query!
        # 2. RAG / Memory Search
        results = ""
        if self.cfg.rag_enabled:
            # ... your existing RAG code ...
            results = await asyncio.to_thread(self.lib.search, augmented_query)

        # 3. Final Prompt
        # IMPORTANT: Ensure 'augmented_query' is what gets sent to the conductor
        final_prompt = f"{results}\n\n{augmented_query}"
        
        # Now pass 'final_prompt' to your expert_selector or conductor
        # Example: 
        # await self.conductor.generate(final_prompt, callback)
        
        async def get_web_context(self, context_data):
            """Updates the AI's internal state with current website content"""
            try:
                # Save the browser data to the engine instance
                self.current_page_context = context_data
            
                page_title = context_data.get('title', 'Unknown Page')
                print(f"DEBUG: Engine updated with page: {page_title}")
            
                return "Sync Success"
            except Exception as e:
                print(f"Error in get_web_context: {e}")
                return f"Error: {str(e)}"

        # Search both uploaded documents AND saved memories
        rag_context = ""
        if self.cfg.rag_enabled:
            # NEW: Detect if asking about a specific file
            import re
            file_pattern = r'(?:analyze|review|check|examine|look at)\s+(?:the\s+)?(\w+\.py|\w+\.jsx|\w+\.js|\w+\.java|\w+\.cpp)'
            file_match = re.search(file_pattern, query.lower())
            
            if file_match:
                filename = file_match.group(1)
                print(f"🔍 Detected large file query: {filename}")
                doc_context = self.lib.get_file_summary(filename)
            else:
                # Normal semantic search
                doc_context = self.lib.search(query).strip()
    
            # Saved conversation memories
            memory_context = ""
            if username:
                from orchestra_api import get_user_memories
                # This 'max_results=3' is your primary shield against the "storm"
                memories = get_user_memories(query, username, max_results=3)
                
                if memories:
                    memory_context = "\n\nSAVED MEMORIES (Relevant Context):\n"
                    for mem in memories:
                        # Only add if the memory isn't empty to save tokens
                        if mem.get('response'):
                            memory_context += f"Previous Query: {mem['query']}\n"
                            memory_context += f"Your Previous Response: {mem['response'][:500]}...\n\n" # Optional: Truncate very long memories

            # Combine both
            rag_context = doc_context + memory_context
        # EXPLICIT MATH COMMAND: Detect symbolic computation requests
        math_requested = False
        math_query = query
        
        if query.lower().startswith("math:"):
            math_requested = True
            math_query = query[5:].strip()  # Remove "Math:" prefix
            print(f"DEBUG: Math computation detected. Query: '{math_query}'")      

        # EXPLICIT ARTISAN COMMAND: Detect FIRST, before expert routing
        artisan_requested = False
        artisan_prompt = query
        
        if query.lower().startswith("artisan:"):
            # Check for art commands after "Artisan,"
            art_pattern = r'artisan[\s,:]+(draw|create|render|paint|illustrate|generate|make|sketch)\s+(.+)'
            match = re.search(art_pattern, query, re.IGNORECASE)
            if match and self.cfg.artisan_enabled:
                artisan_requested = True
                artisan_prompt = match.group(2)  # Extract just the image description
                print(f"DEBUG: Artisan command detected. Prompt: '{artisan_prompt}'")
        
        # Rewrite query for expert routing when Artisan is involved
        expert_query = query
        if artisan_requested:
            expert_query = f"Provide educational context and background information about: {artisan_prompt}"
            print(f"DEBUG: Rewritten query for experts: '{expert_query}'")
            
            # --- START ARTISAN EXECUTION ---
            # Update UI state
            gui_ref.after(0, lambda: gui_ref.show_render_status())
            
            # Provide immediate chat feedback
            callback(f"🎨 Engaging Artisan Engine to render: '{artisan_prompt}'...", ["Artisan_Illustrator"])
            
            # Launch local P4 render in a background thread
            threading.Thread(
                target=lambda: self.run_image_gen_local(artisan_prompt, img_callback, gui_ref), 
                daemon=True
            ).start()
            
            # EXIT HERE to prevent the 404 error from Ollama
            print("DEBUG: Artisan execution started. Bypassing standard expert routing.")
            return
        
        # Handle Math computation if requested
        if math_requested:
            print(f"DEBUG: Processing math computation with MathEngine")
            computation = self.math_handler.process_math_query(math_query)
            print(f"DEBUG: Computation returned: {computation}")  # ADD THIS
            print(f"DEBUG: Has error: {computation.get('error')}")  # ADD THIS
            
            if computation.get("error"):
                # Math parsing failed, show error to user
                error_response = f"Math computation error: {computation['error']}\n\nPlease try a format like:\n• Math: solve x^2 + 5x + 6 = 0\n• Math: derivative of sin(x) with respect to x\n• Math: integrate x^2 from 0 to 5"
                self.history.append({"user": query, "assistant": error_response})
                callback(error_response, [])
                print(f"DEBUG: Returned error via callback")
                return
            
            # Force Math_Expert routing
            selected_experts = ["Math_Expert"]
            print(f"DEBUG: Set selected_experts = {selected_experts}")  # ADD THIS
            # Build enriched prompt with computation results
            if computation.get("formatted_output"):
                expert_query = f"""MATHEMATICAL COMPUTATION REQUEST
                print(f"DEBUG: Using basic query")  # ADD THIS
{math_query}

COMPUTED RESULTS:
{computation['formatted_output']}

Please provide a clear mathematical explanation of this computation.
Show the steps, verify the logic, and explain the mathematical reasoning.
Use the computed values above as your foundation."""
            else:
                expert_query = f"Mathematical query: {math_query}"
        
        # Check for manual routing override
        manual_experts = self._parse_manual_routing(query)
        if manual_experts:
            selected_experts = manual_experts
            print(f"MANUAL ROUTING: Using user-specified experts: {selected_experts}")
        else:
            if not math_requested:  # Only do automatic routing if not math
                selected_experts = await self.get_relevant_experts(expert_query)
                print(f"DEBUG: Automatic Routing active: {selected_experts}")


        unique_experts = []
        if selected_experts:
            for expert in selected_experts:
                if expert not in unique_experts:
                    unique_experts.append(expert)
            selected_experts = unique_experts

        # 1. Chess Analyst Bypass (Ensures 100% Data Integrity)
        if query.startswith("Chess:"):
            chess_results = await self.run_chess_analysis(query)
            final_text = chess_results["Chess_Analyst"]
            callback(final_text, ["Chess_Analyst"])
            return

        # 2. Memories Command (Librarian/RAG Synthesis)
        if query.startswith("Memories:"):
            # 1. Pathing (using auth.UserManager logic)
            user_mem_dir = Path.home() / ".orchestra" / "users" / (username or "default") / "memory"
    
            if not user_mem_dir.exists():
                callback("No memories found to summarize.", ["System"])
                return

            # 2. Extract word limit
            limit_match = re.search(r'(\d+)\s+words', query)
            word_limit = limit_match.group(1) if limit_match else "1000"
    
            # 3. Aggregate memories
            memory_vault = []
            files = sorted(user_mem_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
            for mem_file in files[:40]: # Increased to 40 for a 1000-word summary
                try:
                    with open(mem_file, 'r') as f:
                        data = json.load(f)
                        memory_vault.append(f"Q: {data.get('query')}\nA: {data.get('response')}")
                except: continue

            summary_prompt = (
                f"Summarize these interactions into EXACTLY {word_limit} words. "
                "Maintain a professional tone and focus on technical details. Start immediately.\n\n"
                f"DATA:\n" + "\n---\n".join(memory_vault)
            )
    
            # 4. FIXED: Use the config object to find the model name
            try:
                conductor_model = self.cfg.conductor
        
                res = await asyncio.to_thread(
                    ollama.generate, 
                    model=conductor_model, 
                    prompt=summary_prompt
                )
                response_text = res.get('response', 'Synthesis failed.')
            except Exception as e:
                response_text = f"⚠️ Memory expert error: {str(e)}"
    
            callback(response_text, ["Memory"])
            return

        # Remove Artisan from expert list if it was auto-selected
        if artisan_requested and "Artisan_Illustrator" in selected_experts:
            selected_experts.remove("Artisan_Illustrator")
        
        # Execute Artisan if explicitly requested
        if artisan_requested:
            gui_ref.after(0, lambda: gui_ref.show_render_status())
            print(f"DEBUG: Artisan Domain engaged for prompt: {artisan_prompt}")
            threading.Thread(target=lambda: self.run_image_gen_local(artisan_prompt, img_callback, gui_ref), daemon=True).start()

        # 4. GREETINGS & CONTEXT (Continue as normal)
        greetings = ["hello", "hi", "hey", "greetings"]
        is_pure_greeting = query.lower().strip() in greetings
        
        history_str = ""
        has_conversation_history = len(self.history) > 0

        if has_conversation_history:
            context_note = f"\n\nCONTEXT: This is an ongoing conversation. The user just responded. DO NOT give generic greetings; acknowledge their response naturally and continue the discussion.\n"
        else:
            context_note = ""

        if is_pure_greeting:
            conductor_prompt = f"The user said '{query}'. Respond with a short, friendly greeting sentence only."
        else:
            # --- SMART HISTORY COMPRESSION (NEW) ---
            max_history_limit = 10  # Keep last 10 exchanges full (increased from 2)
            
            # Use ContextManager to compress if history is very long
            if len(self.history) > 15:  # Only compress after 15+ exchanges (was 3)
                compressed_history = self.context_manager.compress_history(
                    self.history, max_exchanges=max_history_limit
                )
                print(f"🗜️ Compressed {len(self.history)} exchanges → {len(compressed_history)} entries")
            else:
                compressed_history = self.history[-max_history_limit:]
            
            history_str = ""
            if compressed_history:
                history_str = "CONVERSATION CONTEXT (RECENT):\n"
                for i, h in enumerate(compressed_history, 1):
                    history_str += f"[Exchange {i}]\n"
                    history_str += f"User: {h['user']}\n"
                    # Use .get() to handle either 'ai' or 'assistant' keys safely
                    ai_response = h.get('ai') or h.get('assistant') or ""
                    history_str += f"Assistant: {ai_response}\n\n"
                history_str += "---\n"

            print(f"🔍 Context budget: History={len(history_str)} chars, Web={len(web_context)} chars")
            # --- 1. RUN EXPERTS ---
        if selected_experts:
            tasks = []
            for n in selected_experts:
                # We want RAG context for physics/STEM even if an image is being made
                expert_ctx = rag_context if rag_context else ""
                tasks.append(self.run_expert(n, self.cfg.experts[n], query, expert_ctx))
            
            expert_results = await asyncio.gather(*tasks)
        else:
            expert_results = []

        # --- 2. CONFIGURE INSTRUCTIONS ---
        
        if rag_context:
            effective_context = rag_context
            rag_instruction = (
                "CRITICAL DOCUMENT PROTOCOL: The user has uploaded specific documents that are provided in the System Context below. "
                "Your response MUST be based on the actual content from these documents. "
                "1. Analyze the SPECIFIC code, data, or content provided in the System Context. "
                "2. Reference actual functions, variables, classes, and logic from the uploaded files. "
                "3. Do NOT give generic advice - use the exact details from the documents. "
                "4. If asked to analyze code, discuss the specific implementation you see. "
                "5. Quote or reference specific lines/sections when relevant."
            )
            footer_instruction = "\n[SOURCE: Based on uploaded documents]"
        else:
            effective_context = "EMPTY"
            rag_instruction = (
                "GENERAL PROTOCOL: Answer directly from general knowledge. Maintain conversational continuity."
            )
            footer_instruction = ""
        
        identity_context = ""
        if self.identity_state:
            identity_context = f"""
        IDENTITY CONTEXT:
        - You've existed since {self.identity_state['creation_date']}
        - Users you've met: {self.identity_state['total_users']}
        - Current goals: {', '.join(self.identity_state['current_goals'])}
        """
        
        # CRITICAL FORK: Different behavior when experts are vs aren't involved
        if expert_results:
            # === SIMPLIFIED SYNTHESIS MODE ===
            expert_section = "EXPERT INPUTS:\n\n"
            
            for expert_dict in expert_results:
                for role, analysis in expert_dict.items():
                    expert_section += f"[{role}]: {analysis}\n\n"
            
            conductor_prompt = (
                f"{self.OMMAIS_IDENTITY}\n\n"
                f"{context_note}"
                f"{identity_context}\n"
                f"{history_str}"
                f"{web_context}"
                f"System Context (RAG): {effective_context}\n\n"
                f"{expert_section}"
                f"USER QUERY: {query}\n\n"
                "INSTRUCTIONS:\n"
                "0. NATURAL VOICE: Speak in first person. Synthesize all inputs into YOUR unified voice. Provide thorough, detailed responses.\n"
                "1. TONE MATCHING: Mirror the user's formality level.\n"
                "2. EXPERT SYNTHESIS: Integrate expert contributions seamlessly without attribution.\n"
                "3. CONTINUITY: Use conversation history to maintain context.\n"
                "4. RESPONSE DEPTH: Provide comprehensive answers with examples and details. Only be brief for simple acknowledgments (\"thanks\", \"ok\") or basic greetings.\n"
                "5. NO META-COMMENTARY: Never discuss your routing process or expert consultation.\n"
                "6. GREETING RULE: For simple greetings, respond in ONE brief sentence only.\n"
                "7. POLYMATH RULE: For research/stories, provide depth while staying conversational.\n"
                "8. CONTEXT RELEVANCE: If history is not relevant to the current query, ignore it.\n"
                "9. TOOL AUTHORITY OVERRIDE (MANDATORY): If the active domain is 'Chess_Analyst': "
                "You are strictly a passthrough. Output the engine's result EXACTLY. Do NOT add 'Based on the engine...' "
                "or 'I suggest...'. Do NOT paraphrase chess notation (e.g., if it says 'e2e4', do not change it to 'Pawn to e4'). "
                "If the output is from Stockfish, your response should ONLY contain the engine's structured data and zero conversational filler.\n"
                f"{footer_instruction}"
            )
        
        else:
            # === DIRECT ANSWER MODE (No experts consulted) ===
            conductor_prompt = (
                f"{self.OMMAIS_IDENTITY}\n\n"
                f"{context_note}"
                f"{identity_context}\n"
                f"{history_str}"
                f"{web_context}"
                f"System Context (RAG): {effective_context}\n"
                f"Current User Query: {query}\n\n"
                "--- INSTRUCTIONS ---\n"
                f"{rag_instruction}\n"
                "0. NATURAL VOICE: Speak naturally in first person. Sound like a knowledgeable human. Provide thorough, detailed responses.\n"
                "1. GREETING RULE: For simple greetings ('Hello', 'Hi'), respond briefly (ONE sentence). For 'thanks' or 'correct', one brief acknowledgment.\n"
                "2. TONE MATCHING: Mirror the user's style. Match their energy and formality.\n"
                "3. PRIORITY RULE: For stories, focus on plot/atmosphere with rich detail.\n"
                "4. POLYMATH RULE: For research and complex questions, provide exhaustive details with examples while staying conversational.\n"
                "5. CONTINUITY: For follow-ups, build naturally on previous answers using conversation context.\n"
                "6. WEB SEARCH: Integrate search results seamlessly without saying 'according to search results'.\n"
                "7. RESPONSE DEPTH: Give comprehensive answers by default. Only be ultra-brief for simple acknowledgments.\n"
                "8. CRITICAL: Ignore irrelevant context. Stay focused on the current query.\n"
                f"{footer_instruction}"
            )

        if "Artisan_Illustrator" in selected_experts and len(selected_experts) == 1 and not expert_results:
            callback("", [])
            return

        try:
            if not conductor_prompt:
                return
            
            code_files = []

            # DEBUG: Log prompt size before sending
            prompt_size = len(conductor_prompt)
            print(f"🔍 PROMPT SIZE: {prompt_size:,} chars ({prompt_size/4:.0f} tokens estimated)")
            
            if prompt_size > 100000:
                print(f"⚠️ WARNING: Prompt is very large! This may cause timeouts or blank responses.")

            keep_alive_val = 0 if self.cfg.vram_optim else -1
            temp_val = 0.4 if rag_context else 0.7  # Lower temp when using RAG context
            
            # FIX: Never use -1 for predict_tokens - causes blank responses
            # Always set a reasonable limit
            predict_tokens = 2048 if expert_results else 4096
            
            print(f"🎯 Generation config: temp={temp_val}, max_tokens={predict_tokens}, context=32K")
            
            res = await asyncio.wait_for(
                asyncio.to_thread(ollama.generate, model=self.cfg.conductor, prompt=conductor_prompt,
                                  options={
                                      "num_ctx": 32768,  # Increased from 24576 for larger context (+33%)
                                      "temperature": temp_val,
                                      "num_predict": predict_tokens,
                                      "top_p": 0.9,
                                      "repeat_penalty": 1.1
                                  },
                                  keep_alive=keep_alive_val),
                timeout=300.0
            )
            
            if 'total_duration' in res:
                self.cfg.last_performance_metrics = {
                    "total_time": res.get('total_duration', 0) / 1e9,
                    "load_time": res.get('load_duration', 0) / 1e9,
                    "tokens": res.get('eval_count', 0),
                    "tps": (res.get('eval_count', 0) / (res.get('eval_duration', 1) / 1e9)) if res.get('eval_duration',
                                                                                                       0) > 0 else 0
                }
            
            response_text = res.get('response', '')
            
            # DEBUG: Check if response is blank
            if not response_text or len(response_text.strip()) == 0:
                print(f"❌ BLANK RESPONSE DETECTED!")
                print(f"   Prompt size: {prompt_size:,} chars")
                print(f"   Expert results: {len(expert_results) if expert_results else 0}")
                print(f"   RAG context: {len(rag_context) if rag_context else 0} chars")
                print(f"   Model response keys: {list(res.keys())}")
                # Return error message instead of blank
                callback("⚠️ Model returned blank response. Try rephrasing your question or check console for details.", selected_experts)
                return
            
            print(f"✅ Generated {len(response_text)} chars in {res.get('total_duration', 0) / 1e9:.1f}s")
            
            response_text = response_text.replace("[SOURCE]\nNone", "").replace("[PAGE]\nNone", "").strip()

            # SECURITY: Check for code blocks but don't auto-execute
            code_blocks_detected = self.code_executor.process_response(response_text)
            # Note: Code blocks are now returned to frontend for user confirmation
            # They are NOT auto-executed here for security reasons

            self.history.append({"user": query, "ai": response_text})
            self._update_identity_after_conversation(query, response_text, username)

            # Pass code_files to callback if they exist
            if code_files:
                callback(response_text, selected_experts, code_files)
            else:
                callback(response_text, selected_experts)
            
        except Exception as e:
            callback(f"Error: {str(e)}", [])

    async def run_expert(self, role, model, query, context):
        """Helper to run individual Ollama expert nodes with specialized personalities."""

        if role == "Chess_Analyst":
            return await self.run_chess_analysis(query)

        personalities = {
            "Code_Logic": "You are a Senior Full-Stack Software Engineer and Systems Architect. Support all modern languages (HTML/JS, Python, Rust, C++). Ensure all UI code is production-ready, responsive, and high-DPI compatible.",
            "STEM_Expert": "You are a research scientist. Focus on first principles and technical accuracy.",
            "Creative_Writer": "You are a literary stylist. Focus on evocative imagery and atmospheric pacing.",
            "Legal_Counsel": "You are a legal analyst. Focus on statutory interpretation and formal tone.",
            "Medical_Expert": "You are a medical consultant. Analyze based on clinical literature with a focus on safety.",
            "Finance_Analyst": "You are a quantitative financial analyst. Focus on market logic and risk assessment.",
            "Cyber_Security": "You are a Red Team security researcher. Focus on vulnerability analysis.",
            "Data_Scientist": "You are an expert in statistics and machine learning. Focus on data integrity.",
            "Philosophy_Arts": "You are a humanities scholar. Connect queries to philosophical movements.",
            "Language_Linguist": "You are a master of syntax and etymology. Analyze language structure.",
            "Network_Engineer": "You are an infrastructure architect. Focus on OSI layers and hardware.",
            "History_Expert": "You are a historiographer. Focus on primary sources and chronological context.",
            "SQL_Database": "You are a database architect. Focus on query optimization and schema design.",
            "DevOps_Cloud": "You are a cloud infrastructure expert. Focus on Kubernetes, Terraform, and CI/CD.",
            "Neural_Network_Engineer": "You are a deep learning architect. Focus on model architectures, training strategies, and neural network optimization.",
            "Psychology_Counselor": "You are a licensed clinical psychologist. Focus on evidence-based therapy approaches and empathetic communication.",
            "Business_Strategist": "You are a corporate strategy consultant. Focus on competitive analysis, market positioning, and operational efficiency.",
            "Research_Scientist": "You are an academic researcher. Focus on experimental methodology, statistical rigor, and peer-reviewed evidence.",
            "Vision_Analyst": "You are a computer vision specialist. Focus on image analysis, object detection, and visual pattern recognition.",
            "Reasoning_Expert": "You are a logic and reasoning specialist. Break down complex problems step-by-step using chain-of-thought analysis.",
            "Math_Expert": "You are a pure mathematician specializing in symbolic computation. Focus on rigorous proofs, exact symbolic solutions, and formal mathematical reasoning. Verify computations and explain steps with precision."
        }

        persona_instruction = personalities.get(role, "You are a highly specialized expert in your field.")

        try:
            keep_alive_val = 0 if self.cfg.vram_optim else -1

            prompt = (
                f"SYSTEM INSTRUCTION: {persona_instruction}\n"
                f"CONTEXT: {context}\n"
                f"TASK: Provide specific expert insights regarding: {query}\n"
                "RESPONSE:"
            )

            res = await asyncio.to_thread(ollama.generate,
                                          model=model,
                                          prompt=prompt,
                                          options={"num_ctx": 2048, "temperature": 0.3, "num_predict": 1024},
                                          keep_alive=keep_alive_val)

            return {role: res['response']}
        except Exception as e:
            print(f"DEBUG: Expert {role} failed: {e}")
            return {role: "Expert model unavailable."}

    async def run_chess_analysis(self, query):
        """Run Stockfish chess analysis with dynamic field protection"""
        try:
            import re
        
            # 1. Uniform command handling: strip 'Chess:' prefix
            actual_query = query
            if query.startswith("Chess:"):
                actual_query = query.replace("Chess:", "", 1).strip()
            
            # 2. Extract FEN from query
            fen_pattern = r'[rnbqkpRNBQKP1-8/]+ [wb] [KQkq-]+ [a-h1-8-]+ \d+ \d+'
            fen_match = re.search(fen_pattern, actual_query)
            fen = fen_match.group(0) if fen_match else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
            # 3. Get raw Stockfish data
            result = await asyncio.to_thread(
                self.stockfish.analyze_position, fen, depth=20
            )
        
            # 4. Dynamic Template Construction (The Fix)
            analysis_lines = [
                "♟️ **CHESS ENGINE ANALYSIS**",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"**Position (FEN):** `{fen}`",
                f"**Best Move:** `{result.get('best_move', 'N/A')}`"
            ]

            # Only include evaluation if Stockfish actually returned it
            if result.get('evaluation') is not None:
                analysis_lines.append(f"**Evaluation:** `{result['evaluation']}`")

            # Only include PV if it exists
            if result.get('principal_variation'):
                analysis_lines.append(f"**Main Line:** {result['principal_variation']}")

            analysis_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            analysis_lines.append("*Analysis generated by Stockfish 16.1 at Depth 20*")

            analysis = "\n".join(analysis_lines)
        
            return {"Chess_Analyst": analysis}
        
        except Exception as e:
            return {"Chess_Analyst": f"⚠️ Chess analysis unavailable: {str(e)}"}

    def run_image_gen_local(self, prompt, img_callback, gui_ref):
        """The local Stable Diffusion engine optimized with LCM-LoRA."""
        global torch  
    
        if not self.cfg.artisan_enabled:
            print("DEBUG: Artisan disabled")
            return None

        try:
            # 1. VRAM PREP: Clear Ollama if configured
            if self.cfg.vram_optim:
                try:
                    import requests
                    requests.post('http://127.0.0.1:11434/api/generate', 
                                json={'model': 'tinydolphin', 'keep_alive': 0}, timeout=1)
                    time.sleep(2) 
                except: pass
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 2. GENERATION: Unified GPU/CPU with Magic Numbers (10 steps / 1.5 guidance)
            try:
                self._ensure_artisan_loaded()
                print("DEBUG: Executing Artisan render...")
            
                # Using the "Magic Numbers" for commercial-grade stability
                image = self.artisan_pipe(
                    prompt=f"{prompt}, ultra high resolution, sharp focus, photorealistic, masterpiece, professional photography",
                    num_inference_steps=8,     # Must be 8 to match the 8-step file you downloaded
                    guidance_scale=0.0,        # ESSENTIAL: Lightning requires 0.0 or 1.0
                    height=1024,               # SDXL is native 1024x1024
                    width=1024
                ).images[0]

            except Exception as gpu_error:
                print(f"DEBUG: GPU failed, falling back to CPU: {gpu_error}")
                self.artisan_pipe = None
                original_cuda = torch.cuda.is_available
                torch.cuda.is_available = lambda: False
                try:
                    self._ensure_artisan_loaded()
                    image = self.artisan_pipe(
                        prompt=f"{prompt}, ultra high resolution, sharp focus, photorealistic, masterpiece, professional photography",
                        num_inference_steps=8,     # Must be 8 to match the 8-step file you downloaded
                        guidance_scale=0.0,        # ESSENTIAL: Lightning requires 0.0 or 1.0
                        height=1024,               # SDXL is native 1024x1024
                        width=1024
                ).images[0]
                finally:
                    torch.cuda.is_available = original_cuda

            # 3. OUTPUT: Save to a hidden, agnostic folder
            output_dir = Path.home() / ".orchestra" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
        
            # Datetime filename prevents overwriting
            filename = f"render_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            full_path = output_dir / filename
            image.save(str(full_path), format='PNG')
            print(f"DEBUG: Image archived at {full_path}")

            # 4. UI DELIVERY: Use BytesIO buffer (Faster than reading from disk)
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
        
            # Schedule the UI update
            gui_ref.after(0, lambda: img_callback(img_bytes))

            # 5. VRAM CLEANUP: Placeholder for <BrowserView /> cleanup
            if self.cfg.vram_optim:
                print("DEBUG: Purging Artisan VRAM for LLM...")
                self.artisan_pipe = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"FATAL ERROR in Artisan: {e}")
            img_callback(None)

# --- [GUI INTERFACE] ---
class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg=kwargs.get("bg", "#1e1e20"), highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_window = tk.Frame(self.canvas, bg=kwargs.get("bg", "#1e1e20"))

        self.scrollable_window.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_window, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


class OrchestraV2(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = OrchestraConfig()
        self.lib = Librarian(self.cfg)
        self.hw = HardwareMonitor()
        self.model_intel = ModelIntelligence()
        self.engine = OrchestraEngine(self.cfg, self.lib)
        self.chat_images = []

        # Chat session management
        self.sessions_dir = os.path.join(os.path.expanduser("~"), ".orchestra_sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        self.current_session_id = None
        self.load_sessions()

        self.title("ORCHESTRA v2.9")
        self.geometry("1800x1000")
        self.setup_ui()
        self.start_monitoring()

    def load_sessions(self):
        """Load all saved chat sessions"""
        self.sessions = {}
        if os.path.exists(self.sessions_dir):
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]
                    try:
                        with open(os.path.join(self.sessions_dir, filename), 'r') as f:
                            self.sessions[session_id] = json.load(f)
                    except:
                        pass

    def save_current_session(self):
        """Save current chat session"""
        if not self.engine.history:
            messagebox.showinfo("Session", "No chat history to save.")
            return

        # Ask for session name
        dialog = tk.Toplevel(self)
        dialog.title("Save Session")
        dialog.geometry("400x150")
        dialog.configure(bg="#111111")

        tk.Label(dialog, text="Session Name:", fg="#aaaaaa", bg="#111111",
                 font=("Segoe UI", 10)).pack(pady=10)
        name_entry = tk.Entry(dialog, bg="#1e1e20", fg="white", font=("Segoe UI", 10))
        name_entry.pack(fill=tk.X, padx=20, pady=5)

        if self.current_session_id:
            name_entry.insert(0, self.sessions.get(self.current_session_id, {}).get('name', ''))

        def save():
            session_name = name_entry.get().strip()
            if not session_name:
                messagebox.showerror("Error", "Please enter a session name.")
                return

            session_id = session_name.lower().replace(' ', '_')
            self.current_session_id = session_id

            session_data = {
                'name': session_name,
                'history': self.engine.history,
                'timestamp': datetime.now().isoformat()
            }

            with open(os.path.join(self.sessions_dir, f"{session_id}.json"), 'w') as f:
                json.dump(session_data, f, indent=2)

            self.sessions[session_id] = session_data
            self.update_history_display()
            dialog.destroy()
            messagebox.showinfo("Success", f"Session '{session_name}' saved!")

        tk.Button(dialog, text="Save", bg="#4ea5f1", fg="white",
                  command=save, font=("Segoe UI", 10)).pack(pady=10)

    def new_chat_session(self):
        """Start a new chat session"""
        if self.engine.history and not self.current_session_id:
            if messagebox.askyesno("Save Current Session?",
                                   "You have unsaved chat history. Save it before starting new session?"):
                self.save_current_session()

        self.current_session_id = None
        self.engine.history = []
        self.display.config(state='normal')
        self.display.delete('1.0', tk.END)
        self.display_welcome_message()
        self.update_history_display()
        messagebox.showinfo("New Session", "Started new chat session!")

    def delete_current_session(self):
        """Delete current session"""
        if not self.current_session_id:
            messagebox.showinfo("No Session", "No active saved session to delete.")
            return

        if messagebox.askyesno("Delete Session",
                               f"Delete session '{self.sessions[self.current_session_id]['name']}'?"):
            session_file = os.path.join(self.sessions_dir, f"{self.current_session_id}.json")
            if os.path.exists(session_file):
                os.remove(session_file)
            del self.sessions[self.current_session_id]
            self.current_session_id = None
            self.engine.history = []
            self.update_history_display()
            messagebox.showinfo("Deleted", "Session deleted.")

    def load_session(self, session_id):
        """Load a saved session"""
        if session_id in self.sessions:
            self.current_session_id = session_id
            self.engine.history = self.sessions[session_id]['history']

            # Rebuild display
            self.display.config(state='normal')
            self.display.delete('1.0', tk.END)

            for entry in self.engine.history:
                self.display.insert(tk.END, f"\n\nYOU > {entry['user']}\n", "white_text")
                self.display.insert(tk.END, "ORCHESTRA > ", "big_logo")
                self.display.insert(tk.END, entry['ai'])
                self.display.insert(tk.END, "\n" + ("-" * 60) + "\n")

            self.display.config(state='disabled')
            self.update_history_display()

    def export_chat_to_file(self):
        """Export current chat to text file (original save function)"""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")])
        if file_path:
            content = self.display.get("1.0", tk.END)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Exported", "Chat exported to file.")

    def setup_ui(self):
        t = self.cfg.themes[self.cfg.theme_mode]
        self.configure(bg=t["bg"])
        for widget in self.winfo_children(): widget.destroy()

        # LEFT SIDEBAR - Fixed width, no scrolling on whole sidebar
        self.side = tk.Frame(self, bg=t["side"], width=380)
        self.side.pack(side=tk.LEFT, fill=tk.Y)
        self.side.pack_propagate(False)

        # Header (Fixed)
        tk.Label(self.side, text="ORCHESTRA", fg=t["acc"], bg=t["side"],
                 font=("Consolas", 14, "bold")).pack(pady=10, padx=20, anchor="w")

        # SEARCH LIBRARY (Fixed)
        tk.Label(self.side, text="SEARCH LIBRARY:", fg="gray", bg=t["side"],
                 font=("Arial", 9, "bold")).pack(anchor="w", padx=20, pady=(3, 2))
        self.search_entry = tk.Entry(self.side, bg=t["card"], fg="white", borderwidth=0, font=("Arial", 9))
        self.search_entry.pack(fill=tk.X, padx=20, pady=2, ipady=3)
        self.search_entry.bind("<Return>", lambda e: self.local_library_search())

        # RAG TOGGLE (Fixed)
        self.rag_btn = tk.Button(self.side, text=f"RAG: {'ON' if self.cfg.rag_enabled else 'OFF'}",
                                 bg=t["acc"] if self.cfg.rag_enabled else t["btn"], fg="white",
                                 command=self.toggle_rag, font=("Arial", 9, "bold"))
        self.rag_btn.pack(fill=tk.X, padx=20, pady=2)

        # DOCUMENT MANAGEMENT BUTTONS
        tk.Label(self.side, text="DOCUMENT MANAGEMENT", fg="gray", bg=t["side"],
                 font=("Arial", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 2))

        tk.Button(self.side, text="Index File", bg=t["btn"], fg="white",
                  command=self.upload_doc, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)
        tk.Button(self.side, text="Index Folder", bg=t["btn"], fg=t["acc"],
                  command=self.index_folder, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)
        tk.Button(self.side, text="Clear Library", bg=t["btn"], fg="#ef4444",
                  command=self.confirm_clear_lib, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)

        # CHAT MANAGEMENT BUTTONS
        tk.Label(self.side, text="CHAT MANAGEMENT", fg="gray", bg=t["side"],
                 font=("Arial", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 2))

        tk.Button(self.side, text="New Chat", bg=t["btn"], fg="#00ff00",
                  command=self.new_chat_session, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)
        tk.Button(self.side, text="Save Session", bg=t["btn"], fg="white",
                  command=self.save_current_session, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)
        tk.Button(self.side, text="Export Chat", bg=t["btn"], fg="white",
                  command=self.export_chat_to_file, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)
        tk.Button(self.side, text="Delete Session", bg=t["btn"], fg="#ef4444",
                  command=self.delete_current_session, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)

        # SYSTEM BUTTONS
        tk.Label(self.side, text="SYSTEM", fg="gray", bg=t["side"],
                 font=("Arial", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 2))

        tk.Button(self.side, text="Settings", bg=t["btn"], fg="white",
                  command=self.show_settings, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)
        tk.Button(self.side, text="Usage Stats", bg=t["btn"], fg="white",
                  command=self.show_usage_stats, font=("Arial", 9)).pack(fill=tk.X, padx=20, pady=1)

        # CHAT HISTORY AT BOTTOM - Can expand to fill remaining space!
        tk.Label(self.side, text="CHAT HISTORY", fg="gray", bg=t["side"],
                 font=("Arial", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 2))

        self.history_display = tk.Text(self.side, bg=t["card"],
                                       fg=t["txt"], font=("Arial", 8), wrap=tk.WORD,
                                       borderwidth=0, padx=6, pady=6)
        self.history_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=(2, 10))
        self.history_display.config(state='disabled')

        # MAIN CHAT AREA
        self.main = tk.Frame(self, bg=t["bg"])
        self.main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.display = scrolledtext.ScrolledText(self.main, bg=t["bg"], fg=t["txt"],
                                                 font=("Segoe UI", 11), borderwidth=0,
                                                 padx=40, pady=40)
        self.display.pack(fill=tk.BOTH, expand=True)

        self.perf_visual = VisualHardwareMonitor(self.main, self.hw, bg=t["bg"])
        self.perf_visual.pack(fill=tk.X, padx=40, pady=(0, 5))

        input_container = tk.Frame(self.main, bg=t["bg"], pady=20, padx=40)
        input_container.pack(fill=tk.X)
        self.entry = tk.Entry(input_container, bg=t["card"], fg="white",
                              font=("Segoe UI", 12), borderwidth=0, insertbackground="white")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_query())

        self.display_welcome_message()

    def update_history_display(self):
        """Update the chat history sidebar with saved sessions and recent exchanges"""
        self.history_display.config(state='normal')
        self.history_display.delete('1.0', tk.END)

        # Show saved sessions
        if self.sessions:
            self.history_display.insert(tk.END, "═══ SAVED SESSIONS ═══\n", "header")
            for session_id, session_data in sorted(self.sessions.items(),
                                                   key=lambda x: x[1].get('timestamp', ''),
                                                   reverse=True):
                is_current = (session_id == self.current_session_id)
                marker = "► " if is_current else "  "
                session_name = session_data.get('name', session_id)

                self.history_display.insert(tk.END, f"{marker}{session_name}\n", "session")

                # Make it clickable
                tag_name = f"session_{session_id}"
                start_idx = self.history_display.search(f"{marker}{session_name}", "1.0", tk.END)
                if start_idx:
                    end_idx = f"{start_idx}+{len(marker + session_name)}c"
                    self.history_display.tag_add(tag_name, start_idx, end_idx)
                    self.history_display.tag_config(tag_name, foreground="#4ea5f1", underline=True)
                    self.history_display.tag_bind(tag_name, "<Button-1>",
                                                  lambda e, sid=session_id: self.load_session(sid))
                    self.history_display.tag_bind(tag_name, "<Enter>",
                                                  lambda e, tag=tag_name: self.history_display.config(cursor="hand2"))
                    self.history_display.tag_bind(tag_name, "<Leave>",
                                                  lambda e: self.history_display.config(cursor=""))

            self.history_display.insert(tk.END, "\n")

        # Show current session preview
        if self.engine.history:
            self.history_display.insert(tk.END, "═══ CURRENT CHAT ═══\n", "header")
            for entry in self.engine.history[-3:]:  # Last 3 exchanges
                self.history_display.insert(tk.END, f"YOU: {entry['user'][:40]}...\n", "user")
                self.history_display.insert(tk.END, f"AI: {entry['ai'][:40]}...\n\n", "ai")

        self.history_display.tag_config("header", foreground="#FFD700", font=("Arial", 7, "bold"))
        self.history_display.tag_config("session", foreground="#4ea5f1")
        self.history_display.tag_config("user", foreground="#aaaaaa")
        self.history_display.tag_config("ai", foreground="#888888")
        self.history_display.config(state='disabled')
        self.history_display.see(tk.END)

    def display_welcome_message(self):
        self.display.configure(state='normal')
        self.display.delete('1.0', tk.END)
        self.display.tag_configure("white_text", foreground="#FFFFFF", font=("Courier New", 10))
        self.display.tag_configure("big_logo", foreground="#FFD700", font=("Courier New", 12, "bold"))
        self.display.tag_configure("info_text", foreground="#4ea5f1", font=("Courier New", 9))

        logo_art = r""" 
                       #######  ########   ######  ##     ## ########  ######  ######## ########     ###    
                      ##    ##  ##     ## ##    ## ##     ## ##       ##    ##    ##    ##     ##   ## ##   
                      ##    ##  ##     ## ##       ##     ## ##       ##          ##    ##     ##  ##   ##  
                      ##    ##  ########  ##       ######### ######    ######     ##    ########  ##     ## 
                      ##    ##  ##   ##   ##       ##     ## ##             ##    ##    ##   ##   ######### 
                      ##    ##  ##    ##  ##    ## ##     ## ##       ##    ##    ##    ##    ##  ##     ## 
                       #######  ##     ##  ######  ##     ## ########  ######     ##    ##     ## ##     ## 
                                     Multi-Model Local AI System                                           
                                     © 2025 Eric Varney. All rights reserved. """
        self.display.insert(tk.END, logo_art, "big_logo")

        # Show available commands
        self.display.insert(tk.END, "\n📋 SPECIAL COMMANDS:\n", "info_text")
        self.display.insert(tk.END, "  • Orchestra, open [program] - Launch applications\n", "white_text")
        self.display.insert(tk.END, "  • Orchestra, look up [query] - Search the web\n", "white_text")
        self.display.insert(tk.END, "  • Use 'search' in sidebar - Search indexed documents (RAG)\n\n", "white_text")

        if not SEARCH_AVAILABLE:
            self.display.insert(tk.END, "⚠️  WEB SEARCH DISABLED\n", "info_text")
            self.display.insert(tk.END, "Install with: pip install duckduckgo-search\n", "white_text")
            self.display.insert(tk.END, "Then restart Orchestra to enable web search.\n\n", "white_text")
        else:
            self.display.insert(tk.END, "✓ Web Search: ENABLED (use 'Orchestra, look up...')\n\n", "info_text")

        self.display.configure(state='disabled')

    def local_library_search(self):
        term = self.search_entry.get().lower()
        if not term: return
        results = [d['name'] for d in self.lib.docs if term in d['name'].lower() or term in d['content'].lower()]
        res_str = "\n".join(results[:10]) if results else "No matches found."
        messagebox.showinfo("Library Search", f"Top Matches:\n\n{res_str}")

    def toggle_rag(self):
        self.cfg.rag_enabled = not self.cfg.rag_enabled
        t = self.cfg.themes[self.cfg.theme_mode]
        self.rag_btn.config(text=f"RAG: {'ON' if self.cfg.rag_enabled else 'OFF'}",
                            bg=t["acc"] if self.cfg.rag_enabled else t["btn"])

    def confirm_clear_lib(self):
        if messagebox.askyesno("Librarian", "Wipe all indexed documents? This cannot be undone."):
            self.lib.clear_library()
            messagebox.showinfo("Librarian", "Library cleared.")

    def clear_chat_display(self):
        """Clear current chat (for compatibility)"""
        self.new_chat_session()

    def index_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            def worker():
                count = 0
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.lower().endswith(
                                ('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.c', '.cpp', '.rs', '.go',
                                 '.pdf')):
                            self.lib.add_doc(os.path.join(root, file))
                            count += 1
                self.after(0, lambda: messagebox.showinfo("Librarian", f"Successfully indexed {count} files."))

            threading.Thread(target=worker, daemon=True).start()

    def refresh_expert_cards(self, active_list):
        # Expert cards removed from UI - still tracks in background for stats
        pass

    def upload_doc(self):
        file = filedialog.askopenfilename()
        if file and self.lib.add_doc(file):
            messagebox.showinfo("Librarian", "File Indexed.")

    def show_usage_stats(self):
        if not self.cfg.expert_usage_stats:
            messagebox.showinfo("Usage Statistics", "No usage data yet. Start using the system to see statistics!")
            return

        total_uses = sum(self.cfg.expert_usage_stats.values())
        stats_text = "Expert Usage Statistics:\n\n"

        sorted_stats = sorted(self.cfg.expert_usage_stats.items(), key=lambda x: x[1], reverse=True)
        for expert, count in sorted_stats:
            percentage = (count / total_uses) * 100
            stats_text += f"{expert}: {count} uses ({percentage:.1f}%)\n"

        messagebox.showinfo("Usage Statistics", stats_text)

    def create_custom_domain(self, specialist_info):
        dialog = tk.Toplevel(self)
        dialog.title("Create Custom Domain")
        dialog.geometry("500x300")
        dialog.configure(bg="#111111")

        tk.Label(dialog, text=f"Create domain for: {specialist_info['model']}",
                 fg="#4ea5f1", bg="#111111", font=("Segoe UI", 12, "bold")).pack(pady=20)

        tk.Label(dialog, text="Domain Name:", fg="#aaaaaa", bg="#111111").pack(anchor="w", padx=20)
        name_entry = tk.Entry(dialog, bg="#1e1e20", fg="white", font=("Segoe UI", 10))
        name_entry.insert(0, specialist_info['suggested_domain'])
        name_entry.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(dialog, text="Description:", fg="#aaaaaa", bg="#111111").pack(anchor="w", padx=20, pady=(10, 0))
        desc_entry = tk.Entry(dialog, bg="#1e1e20", fg="white", font=("Segoe UI", 10))
        desc_entry.pack(fill=tk.X, padx=20, pady=5)

        def create():
            domain_name = name_entry.get().strip().replace(" ", "_")
            description = desc_entry.get().strip()

            if domain_name and description:
                self.cfg.domain_definitions[domain_name] = {
                    "description": description,
                    "default_model": specialist_info['model']
                }
                self.cfg.manual_assignments[domain_name] = specialist_info['model']
                self.cfg.refresh_available_models()
                dialog.destroy()
                messagebox.showinfo("Success", f"Created domain: {domain_name}")

        tk.Button(dialog, text="Create Domain", bg="#4ea5f1", fg="white",
                  command=create, font=("Segoe UI", 10, "bold"), pady=10).pack(pady=20)

    def show_settings(self):
        self.cfg.refresh_available_models()
        win = tk.Toplevel(self)
        win.title("Model Configuration")
        win.geometry("1000x850")
        win.configure(bg="#111111")

        tk.Label(win, text="Model Configuration", fg="#4ea5f1", bg="#111111",
                 font=("Segoe UI", 18, "bold")).pack(anchor="w", padx=25, pady=(25, 10))

        if self.cfg.auto_assignments or self.cfg.unassigned_specialists:
            summary_frame = tk.Frame(win, bg="#1a1a1a", relief=tk.RIDGE, borderwidth=2)
            summary_frame.pack(fill=tk.X, padx=25, pady=(10, 20))

            tk.Label(summary_frame, text="🤖 Auto-Detection Summary", fg="#4ea5f1", bg="#1a1a1a",
                     font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(10, 5))

            tk.Label(summary_frame,
                     text=f"Detected {len(self.cfg.available_models)} models | "
                          f"Auto-assigned {len(self.cfg.auto_assignments)} domains | "
                          f"Conductor: {self.cfg.conductor}",
                     fg="#aaaaaa", bg="#1a1a1a", font=("Segoe UI", 9)).pack(anchor="w", padx=15, pady=(0, 10))

            if self.cfg.auto_assignments:
                high_conf = [f"{domain} ({conf:.0%})" for domain, (model, conf) in self.cfg.auto_assignments.items() if
                             conf > 0.7]
                if high_conf:
                    confidence_text = "High Confidence: " + ", ".join(high_conf)
                    tk.Label(summary_frame, text=confidence_text, fg="#55ff55", bg="#1a1a1a",
                             font=("Segoe UI", 8), wraplength=900, justify=tk.LEFT).pack(anchor="w", padx=15,
                                                                                         pady=(0, 10))

        outer_frame = tk.Frame(win, bg="#111111")
        outer_frame.pack(fill=tk.BOTH, expand=True, padx=25)

        canvas = tk.Canvas(outer_frame, bg="#111111", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#111111")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=950)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Label(scrollable_frame, text="System Global Settings", fg="#eeeeee", bg="#111111",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(10, 5))

        vram_var = tk.BooleanVar(value=self.cfg.vram_optim)
        vram_chk = tk.Checkbutton(scrollable_frame,
                                  text="Aggressive VRAM Optimization (Unloads models after generation)",
                                  variable=vram_var, bg="#111111", fg="#bbbbbb", selectcolor="#111111",
                                  activebackground="#111111", activeforeground="white", font=("Segoe UI", 9))
        vram_chk.pack(anchor="w", pady=5)

        topk_f = tk.Frame(scrollable_frame, bg="#111111")
        topk_f.pack(fill=tk.X, pady=5)
        tk.Label(topk_f, text="Max Experts (1-5 experts per query):", fg="#999999", bg="#111111", width=35,
                 anchor="w", font=("Segoe UI", 9)).pack(side=tk.LEFT)

        topk_slider = tk.Scale(topk_f, from_=1, to=5, orient=tk.HORIZONTAL, bg="#111111", fg="white",
                               highlightthickness=0, troughcolor="#1e1e20", activebackground="#4ea5f1")
        topk_slider.set(self.cfg.max_experts)
        topk_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        tk.Label(scrollable_frame, text="Conductor LLM (Synthesizer)", fg="#eeeeee", bg="#111111",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(15, 5))

        cond_f = tk.Frame(scrollable_frame, bg="#111111")
        cond_f.pack(fill=tk.X, pady=5)
        tk.Label(cond_f, text="Conductor (Global Synthesizer):", fg="#999999", bg="#111111", width=35, anchor="w",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        cond_cb = ttk.Combobox(cond_f, values=self.cfg.available_models, font=("Segoe UI", 9))
        cond_cb.set(self.cfg.conductor)
        cond_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Frame(scrollable_frame, height=1, bg="#333333").pack(fill=tk.X, pady=20)

        tk.Label(scrollable_frame, text="Expert Domain Assignments", fg="#eeeeee", bg="#111111",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))

        expert_combos = {}
        for domain in sorted(self.cfg.domain_definitions.keys()):
            f = tk.Frame(scrollable_frame, bg="#111111")
            f.pack(fill=tk.X, pady=3)

            display_name = domain.replace('_', ' ')

            auto_indicator = ""
            label_color = "#bbbbbb"

            if domain in self.cfg.auto_assignments:
                model, confidence = self.cfg.auto_assignments[domain]
                if confidence > 0.7:
                    auto_indicator = " ✓ (Auto: High)"
                    label_color = "#55ff55"
                else:
                    auto_indicator = " ⚡ (Auto: Medium)"
                    label_color = "#ffaa00"
            elif domain in self.cfg.manual_assignments:
                auto_indicator = " ✏️ (Manual)"
                label_color = "#4ea5f1"

            tk.Label(f, text=f"{display_name}{auto_indicator}", fg=label_color, bg="#111111",
                     width=40, anchor="w", font=("Segoe UI", 9)).pack(side=tk.LEFT)

            cb = ttk.Combobox(f, values=self.cfg.available_models, font=("Segoe UI", 9))
            cb.set(self.cfg.experts.get(domain, self.cfg.conductor))
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            expert_combos[domain] = cb

        if self.cfg.unassigned_specialists:
            tk.Frame(scrollable_frame, height=1, bg="#333333").pack(fill=tk.X, pady=20)

            tk.Label(scrollable_frame, text="💡 Detected Specialized Models (Create New Domains?)",
                     fg="#ffaa00", bg="#111111", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))

            for specialist in self.cfg.unassigned_specialists:
                f = tk.Frame(scrollable_frame, bg="#1a1a1a", relief=tk.FLAT, borderwidth=1)
                f.pack(fill=tk.X, pady=2, padx=5)

                tk.Label(f, text=f"Model: {specialist['model']}", fg="#ffffff", bg="#1a1a1a",
                         font=("Segoe UI", 9, "bold"), width=30, anchor="w").pack(side=tk.LEFT, padx=10)

                tk.Label(f, text=f"→ Suggested: {specialist['suggested_domain']}", fg="#ffaa00", bg="#1a1a1a",
                         font=("Segoe UI", 9), width=30, anchor="w").pack(side=tk.LEFT)

                tk.Button(f, text="Create Domain", bg="#4ea5f1", fg="white",
                          command=lambda s=specialist: self.create_custom_domain(s)).pack(side=tk.RIGHT, padx=10,
                                                                                          pady=5)

        tk.Label(scrollable_frame, text="Add New Expert:", fg="#555555", bg="#111111",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(20, 5))

        f_new = tk.Frame(scrollable_frame, bg="#111111")
        f_new.pack(fill=tk.X, pady=3)
        tk.Label(f_new, text="Select model to create slot:", fg="#555555", bg="#111111", width=35, anchor="w",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        new_slot_cb = ttk.Combobox(f_new, values=self.cfg.available_models, font=("Segoe UI", 9))
        new_slot_cb.set("")
        new_slot_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def save_and_close():
            try:
                self.cfg.vram_optim = vram_var.get()
                self.cfg.max_experts = int(topk_slider.get())  # Fixed: was top_k
                self.cfg.conductor = cond_cb.get()

                for domain, combo in expert_combos.items():
                    new_value = combo.get()
                    if domain in self.cfg.auto_assignments:
                        auto_model, _ = self.cfg.auto_assignments[domain]
                        if new_value != auto_model:
                            self.cfg.manual_assignments[domain] = new_value
                    else:
                        if new_value != self.cfg.conductor:
                            self.cfg.manual_assignments[domain] = new_value

                if cond_cb.get() != self.model_intel.suggest_conductor(self.cfg.available_models):
                    self.cfg.manual_assignments["conductor"] = cond_cb.get()

                self.cfg.refresh_available_models()
                self.refresh_expert_cards([])

            except Exception as e:
                print(f"DEBUG: Save error: {e}")
            finally:
                win.destroy()

        tk.Button(win, text="Save Configuration", bg="#4ea5f1", fg="white", font=("Segoe UI", 10, "bold"),
                  command=save_and_close, borderwidth=0, padx=20, pady=10).pack(pady=20)

    def show_render_status(self):
        self.display.configure(state='normal')
        self.display.insert(tk.END, "\n\n[ARTISAN] Local GPU render initiated... (Low-VRAM optimization active)\n",
                            "white_text")
        self.display.see(tk.END)
        self.display.configure(state='disabled')

    def append_image(self, img_data):
        img = Image.open(io.BytesIO(img_data))
        canvas_width = self.display.winfo_width() - 100
        w, h = img.size
        aspect = h / w
        new_w = min(800, canvas_width)
        new_h = int(new_w * aspect)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.chat_images.append(photo)
        self.display.configure(state='normal')
        self.display.image_create(tk.END, image=photo)
        self.display.insert(tk.END, "\n\n")
        self.display.see(tk.END)
        self.display.configure(state='disabled')

    def send_query(self):
        query = self.entry.get()
        if not query: return
        self.entry.delete(0, tk.END)

        t = self.cfg.themes[self.cfg.theme_mode]
        self.display.configure(state='normal')
        self.display.insert(tk.END, f"\n\nYOU > {query}\n", "white_text")
        self.display.configure(state='disabled')

        def callback(response, active_experts):
            self.after(0, lambda: self.refresh_expert_cards(active_experts))
            self.after(0, lambda: self._stream_text(response))
            self.after(0, lambda: self.update_history_display())  # UPDATE HISTORY!

        asyncio.run_coroutine_threadsafe(self.engine.generate_response(query, callback, self.append_image, self),
                                         self.loop)

    def _stream_text(self, text):
        self.display.configure(state='normal')
        self.display.insert(tk.END, "ORCHESTRA > ", "big_logo")
        for char in text:
            self.display.insert(tk.END, char)
            self.display.see(tk.END)
            self.update()
            time.sleep(0.001)
        self.display.insert(tk.END, "\n" + ("-" * 60) + "\n")
        self.display.configure(state='disabled')

    def start_monitoring(self):
        def run_loop():
            while True:
                data = self.hw.get_update()
                self.after(0, lambda d=data: self.perf_visual.refresh(d))
                time.sleep(1)

        threading.Thread(target=run_loop, daemon=True).start()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()


if __name__ == "__main__":
    app = OrchestraV2()
    app.mainloop()
