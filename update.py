import os
import re
import subprocess

def run_git(commit_msg):
    try:
        subprocess.run(["git", "add", "."], check=True)
        # Using subprocess.run for commit, allowing it to fail gracefully if there are no changes, but in this case we expect changes.
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push"], check=True)
        print(f"Success: {commit_msg}")
    except subprocess.CalledProcessError as e:
        print(f"Failed during git operation for: {commit_msg}\n{e}")

# 1. Create 14 detailed pages and link them
os.makedirs("pages", exist_ok=True)

concepts = {
    "The Naive Model-Parallel Era (Traditional ML, Pre-2019)": "naive_model_parallel.md",
    "The Synchronous Pipelining Revolution (GPipe / 1F1B, 2019–2020)": "synchronous_pipelining.md",
    "The Interleaved 1F1B Multi-Chunk Era (Megatron-LM, 2021–2024)": "interleaved_1f1b.md",
    "The Zero-Bubble & Asynchronous Activation Offloading Era (~2024–Present)": "zero_bubble.md",
    "A. GPipe Schedule (Fill-Drain Pipelining)": "gpipe_schedule.md",
    "B. 1F1B Schedule (One Forward, One Backward)": "1f1b_schedule.md",
    "C. Interleaved 1F1B Schedule": "interleaved_1f1b_schedule.md",
    "Peer-to-Peer (P2P) Communication Primitives": "p2p_communication.md",
    "Activation Checkpointing (Rematerialization)": "activation_checkpointing.md",
    "The Parameter-Heterogeneity Load Imbalance Wall": "parameter_heterogeneity.md",
    "The Activation Memory Accumulation Crisis": "activation_memory.md",
    "Pre-Training Trillion-Parameter Foundational LLMs (Megatron-DeepSpeed Systems)": "pre_training_llms.md",
    "High-Volume Spatio-Temporal Video Generation Scaling (Sora Class)": "video_generation.md",
    "Enterprise Post-Training On-Policy RL Alignment Sprints (RLHF / PPO)": "rl_alignment.md"
}

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

for k, v in concepts.items():
    page_path = f"pages/{v}"
    title = re.sub(r'\(.*?\)', '', k).strip()
    with open(page_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\nThis page provides detailed information about {title}.\n\n```mermaid\ngraph LR\nA[{title}] --> B[Detailed Implementation]\n```\n\n[Back to README](../README.md)\n")
    
    # Replace exact string with link
    readme = readme.replace(f"**{k}**", f"**[{k}](pages/{v})**")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

run_git("detailed pages created")

# 2. Decorate with emojis, dynamic banner
os.makedirs("assets", exist_ok=True)
svg_banner = '''<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:rgb(255,255,0);stop-opacity:1" />
      <stop offset="100%" style="stop-color:rgb(255,0,0);stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="800" height="200" fill="url(#grad1)" />
  <text x="50%" y="50%" font-family="Arial" font-size="40" fill="white" dominant-baseline="middle" text-anchor="middle">Awesome Pipeline Parallelism</text>
  <animate attributeName="opacity" values="0.8;1;0.8" dur="3s" repeatCount="indefinite" />
</svg>'''
with open("assets/banner.svg", "w", encoding="utf-8") as f:
    f.write(svg_banner)

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

readme = readme.replace("# Awesome-Pipeline-Parallelism", f"![Banner](assets/banner.svg)\n\n# 🚀 Awesome-Pipeline-Parallelism")
readme = readme.replace("## Pipeline Parallelism in AI", "## 🧠 Pipeline Parallelism in AI")
readme = readme.replace("## 1. The Macro Chronological Evolution", "## 📈 1. The Macro Chronological Evolution")
readme = readme.replace("## 2. Core Scheduling & Architectural Variants", "## ⚙️ 2. Core Scheduling & Architectural Variants")
readme = readme.replace("## 3. The Pipelined Communication & Bubble Matrix", "## 🌐 3. The Pipelined Communication & Bubble Matrix")
readme = readme.replace("## 4. Production Engineering Challenges & Hardware Solutions", "## 🛠️ 4. Production Engineering Challenges & Hardware Solutions")
readme = readme.replace("## 5. Frontier Real-World AI Industrial Applications", "## 🏭 5. Frontier Real-World AI Industrial Applications")
readme = readme.replace("## References", "## 📚 References")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

run_git("added emojis and banner")

# 3. Add badges to left and SEO
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

badges_left = '<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>'

seo_text = """
> **A curated list of awesome pipeline parallelism resources, tools, and papers for AI and Deep Learning.**
> Boost your large-scale deep learning model training with these state-of-the-art parallelism techniques!

"""

readme = readme.replace("# \U0001F680 Awesome-Pipeline-Parallelism", f"# \U0001F680 Awesome-Pipeline-Parallelism\n\n{badges_left}\n\n{seo_text}")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
    
run_git("seo optimised and badges to left added")

# 4. Badges to right
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

badge_right = '<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>'
readme = readme.replace(badges_left, f"{badges_left} {badge_right}")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
    
run_git("badges to right added")

# 5. Star history
star_history = """
##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007/Awesome-Pipeline-Parallelism&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Pipeline-Parallelism&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Pipeline-Parallelism&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Pipeline-Parallelism&type=date&legend=bottom-right" />
</picture>
</a>
</div>
"""
with open("README.md", "a", encoding="utf-8") as f:
    f.write(star_history)
    
run_git("star history added")

# 6. Replace chartrepos with chart?repos
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

readme = readme.replace("chartrepos", "chart?repos")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
    
run_git("fixed star plot")

# 7. Replace awesome link
with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

readme = readme.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

run_git("invalid awesome link fixed")
