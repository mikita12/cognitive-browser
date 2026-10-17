# Demo Recording Guide

This guide describes how to record an interactive, high-quality demonstration GIF showcasing Cognitive Browser's Reflex Core (~25 ms latency) and Cortex escalation in action.

*(Note: Do not check large binary GIF files into git history; host demo GIFs as GitHub release assets or via external CDN).*

---

## 1. Recommended Tooling

We recommend [VHS by Charmbracelet](https://github.com/charmbracelet/vhs) for fully automated, reproducible terminal GIF recordings:

```bash
# Install VHS (on Linux via homebrew, pacman, or binary download)
sudo apt install ttyrec ffmpeg # or brew install vhs
# Or download binary from: https://github.com/charmbracelet/vhs/releases
```

Alternative GUI screen recorders:
- **Peek** (`sudo apt install peek`) — Lightweight screen area recorder for Linux / Wayland / X11.
- **Asciinema + agg** (`asciinema rec demo.cast && agg demo.cast demo.gif`).

---

## 2. Demo Scenarios

Choose one of two demonstration scenarios depending on whether you want a pure terminal view or a side-by-side browser view:

### Option A: Test Suite Demonstration (Fast & Reproducible)
Demonstrates the Reflex Core making local NLI routing decisions without network latency:

```bash
pytest tests/ -v
```

### Option B: Headful Browser Automation
Shows a live browser window filling forms, dismissing cookie overlays, and completing tasks in real time:

Edit `examples/wroclaw_pizza_hunter.py` to set `headless=False`:
```python
browser = await p.chromium.launch(headless=False, slow_mo=100)
```

---

## 3. VHS Script Template (`demo.tape`)

Create a `demo.tape` file with the following contents:

```tape
# VHS recording script for Cognitive Browser
Output demo.gif

Set FontSize 15
Set FontFamily "JetBrains Mono"
Set Width 1200
Set Height 680
Set Padding 20
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful

Type "pytest tests/ -v"
Sleep 500ms
Enter

# Wait for test suite execution
Sleep 5s
Sleep 500ms
Enter
Sleep 4s
```

Run VHS to generate the optimized GIF:
```bash
vhs demo.tape
```

---

## 4. GIF Optimization & Hosting Guidelines

1. **Target Dimensions:** 1200x680 px or 1080x720 px.
2. **Target File Size:** Keep below 4 MB so GitHub renders it smoothly without stutter.
3. **Palette Optimization:** If recording manually with `ffmpeg`, use a two-pass palette filter:
   ```bash
   ffmpeg -i input.mp4 -vf "fps=15,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 demo.gif
   ```
4. **Embedding in README:** Upload the GIF to a GitHub issue or Release, then reference it using standard markdown:
   ```markdown
   ![Cognitive Browser Demo](https://github.com/mikita12/cognitive-browser/releases/download/v0.1.0/demo.gif)
   ```
