# Fourier Circle Drawing

> Turn any image into an animated Fourier epicycle drawing — rotating circles that trace your image's contours using the Discrete Fourier Transform.

---

## Demo

| Input image | Fourier animation |
|:-----------:|:-----------------:|
| ![input](images/test.png) | ![demo](images/test.gif) |

---

## What is this?

This tool uses the **Discrete Fourier Transform (DFT)** to decompose the contours of an image into a set of rotating circles (epicycles). When the circles spin together at different frequencies and radii, their combined tip traces the original shape — just like how any periodic signal can be reconstructed from sine waves.

Inspired by [3Blue1Brown's Fourier series videos](https://www.youtube.com/watch?v=r6sGWTCMz2k).

---

## Features

- **Load any image** (PNG / JPG / BMP) — the app auto-extracts its contours
- **Real-time preview** of the Fourier reconstruction inside the GUI
- **Interactive animation** rendered with Pygame showing all the spinning epicircles
- **Detail Level** slider (0–100 %) controls how many Fourier coefficients are used
  - 0 % = abstract / minimal  →  100 % = maximum fidelity
- **Scale** slider (10–500 %) zooms the preview and animation
- Precise parameter input via **spinboxes** (type a value or use ↑↓ keys)
- **Save Preview as PNG** — export the current static preview with one click
- **Auto GIF recording** — every animation is automatically recorded; click **Download GIF Recording** to save it to the `recordings/` folder
- **Screenshot shortcut** — press **S** in the animation window → saved to `screenshots/` with a timestamp

---

## Requirements

Python 3.8+

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image → SVG contour extraction |
| `numpy` | FFT computation |
| `pyqt5` | Main GUI window |
| `pygame` | Animated epicycle rendering |
| `Pillow` | GIF recording |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the GUI
python main.py
```

### Step-by-step

1. Click **① Load Image** — choose any PNG or JPG
2. Click **② Generate Preview** — the app converts the image to SVG contours and runs the FFT
3. Adjust **Detail Level** and **Scale** to taste
4. Click **Save Preview as PNG** to export the static result
5. Click **③ Start Animation** — watch the epicircles draw your image!
6. When the animation window closes, click **Download GIF Recording** to save the GIF to `recordings/`

### Animation window controls

| Key | Effect |
|-----|--------|
| **S** | Save screenshot → `screenshots/fourier_YYYYMMDD_HHMMSS.png` |
| **ESC** | Exit the animation |

---

## CLI usage

```bash
# GUI with a specific starting detail level (0–300 coefficients)
python main.py --gui --coeffs 150

# Run animation directly on an SVG file
python main.py --svg path/to/file.svg --coeffs 100 --scale 1.2
```

---

## How it works

```
Image (PNG/JPG)
    │
    ▼  OpenCV: threshold + findContours
SVG paths  (series of (x, y) points per stroke)
    │
    ▼  NumPy FFT  →  low-pass filter (keep N coefficients)
Fourier coefficients  [amplitude, frequency, phase]
    │
    ▼  IFFT reconstruction  /  real-time synthesis
Animated epicircles  →  traces the original shape
```

---

## Project structure

```
Fourier_drawing/
├── main.py          # Entry point — GUI + CLI
├── fft.py           # FFT processing + Pygame animation + GIF recording
├── SVG.py           # Bitmap → SVG contour converter
├── requirements.txt
├── icon.png         # App icon
├── images/          # Intermediate files (temp SVG, demo assets)
├── recordings/      # Auto-saved GIF recordings
└── screenshots/     # Auto-saved animation screenshots
```

---

## References

- [3Blue1Brown — But what is a Fourier series?](https://www.youtube.com/watch?v=r6sGWTCMz2k)
- [Fourier series interactive demo (jezzamon.com)](https://www.jezzamon.com/fourier/index.html)
- [FourierCircleDrawing (ruanluyu)](https://github.com/ruanluyu/FourierCircleDrawing)

---

## License

MIT
