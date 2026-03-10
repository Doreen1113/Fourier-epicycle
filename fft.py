import os
import math
import re
import datetime
import colorsys
import tempfile

import numpy as np
from numpy.fft import fft, ifft
import pygame
from pygame.locals import *

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def fftProcess(svg_filename: str, n_coeffs: int = 50):
    """Parse an SVG file, run FFT on each path, and return Fourier coefficients
    together with the global bounding-box center."""
    with open(svg_filename, "r") as f:
        content = f.read()

    paths = re.findall(r'\bd="(.*?)"', content)
    all_pts = []
    for path in paths:
        nums = re.findall(r"-?\d+\.?\d*", path)
        for i in range(0, len(nums), 2):
            all_pts.append((float(nums[i]), float(nums[i + 1])))

    if not all_pts:
        return [], (0, 0)

    # Compute global bounding-box center so all paths stay aligned.
    center_x = (max(p[0] for p in all_pts) + min(p[0] for p in all_pts)) / 2
    center_y = (max(p[1] for p in all_pts) + min(p[1] for p in all_pts)) / 2

    all_segments_fourier = []
    for path in paths:
        nums = re.findall(r"-?\d+\.?\d*", path)
        segment = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]
        if len(segment) < 5:
            continue

        # Convert coordinates to complex numbers, shifted to the global center.
        y = np.array([complex(p[0] - center_x, p[1] - center_y) for p in segment])
        N = len(y)
        yy = fft(y)

        # Low-pass filter: zero out the middle frequencies, keep n_coeffs coefficients.
        if n_coeffs and n_coeffs < N:
            half = n_coeffs // 2
            yy[half : N - half] = 0

        # Convert to (amplitude, frequency, phase) tuples.
        # Frequency indices above N/2 represent negative frequencies.
        PP = []
        for i, v in enumerate(yy):
            freq = i if i <= N // 2 else i - N
            amp = abs(v) / N
            phase = np.angle(v)
            PP.append([amp, freq, phase])

        # Do not sort — preserve original frequency ordering for consistent IFFT reconstruction.
        all_segments_fourier.append(PP)

    return all_segments_fourier, (center_x, center_y)


def get_reconstructed_points(svg_filename: str, n_coeffs: int = 50):
    """Parse SVG paths, apply FFT → low-pass filter → IFFT on each path.
    Returns (list_of_point_lists, (gminx, gmaxx, gminy, gmaxy)).
    """
    with open(svg_filename, "r") as f:
        content = f.read()

    paths = re.findall(r'\bd="(.*?)"', content)
    global_all_pts = []
    for path in paths:
        nums = re.findall(r"-?\d+\.?\d*e?-?\d*?", path)
        pts = []
        for i in range(0, len(nums), 2):
            try:
                x = float(nums[i])
                y = float(nums[i + 1])
            except Exception:
                continue
            pts.append((x, y))
        if pts:
            global_all_pts.extend(pts)

    if not global_all_pts:
        return [], (0, 0, 0, 0)

    xs = [p[0] for p in global_all_pts]
    ys = [p[1] for p in global_all_pts]
    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2

    all_reconstructed = []
    gminx = gminy = float('inf')
    gmaxx = gmaxy = float('-inf')

    for path in paths:
        nums = re.findall(r"-?\d+\.?\d*e?-?\d*?", path)
        pts = []
        for i in range(0, len(nums), 2):
            try:
                x = float(nums[i])
                y = float(nums[i + 1])
            except Exception:
                continue
            pts.append((x, y))

        if len(pts) < 3:
            continue

        # Build complex array centered at the global origin.
        y_complex = np.array(
            [complex(px - center_x, py - center_y) for px, py in pts], dtype=complex
        )
        N = len(y_complex)
        yy = fft(y_complex)

        # Low-pass: zero middle frequencies, keep n_coeffs (head + tail).
        if n_coeffs is not None and 0 < n_coeffs < N:
            half = int(n_coeffs) // 2
            yy[half : N - half] = 0

        rec = ifft(yy)
        rec_pts = [(float(z.real + center_x), float(z.imag + center_y)) for z in rec]

        for x, y in rec_pts:
            gminx = min(gminx, x); gmaxx = max(gmaxx, x)
            gminy = min(gminy, y); gmaxy = max(gmaxy, y)

        all_reconstructed.append(rec_pts)

    return all_reconstructed, (gminx, gmaxx, gminy, gmaxy)


# === Animation: Fourier epicycle renderer ===
def draw(filename: str, n_coeffs: int = 50, user_scale: float = 1.0):
    """Run the Fourier epicycle animation.
    Returns the path to a temp GIF file of the recording, or None.
    """
    WINDOW_W, WINDOW_H = 1200, 800
    FPS = 60
    point_size = 1
    thickness = 2
    start_xy = (WINDOW_W // 2, WINDOW_H // 2)
    b_length = 10000

    # GIF recording settings
    GIF_CAPTURE_EVERY = 2                               # 1 in every 2 frames → 30 fps
    GIF_W, GIF_H     = 600, 400                         # output resolution (50 % of window)
    GIF_MAX_FRAMES   = 150                              # cap at ~5 s of recording
    GIF_FRAME_MS     = int(1000 * GIF_CAPTURE_EVERY / FPS)  # ms per GIF frame (~33 ms)

    all_segments, _ = fftProcess(filename, n_coeffs=n_coeffs)
    if not all_segments:
        print("Error: No paths found in SVG.")
        return None

    # Auto-fit: use the same bbox-based calibration as the GUI preview.
    _, rec_bbox = get_reconstructed_points(filename, n_coeffs)
    svg_w = max(1.0, rec_bbox[1] - rec_bbox[0])
    svg_h = max(1.0, rec_bbox[3] - rec_bbox[2])
    base_fit = min(WINDOW_W / svg_w, WINDOW_H / svg_h) * 0.8
    scale = base_fit * user_scale

    pygame.init()
    # Fixed-size window — RESIZABLE omitted so the maximize button is disabled.
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.DOUBLEBUF)
    pygame.display.set_caption(
        f"Fourier Circle Drawing  ·  {os.path.basename(filename)}"
        f"  ·  Detail {round(n_coeffs / 300 * 100)}%"
    )
    _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    if os.path.exists(_icon_path):
        pygame.display.set_icon(pygame.image.load(_icon_path))

    try:
        font_ui   = pygame.font.SysFont("segoeui", 14)
        font_hint = pygame.font.SysFont("segoeui", 13)
    except Exception:
        font_ui   = pygame.font.Font(None, 18)
        font_hint = pygame.font.Font(None, 16)

    # --- Epicircle definition ---
    class Circle:
        def __init__(self, r, freq, phase, color, father=None):
            self.r = r
            self.freq = freq    # rotation frequency
            self.phase = phase  # initial phase
            self.color = color
            self.father = father
            self.x, self.y = 0, 0

        def update(self, t):
            # Fourier synthesis: r * exp(i * (2π·freq·t + phase)); t is in [0, 1].
            angle = self.freq * 2 * math.pi * t + self.phase
            if self.father:
                self.x = self.father.x + self.r * math.cos(angle) * scale
                self.y = self.father.y + self.r * math.sin(angle) * scale
            else:
                # Root circle is anchored at the screen center.
                self.x, self.y = start_xy

        def draw(self, screen, show_circles=True):
            if not self.father:
                return
            if show_circles:
                # Draw guide circle and spoke in a dimmed color.
                alpha_color = tuple(map(lambda x: x // 4, self.color))
                center = (int(self.father.x), int(self.father.y))
                radius = max(int(abs(self.r) * scale), 1)
                pygame.draw.circle(screen, alpha_color, center, radius, 1)
                pygame.draw.line(screen, self.color, center, (int(self.x), int(self.y)), 1)
            # Draw the tip dot.
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), point_size)

    # --- Per-path epicycle chain ---
    class PathDrawer:
        def __init__(self, fourier_list, color):
            self.circles = []
            self.points  = []
            self.color   = color
            self.t       = 0.0

            # Build the circle chain; root is a stationary center point.
            root = Circle(0, 0, 0, color)
            self.circles.append(root)
            for p in fourier_list:
                # p = [amplitude, frequency, phase]
                c = Circle(p[0], p[1], p[2], color, father=self.circles[-1])
                self.circles.append(c)

        def step(self, screen, dt):
            self.t += dt
            if self.t > 1.0:
                self.t = 0  # loop the animation

            # Update every circle's position.
            for c in self.circles:
                c.update(self.t)
                c.draw(screen)

            # Record the tip position to trace the path.
            tail = self.circles[-1]
            self.points.append((tail.x, tail.y))

            # Cap trail length to prevent unbounded memory growth.
            if len(self.points) > b_length:
                self.points.pop(0)

            # Draw the trail (the traced shape).
            if len(self.points) > 2:
                pygame.draw.lines(screen, self.color, False, self.points, thickness)

    # --- Build a drawer for each path segment ---
    drawers = []
    for idx, segment in enumerate(all_segments):
        hue = idx / len(all_segments)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        color = (int(r * 255), int(g * 255), int(b * 255))
        drawers.append(PathDrawer(segment, color))

    clock = pygame.time.Clock()
    running          = True
    dt               = 1.0 / (FPS * 5)   # 5 s per full cycle
    pending_screenshot = False
    flash_msg        = ""
    flash_until      = 0
    frame_counter    = 0
    raw_frames       = []  # list of numpy (H, W, 3) uint8 arrays for GIF

    def _take_screenshot():
        """Capture the current frame (no UI overlays) and save to screenshots/."""
        os.makedirs("screenshots", exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"screenshots/fourier_{ts}.png"
        pygame.image.save(screen, name)
        return name

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_s:
                    pending_screenshot = True

        screen.fill((17, 17, 27))

        for drawer in drawers:
            drawer.step(screen, dt)

        # --- Capture clean frame for GIF (before UI overlays) ---
        if _PIL_AVAILABLE and len(raw_frames) < GIF_MAX_FRAMES:
            if frame_counter % GIF_CAPTURE_EVERY == 0:
                small = pygame.transform.scale(screen, (GIF_W, GIF_H))
                arr   = pygame.surfarray.array3d(small)
                raw_frames.append(arr.transpose([1, 0, 2]).copy())

        # --- Screenshot (also before UI overlays) ---
        if pending_screenshot:
            saved       = _take_screenshot()
            flash_msg   = f"Saved  {os.path.basename(saved)}"
            flash_until = now + 2200
            pending_screenshot = False

        # --- Minimal hint bar (bottom-left) ---
        if _PIL_AVAILABLE:
            rec_text = "● REC" if len(raw_frames) < GIF_MAX_FRAMES else "■ MAX"
            rec_color = (240, 100, 100) if len(raw_frames) < GIF_MAX_FRAMES else (150, 150, 150)
            rec_surf = font_hint.render(rec_text, True, rec_color)
            screen.blit(rec_surf, (10, WINDOW_H - 22))
            hint_surf = font_hint.render("  |  S: screenshot  |  ESC: exit", True, (88, 91, 112))
            screen.blit(hint_surf, (10 + rec_surf.get_width(), WINDOW_H - 22))
        else:
            hint_surf = font_hint.render("S: screenshot  |  ESC: exit", True, (88, 91, 112))
            screen.blit(hint_surf, (10, WINDOW_H - 22))

        # --- Flash confirmation (fades out over the last 0.5 s) ---
        if flash_msg and now < flash_until:
            alpha      = min(255, (flash_until - now) * 255 // 500)
            flash_surf = font_ui.render(flash_msg, True, (166, 227, 161))
            flash_surf.set_alpha(alpha)
            screen.blit(flash_surf, (WINDOW_W - flash_surf.get_width() - 14, WINDOW_H - 56))

        pygame.display.update()
        clock.tick(FPS)
        frame_counter += 1

    pygame.quit()

    # --- Encode and save GIF to a temp file after the window closes ---
    if not _PIL_AVAILABLE or not raw_frames:
        return None

    try:
        tmp_fd, gif_path = tempfile.mkstemp(suffix=".gif", prefix="fourier_anim_")
        os.close(tmp_fd)

        # Build a single global palette from a vertical strip of sampled frames.
        # Using one shared palette across all frames eliminates per-frame flicker
        # and gives more consistent colors throughout the animation.
        sample_step = max(1, len(raw_frames) // 20)
        strip = np.vstack(raw_frames[::sample_step])          # (n*H, W, 3)
        global_q = _PILImage.fromarray(strip).quantize(colors=256, method=2, dither=0)

        # Re-quantize every frame against the shared palette (dither=0: no noise).
        pil_frames = [
            _PILImage.fromarray(arr).quantize(palette=global_q, dither=0)
            for arr in raw_frames
        ]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=GIF_FRAME_MS,
            optimize=False,
        )
        return gif_path
    except Exception as e:
        print(f"GIF creation failed: {e}")
        try:
            os.remove(gif_path)
        except Exception:
            pass
        return None
