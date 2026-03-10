"""
main.py — single entry point for Fourier Circle Drawing.

Usage:
    python main.py                        # launch GUI (default)
    python main.py --gui                  # launch GUI explicitly
    python main.py --svg path/to/file.svg # run animation directly on an SVG
    python main.py --coeffs 150           # start GUI with a custom detail level
"""

import argparse
import os
import sys
import datetime
import cv2

from PyQt5.QtWidgets import (
    QWidget, QApplication, QHBoxLayout, QVBoxLayout, QFormLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QGroupBox,
    QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QIcon

from SVG import bitmap_to_contour_svg
import fft

# ---------------------------------------------------------------------------
# Catppuccin Mocha dark theme
# ---------------------------------------------------------------------------
DARK_STYLE = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}
QSpinBox {
    background: #313244;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 3px 6px;
    min-width: 68px;
}
QSpinBox:focus {
    border-color: #89b4fa;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton#btn_anim {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border: none;
    font-size: 14px;
    padding: 12px;
}
QPushButton#btn_anim:hover {
    background-color: #94e2d5;
}
QPushButton#btn_save_preview {
    background-color: #45475a;
    color: #cdd6f4;
    border: 1px solid #585b70;
}
QPushButton#btn_save_preview:hover {
    background-color: #585b70;
    border-color: #a6adc8;
}
QPushButton#btn_save_preview:disabled,
QPushButton#btn_download_gif:disabled {
    background-color: #1e1e2e;
    color: #45475a;
    border: 1px solid #313244;
}
QPushButton#btn_download_gif {
    background-color: #cba6f7;
    color: #1e1e2e;
    border: none;
    font-weight: bold;
    padding: 10px;
}
QPushButton#btn_download_gif:hover {
    background-color: #b4befe;
}
QLabel#status_label {
    color: #a6adc8;
    font-size: 11px;
    padding: 4px 2px;
    border-top: 1px solid #313244;
}
"""

# Maximum side length (px) before prompting the user about performance
_MAX_SIDE_PX = 4000
# Minimum side length (px) to ensure meaningful contours
_MIN_SIDE_PX = 32


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class InteractiveWindow(QWidget):
    MAX_COEFFS = 300  # 100 % == 300 Fourier coefficients

    def __init__(self, init_coeffs=200):
        super().__init__()
        self.fname             = None
        self.svg_name          = None
        self.preview_bbox      = None
        self.preview_strokes   = None
        self._preview_pixmap   = None   # last rendered pixmap (for PNG export)
        self._temp_gif_path    = None   # path to the latest temp GIF recording

        # Preview / animation canvas size
        self.anim_view_w = 1200
        self.anim_view_h = 800

        self.setWindowTitle("Fourier Circle Drawing")
        self.resize(1570, 880)
        self.setStyleSheet(DARK_STYLE)

        _icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))

        # ── Root layout ──────────────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # ── Parameters group ──────────────────────────────────────────────
        param_group = QGroupBox("Parameters")
        param_form  = QFormLayout()
        param_form.setSpacing(6)
        param_form.setLabelAlignment(Qt.AlignLeft)

        # Detail level: slider (0–300 coefficients) shown as 0–100 %
        init_pct = self._coeffs_to_pct(min(init_coeffs, self.MAX_COEFFS))
        self.label_fft  = QLabel(f"Detail Level:  {init_pct}%")
        self.slider_fft = QSlider(Qt.Horizontal)
        self.slider_fft.setRange(0, self.MAX_COEFFS)
        self.slider_fft.setValue(min(init_coeffs, self.MAX_COEFFS))
        self.slider_fft.setTickInterval(30)
        self.slider_fft.setToolTip(
            "Fourier coefficients used for reconstruction.\n"
            "Higher % = more detail.  Lower % = smoother / more abstract."
        )
        self.spin_fft = QSpinBox()
        self.spin_fft.setRange(0, 100)
        self.spin_fft.setSuffix(" %")
        self.spin_fft.setValue(init_pct)
        self.spin_fft.setToolTip("Type a value or use ↑↓ for fine adjustment.")

        fft_row = QHBoxLayout()
        fft_row.addWidget(self.slider_fft, 1)
        fft_row.addWidget(self.spin_fft)

        # Preview / animation scale
        self.label_scale  = QLabel("Scale:  100%")
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(10, 500)
        self.slider_scale.setValue(100)
        self.slider_scale.setToolTip("Zoom the preview and animation output.")
        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(10, 500)
        self.spin_scale.setSuffix(" %")
        self.spin_scale.setValue(100)
        self.spin_scale.setToolTip("Type a value or use ↑↓ for fine adjustment.")

        scale_row = QHBoxLayout()
        scale_row.addWidget(self.slider_scale, 1)
        scale_row.addWidget(self.spin_scale)

        param_form.addRow(self.label_fft)
        param_form.addRow(fft_row)
        param_form.addRow(QLabel(""))
        param_form.addRow(self.label_scale)
        param_form.addRow(scale_row)
        param_group.setLayout(param_form)
        left_panel.addWidget(param_group)

        # ── Actions group ─────────────────────────────────────────────────
        action_group  = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        action_layout.setSpacing(8)

        btn_load = QPushButton("① Load Image")
        btn_gen  = QPushButton("② Generate Preview")

        self.btn_save_preview = QPushButton("   Save Preview as PNG")
        self.btn_save_preview.setObjectName("btn_save_preview")
        self.btn_save_preview.setEnabled(False)

        btn_anim = QPushButton("③ Start Animation  (Pygame)")
        btn_anim.setObjectName("btn_anim")

        self.btn_download_gif = QPushButton("   Download GIF Recording")
        self.btn_download_gif.setObjectName("btn_download_gif")
        self.btn_download_gif.setEnabled(False)
        self.btn_download_gif.setToolTip(
            "Download the GIF recorded during the last animation.\n"
            "If you dismiss this dialog the recording will be discarded."
        )

        btn_load.clicked.connect(self.loadImg)
        btn_gen.clicked.connect(self.genResult)
        self.btn_save_preview.clicked.connect(self.savePreview)
        btn_anim.clicked.connect(self.runAnim)
        self.btn_download_gif.clicked.connect(self.downloadGif)

        action_layout.addWidget(btn_load)
        action_layout.addWidget(btn_gen)
        action_layout.addWidget(self.btn_save_preview)
        action_layout.addWidget(btn_anim)
        action_layout.addWidget(self.btn_download_gif)
        action_group.setLayout(action_layout)
        left_panel.addWidget(action_group)

        # ── Hint group ────────────────────────────────────────────────────
        hint_group  = QGroupBox("Animation Shortcuts")
        hint_layout = QVBoxLayout()
        hint_layout.setSpacing(2)
        for text in ("S  — save screenshot (timestamped)", "ESC — exit animation"):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
            hint_layout.addWidget(lbl)
        hint_group.setLayout(hint_layout)
        left_panel.addWidget(hint_group)

        # ── Status label ──────────────────────────────────────────────────
        self.status_label = QLabel("Ready — load an image to begin.")
        self.status_label.setObjectName("status_label")
        self.status_label.setWordWrap(True)
        left_panel.addWidget(self.status_label)
        left_panel.addStretch()

        # ── Preview canvas ────────────────────────────────────────────────
        preview_group  = QGroupBox(f"Preview  ({self.anim_view_w} × {self.anim_view_h})")
        preview_layout = QVBoxLayout()
        self.view = QLabel("Load an image, then click  'Generate Preview'")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setFixedSize(self.anim_view_w, self.anim_view_h)
        self.view.setStyleSheet(
            "background: #11111b; border: 1px solid #45475a;"
            "border-radius: 5px; color: #585b70; font-size: 15px;"
        )
        preview_layout.addWidget(self.view)
        preview_group.setLayout(preview_layout)

        root.addLayout(left_panel, 1)
        root.addWidget(preview_group)

        # ── Signal connections (bidirectional slider ↔ spinbox) ───────────
        self.slider_fft.valueChanged.connect(self._on_fft_slider)
        self.spin_fft.valueChanged.connect(self._on_fft_spin)
        self.slider_scale.valueChanged.connect(self._on_scale_slider)
        self.spin_scale.valueChanged.connect(self._on_scale_spin)

    # ── Conversion helpers ────────────────────────────────────────────────
    def _coeffs_to_pct(self, v: int) -> int:
        return round(v / self.MAX_COEFFS * 100)

    def _pct_to_coeffs(self, pct: int) -> int:
        return round(pct / 100 * self.MAX_COEFFS)

    # ── Slider / spinbox sync ─────────────────────────────────────────────
    def _on_fft_slider(self, v: int):
        pct = self._coeffs_to_pct(v)
        self.spin_fft.blockSignals(True)
        self.spin_fft.setValue(pct)
        self.spin_fft.blockSignals(False)
        self.label_fft.setText(f"Detail Level:  {pct}%")
        self.livePreview()

    def _on_fft_spin(self, pct: int):
        v = self._pct_to_coeffs(pct)
        self.slider_fft.blockSignals(True)
        self.slider_fft.setValue(v)
        self.slider_fft.blockSignals(False)
        self.label_fft.setText(f"Detail Level:  {pct}%")
        self.livePreview()

    def _on_scale_slider(self, v: int):
        self.spin_scale.blockSignals(True)
        self.spin_scale.setValue(v)
        self.spin_scale.blockSignals(False)
        self.label_scale.setText(f"Scale:  {v}%")
        self.livePreview()

    def _on_scale_spin(self, v: int):
        self.slider_scale.blockSignals(True)
        self.slider_scale.setValue(v)
        self.slider_scale.blockSignals(False)
        self.label_scale.setText(f"Scale:  {v}%")
        self.livePreview()

    # ── Status helper ─────────────────────────────────────────────────────
    def setStatus(self, msg: str):
        self.status_label.setText(msg)
        QApplication.processEvents()

    # ── Image validation ─────────────────────────────────────────────────
    def _validate_image(self, path: str) -> bool:
        """Read the image with OpenCV and check basic requirements.
        Returns True if valid; shows a dialog and returns False otherwise."""
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(
                self, "Invalid Image",
                "Cannot read this file as an image.\n"
                "Please choose a valid PNG, JPG, or BMP file."
            )
            return False

        h, w = img.shape[:2]

        if w < _MIN_SIDE_PX or h < _MIN_SIDE_PX:
            QMessageBox.warning(
                self, "Image Too Small",
                f"Image is too small ({w} × {h} px).\n"
                f"Minimum recommended size: {_MIN_SIDE_PX} × {_MIN_SIDE_PX} px.\n\n"
                "Very small images produce too few contour points for meaningful results."
            )
            return False

        if w > _MAX_SIDE_PX or h > _MAX_SIDE_PX:
            reply = QMessageBox.question(
                self, "Large Image",
                f"This image is large ({w} × {h} px) and may take a while to process.\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return False

        # Warn if the image is almost entirely dark (tool expects dark lines on light bg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if gray.mean() < 30:
            QMessageBox.warning(
                self, "Very Dark Image",
                f"The image is almost entirely dark (mean brightness: {gray.mean():.0f}/255).\n\n"
                "This tool works best with dark outlines on a white or light background.\n"
                "Tip: try inverting the image colours before loading."
            )
            # Not a hard block — the user may still proceed

        return True

    # ── Main actions ──────────────────────────────────────────────────────
    def loadImg(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not fname:
            return
        if not self._validate_image(fname):
            return

        self.fname           = fname
        self.svg_name        = None
        self.preview_strokes = None
        self.preview_bbox    = None
        self._preview_pixmap = None
        self.btn_save_preview.setEnabled(False)

        pix = QPixmap(fname).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)
        self.setStatus(f"Loaded: {os.path.basename(fname)}\n"
                       "Click  'Generate Preview'  to process.")

    def genResult(self):
        if not self.fname:
            self.setStatus("No image loaded. Please load an image first.")
            return
        os.makedirs("./images", exist_ok=True)
        self.setStatus("Processing: image → SVG contours → FFT …")
        self.svg_name = "./images/temp.svg"
        bitmap_to_contour_svg(self.fname, self.svg_name)

        res = fft.get_reconstructed_points(self.svg_name, self.slider_fft.value())
        strokes, bbox = res if isinstance(res, tuple) else (res, None)

        if not strokes:
            self.setStatus("No contours found.")
            QMessageBox.warning(
                self, "No Contours Found",
                "No drawable contours were found in this image.\n\n"
                "This usually means the image:\n"
                "  • Is all white / very light with no dark regions\n"
                "  • Has no clear edges or outlines\n\n"
                "Tips:\n"
                "  • Use an image with dark lines on a white background\n"
                "  • Try a high-contrast outline drawing or silhouette\n"
                "  • Try adjusting image contrast before loading"
            )
            return

        self.preview_strokes = strokes
        self.preview_bbox    = bbox
        self.livePreview()
        self.setStatus(
            "Preview ready.  Adjust parameters or start the animation.\n"
            "In the animation window: [S] screenshot  |  ESC exit"
        )

    def livePreview(self):
        if not self.svg_name:
            return

        res        = fft.get_reconstructed_points(self.svg_name, self.slider_fft.value())
        strokes    = res[0] if isinstance(res, tuple) else res
        scale_user = self.slider_scale.value() / 100.0

        pix = QPixmap(self.anim_view_w, self.anim_view_h)
        pix.fill(QColor(17, 17, 27))

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(137, 180, 250), 2))

        cx, cy = self.anim_view_w / 2.0, self.anim_view_h / 2.0

        if self.preview_bbox and len(self.preview_bbox) == 4:
            minx, maxx, miny, maxy = self.preview_bbox
            bbox_cx     = (minx + maxx) / 2.0
            bbox_cy     = (miny + maxy) / 2.0
            svg_w       = max(1.0, maxx - minx)
            svg_h       = max(1.0, maxy - miny)
            base_fit    = min(self.anim_view_w / svg_w, self.anim_view_h / svg_h) * 0.8
            final_scale = base_fit * scale_user
        else:
            bbox_cx, bbox_cy = 0.0, 0.0
            final_scale = scale_user

        for s in strokes:
            if not s or len(s) < 2:
                continue
            for i in range(len(s) - 1):
                x1 = cx + (s[i][0]      - bbox_cx) * final_scale
                y1 = cy + (s[i][1]      - bbox_cy) * final_scale
                x2 = cx + (s[i + 1][0] - bbox_cx) * final_scale
                y2 = cy + (s[i + 1][1] - bbox_cy) * final_scale
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        painter.end()
        self._preview_pixmap = pix
        self.view.setPixmap(pix)
        self.btn_save_preview.setEnabled(True)

    def savePreview(self):
        """Export the current static preview as a PNG file."""
        if self._preview_pixmap is None:
            return
        ts           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"preview_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preview Image", default_name,
            "PNG Images (*.png);;All Files (*)"
        )
        if path:
            self._preview_pixmap.save(path, "PNG")
            self.setStatus(f"Preview saved: {os.path.basename(path)}")

    def runAnim(self):
        if not self.svg_name:
            self.setStatus("Please generate a preview first.")
            QMessageBox.warning(self, "Warning", "Please generate a preview first.")
            return

        # Discard previous recording before starting a new one.
        self._cleanup_temp_gif()

        self.setStatus("Animation running…  (close the Pygame window to stop)")
        gif_path = fft.draw(
            self.svg_name,
            self.slider_fft.value(),
            self.slider_scale.value() / 100.0,
        )

        if gif_path and os.path.exists(gif_path):
            self._temp_gif_path = gif_path
            self.btn_download_gif.setEnabled(True)
            self.setStatus(
                "Animation ended.  GIF recording is ready.\n"
                "Click 'Download GIF Recording' to save, or start a new animation to discard it."
            )
        else:
            self.setStatus("Animation ended.")

    def downloadGif(self):
        """Save the GIF recording to recordings/ and discard the temp file."""
        if not self._temp_gif_path or not os.path.exists(self._temp_gif_path):
            return
        import shutil
        rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(rec_dir, f"fourier_{ts}.gif")
        shutil.copy2(self._temp_gif_path, dest)
        self._cleanup_temp_gif()
        self.setStatus(f"GIF saved → recordings/fourier_{ts}.gif")

    # ── Temp GIF lifecycle ────────────────────────────────────────────────
    def _cleanup_temp_gif(self):
        """Delete the temp GIF file and reset the download button."""
        if self._temp_gif_path:
            try:
                if os.path.exists(self._temp_gif_path):
                    os.remove(self._temp_gif_path)
            except Exception:
                pass
            self._temp_gif_path = None
        self.btn_download_gif.setEnabled(False)

    def closeEvent(self, event):
        """Clean up the temp GIF when the application window closes."""
        self._cleanup_temp_gif()
        event.accept()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_gui(init_coeffs: int = 50):
    app  = QApplication(sys.argv)
    tool = InteractiveWindow(init_coeffs=init_coeffs)
    tool.show()
    return app.exec()


def run_svg(svg_path: str, n_coeffs: int = None, user_scale: float = 1.0):
    if not os.path.exists(svg_path):
        print(f"SVG not found: {svg_path}")
        return 2
    fft.draw(svg_path, n_coeffs=n_coeffs, user_scale=user_scale)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fourier Circle Drawing")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true", help="Launch the GUI (default)")
    group.add_argument("--svg", metavar="PATH",      help="Run animation directly on an SVG file")
    parser.add_argument("--coeffs", type=int,   default=50,  help="Fourier coefficients to keep (0–300)")
    parser.add_argument("--scale",  type=float, default=1.0, help="Scale multiplier (1.0 = 100 %%)")

    args = parser.parse_args()

    if args.svg:
        return run_svg(args.svg, n_coeffs=args.coeffs, user_scale=args.scale)
    return run_gui(init_coeffs=args.coeffs)


if __name__ == "__main__":
    sys.exit(main())
