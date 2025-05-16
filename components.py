# components.py
from PyQt6.QtWidgets import QPushButton, QLabel, QProgressBar, QFrame, QGraphicsDropShadowEffect, QSlider
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtCore import Qt, QSize
import numpy as np

COLORS = {
    "primary": "#1DB954",
    "secondary": "#4A90E2",
    "danger": "#FF5252",
    "background": "#121212",
    "card_bg": "#212121",
    "text_primary": "#FFFFFF",
    "text_secondary": "#B3B3B3",
    "slider_groove": "#535353",
    "slider_handle": "#1DB954",
    "progress_bg": "#535353",
    "accent": "#9B59B6",
    "hover": "#333333",
}

class RoundedButton(QPushButton):
    def __init__(self, text, parent=None, color=COLORS["primary"], icon=None, size=(36, 120)):
        super().__init__(text, parent)
        self.setFont(QFont("Segoe UI", 10))
        self.setMinimumHeight(size[0])
        self.setMinimumWidth(size[1])
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.color = color

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {COLORS["text_primary"]};
                border-radius: 18px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color, 10)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(color, 10)};
            }}
        """)

    def _lighten_color(self, hex_color, percent):
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c * (1 + percent / 100))) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _darken_color(self, hex_color, percent):
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        rgb = tuple(max(0, int(c * (1 - percent / 100))) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

# Se quiser, também mova StyledLabel, CardFrame, etc. para cá.
