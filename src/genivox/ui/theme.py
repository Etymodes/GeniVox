"""Visual constants and the application-wide dark theme."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

COLORS = {
    "background": "#0B1018",
    "sidebar": "#0D1420",
    "surface": "#111A27",
    "surface_alt": "#162131",
    "surface_hover": "#1C2A3D",
    "border": "#26364A",
    "border_strong": "#36506C",
    "text": "#E8EEF7",
    "text_muted": "#92A3B8",
    "primary": "#5B8CFF",
    "primary_hover": "#76A0FF",
    "cyan": "#45D4D0",
    "green": "#50CF91",
    "orange": "#FFB45E",
    "red": "#FF6E78",
    "purple": "#B58CFF",
}


STYLE_SHEET = f"""
QWidget {{
    color: {COLORS['text']};
    background: transparent;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {COLORS['background']};
}}
QToolTip {{
    color: {COLORS['text']};
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border_strong']};
    padding: 5px;
}}
QFrame#sidebar {{
    background: {COLORS['sidebar']};
    border-right: 1px solid {COLORS['border']};
}}
QFrame#topBar {{
    background: {COLORS['background']};
    border-bottom: 1px solid {COLORS['border']};
}}
QFrame#card, QFrame#panel {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}
QFrame#metricCard {{
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 9px;
}}
QLabel#appMark {{
    color: white;
    background: {COLORS['primary']};
    border-radius: 8px;
    font-size: 17px;
    font-weight: 700;
}}
QLabel#appName {{
    color: white;
    font-size: 18px;
    font-weight: 700;
}}
QLabel#pageTitle {{
    color: white;
    font-size: 24px;
    font-weight: 700;
}}
QLabel#sectionTitle {{
    color: white;
    font-size: 15px;
    font-weight: 650;
}}
QLabel#muted, QLabel#pageSubtitle, QLabel#metricLabel {{
    color: {COLORS['text_muted']};
}}
QLabel#metricValue {{
    color: white;
    font-size: 22px;
    font-weight: 700;
}}
QLabel#chip {{
    color: {COLORS['cyan']};
    background: #153039;
    border: 1px solid #24515C;
    border-radius: 9px;
    padding: 2px 8px;
}}
QListWidget#navigation {{
    background: transparent;
    border: none;
    outline: none;
    padding: 4px 8px;
}}
QListWidget#navigation::item {{
    color: {COLORS['text_muted']};
    border-radius: 7px;
    padding: 10px 12px;
    margin: 2px 0;
}}
QListWidget#navigation::item:hover {{
    color: {COLORS['text']};
    background: {COLORS['surface_hover']};
}}
QListWidget#navigation::item:selected {{
    color: white;
    background: #213657;
    border-left: 3px solid {COLORS['primary']};
}}
QPushButton {{
    color: {COLORS['text']};
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 7px;
    min-height: 32px;
    padding: 0 13px;
}}
QPushButton:hover {{
    background: {COLORS['surface_hover']};
    border-color: #4C6E91;
}}
QPushButton:pressed {{
    background: #0F1723;
}}
QPushButton:disabled {{
    color: #58697D;
    background: #121A24;
    border-color: #233041;
}}
QPushButton#primaryButton {{
    color: white;
    background: {COLORS['primary']};
    border-color: {COLORS['primary']};
    font-weight: 650;
}}
QPushButton#primaryButton:hover {{
    background: {COLORS['primary_hover']};
    border-color: {COLORS['primary_hover']};
}}
QPushButton#dangerButton {{
    color: {COLORS['red']};
    background: #2B1820;
    border-color: #63313A;
}}
QPushButton#recordButton:checked {{
    color: white;
    background: #C74957;
    border-color: {COLORS['red']};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    color: {COLORS['text']};
    background: #0D1520;
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    min-height: 30px;
    padding: 2px 8px;
    selection-background-color: {COLORS['primary']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS['primary']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    color: {COLORS['text']};
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border_strong']};
    selection-background-color: #28436B;
}}
QTableWidget, QTableView, QTreeWidget {{
    color: {COLORS['text']};
    background: #0D1520;
    alternate-background-color: #101B29;
    border: 1px solid {COLORS['border']};
    border-radius: 7px;
    gridline-color: {COLORS['border']};
    outline: none;
}}
QTableWidget::item, QTableView::item, QTreeWidget::item {{
    padding: 6px;
}}
QTableWidget::item:selected, QTableView::item:selected,
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: #28436B;
}}
QHeaderView::section {{
    color: {COLORS['text_muted']};
    background: {COLORS['surface_alt']};
    border: none;
    border-right: 1px solid {COLORS['border']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 8px;
    font-weight: 650;
}}
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    top: -1px;
}}
QTabBar::tab {{
    color: {COLORS['text_muted']};
    background: transparent;
    border-bottom: 2px solid transparent;
    padding: 9px 15px;
}}
QTabBar::tab:selected {{
    color: white;
    border-bottom-color: {COLORS['primary']};
}}
QGroupBox {{
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 13px;
    padding-top: 12px;
    font-weight: 650;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}}
QCheckBox, QRadioButton {{
    spacing: 7px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
}}
QCheckBox::indicator {{
    background: #0D1520;
    border: 1px solid {COLORS['border_strong']};
    border-radius: 4px;
}}
QCheckBox::indicator:checked {{
    background: {COLORS['primary']};
    border-color: {COLORS['primary']};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: #26364A;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {COLORS['primary']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: white;
    border: 2px solid {COLORS['primary']};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}}
QProgressBar {{
    color: {COLORS['text']};
    background: #0D1520;
    border: 1px solid {COLORS['border']};
    border-radius: 5px;
    text-align: center;
    min-height: 17px;
}}
QProgressBar::chunk {{
    background: {COLORS['primary']};
    border-radius: 4px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #354861;
    border-radius: 4px;
    min-height: 28px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background: {COLORS['border']};
    width: 1px;
    height: 1px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the GeniVox palette and stylesheet to an application."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["background"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0D1520"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["surface_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["primary"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(STYLE_SHEET)
