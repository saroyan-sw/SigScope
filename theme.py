"""
sigscope.theme
==============

Modern, minimal DARK theme for the whole app.

Provides:
    apply_theme(app)          -> global QSS + font + pyqtgraph config
    style_plot(plotwidget)    -> sleek axes/grid/legend on a PlotWidget
    style_imageview(iv, cmap) -> declutter + colormap an ImageView
    C                         -> palette dict (use for pens / line colours)
"""

from __future__ import annotations
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
C = {
    "bg":        "#0d1014",   # app background
    "surface":   "#161b22",   # cards / panels
    "surface2":  "#1c222c",   # inputs
    "surface3":  "#232b37",   # hover
    "border":    "#2a323f",
    "text":      "#e6edf3",
    "muted":     "#8b94a3",
    "faint":     "#5b6472",
    "accent":    "#4f8ff7",   # primary blue
    "accent_hi": "#6ba4ff",
    "accent_dim":"#2b4d7a",
    "teal":      "#2dd4bf",
    "good":      "#3fb950",
    "warn":      "#d29922",
    "danger":    "#f0616d",
    # plot line palette (colour-blind friendly, vivid on dark)
    "lines": ["#4f8ff7", "#f0616d", "#2dd4bf", "#f5b942",
              "#b57cff", "#ff8f4d", "#4ddbff", "#7ee787"],
}


# --------------------------------------------------------------------------
# global stylesheet
# --------------------------------------------------------------------------
def _qss():
    c = C
    return f"""
    * {{
        font-family: "Inter", "Segoe UI", "SF Pro Display", "Ubuntu", sans-serif;
        font-size: 13px;
        color: {c['text']};
        outline: none;
    }}
    QMainWindow, QWidget {{ background-color: {c['bg']}; }}

    QScrollArea {{ border: none; background: transparent; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}

    /* ---- section card ---- */
    QFrame#card {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 12px;
    }}
    QLabel#section {{
        color: {c['muted']};
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 0px;
    }}
    QLabel#note {{ color: {c['faint']}; font-size: 11px; }}
    QLabel#info {{ color: {c['muted']}; font-size: 12px; }}
    QLabel#appTitle {{ font-size: 16px; font-weight: 700; color: {c['text']}; }}
    QLabel#appSub  {{ font-size: 11px; color: {c['faint']}; }}
    QLabel#workspaceTitle {{ font-size: 22px; font-weight: 700; color: {c['text']}; }}
    QLabel#workspaceSub {{ font-size: 12px; color: {c['muted']}; }}
    QLabel#dataBadge {{
        background: {c['accent_dim']}; color: {c['accent_hi']};
        border: 1px solid {c['accent']}; border-radius: 11px;
        padding: 5px 11px; font-size: 10px; font-weight: 700;
    }}
    QLabel#panelHeading {{ font-size: 15px; font-weight: 700; color: {c['text']}; }}
    QLabel#panelDescription {{ font-size: 11px; color: {c['muted']}; }}
    QLabel#dataInfo {{
        background: {c['surface2']}; border: 1px solid {c['border']};
        border-radius: 8px; padding: 9px; color: {c['muted']};
    }}
    QLabel#selectionSummary {{ color: {c['teal']}; font-size: 11px; font-weight: 600; }}
    QLabel#vizTitle {{ font-size: 13px; font-weight: 700; color: {c['text']}; }}
    QLabel#plotEmpty {{
        background: {c['surface2']}; border: 1px dashed {c['border']};
        border-radius: 8px; padding: 30px; color: {c['faint']};
    }}
    QFrame#workspacePanel, QFrame#vizCard {{
        background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 11px;
    }}

    /* ---- buttons ---- */
    QPushButton {{
        background-color: {c['surface2']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 8px 12px;
        text-align: left;
    }}
    QPushButton:hover {{ background-color: {c['surface3']};
                         border-color: {c['accent_dim']}; }}
    QPushButton:pressed {{ background-color: {c['accent_dim']}; }}
    QPushButton#primary {{
        background-color: {c['accent']};
        border: 1px solid {c['accent']};
        color: #ffffff; font-weight: 600; text-align: center;
    }}
    QPushButton#primary:hover {{ background-color: {c['accent_hi']}; }}
    QPushButton#primary:pressed {{ background-color: {c['accent_dim']}; }}
    QPushButton#computeButton {{
        background: {c['accent']}; color: white; border: 1px solid {c['accent']};
        border-radius: 8px; padding: 11px; font-weight: 700; text-align: center;
    }}
    QPushButton#computeButton:hover {{ background: {c['accent_hi']}; }}
    QPushButton#computeButton:disabled {{
        background: {c['surface3']}; color: {c['faint']}; border-color: {c['border']};
    }}
    QPushButton#smallGhost {{
        background: transparent; color: {c['muted']}; padding: 5px 8px;
        text-align: center;
    }}
    QPushButton#smallGhost:hover {{ color: {c['text']}; border-color: {c['faint']}; }}
    QPushButton#ghost {{ background: transparent; color: {c['muted']};
                         text-align: center; }}
    QPushButton#ghost:hover {{ color: {c['danger']};
                               border-color: {c['danger']}; }}

    /* ---- inputs ---- */
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {c['surface2']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 6px 8px;
        min-height: 18px;
        selection-background-color: {c['accent']};
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {c['accent_dim']}; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {c['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {c['muted']};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['surface2']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        selection-background-color: {c['accent']};
        padding: 4px;
    }}
    QSpinBox::up-button, QSpinBox::down-button,
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 16px; border: none; background: transparent; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-bottom: 5px solid {c['muted']}; width:0;height:0; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-top: 5px solid {c['muted']}; width:0;height:0; }}

    /* ---- checkbox / radio ---- */
    QCheckBox, QRadioButton {{ spacing: 8px; color: {c['text']}; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 17px; height: 17px;
        border: 1px solid {c['border']};
        background: {c['surface2']};
    }}
    QCheckBox::indicator {{ border-radius: 5px; }}
    QRadioButton::indicator {{ border-radius: 9px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
        border-color: {c['accent']}; }}
    QCheckBox::indicator:checked {{
        background: {c['accent']}; border-color: {c['accent']};
        image: none; }}
    QRadioButton::indicator:checked {{
        background: {c['accent']}; border: 4px solid {c['surface2']};
        border-radius: 9px; }}

    /* ---- table ---- */
    QTableWidget {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        gridline-color: {c['border']};
    }}
    QHeaderView::section {{
        background-color: {c['surface2']};
        color: {c['muted']};
        border: none; border-right: 1px solid {c['border']};
        border-bottom: 1px solid {c['border']};
        padding: 6px; font-weight: 600; }}
    QTableWidget::item {{ padding: 4px; }}
    QTableWidget::item:selected {{ background: {c['accent_dim']}; }}
    QTreeWidget {{
        background: {c['surface2']}; border: 1px solid {c['border']};
        border-radius: 8px; padding: 5px; alternate-background-color: {c['surface']};
    }}
    QTreeWidget::item {{ padding: 4px; border-radius: 4px; }}
    QTreeWidget::item:hover {{ background: {c['surface3']}; }}
    QTreeWidget::item:selected {{ background: {c['accent_dim']}; }}
    QTreeWidget::item:disabled {{ color: {c['faint']}; }}

    QTabWidget::pane {{
        border: 1px solid {c['border']}; border-radius: 10px;
        background: {c['bg']}; top: -1px;
    }}
    QTabBar::tab {{
        background: {c['surface']}; color: {c['muted']};
        border: 1px solid {c['border']}; border-bottom: none;
        padding: 10px 18px; margin-right: 4px;
        border-top-left-radius: 8px; border-top-right-radius: 8px;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{
        background: {c['surface2']}; color: {c['text']};
        border-top: 2px solid {c['accent']};
    }}
    QTabBar::tab:hover:!selected {{ color: {c['text']}; }}

    /* ---- status bar / tooltip ---- */
    QStatusBar {{ background: {c['surface']}; color: {c['muted']};
                  border-top: 1px solid {c['border']}; }}
    QStatusBar::item {{ border: none; }}
    QToolTip {{ background-color: {c['surface3']}; color: {c['text']};
                border: 1px solid {c['border']}; border-radius: 6px;
                padding: 6px; }}

    /* ---- scrollbars ---- */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {c['surface3']};
        border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['faint']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {c['surface3']};
        border-radius: 5px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QMessageBox {{ background: {c['surface']}; }}
    QSplitter::handle {{ background: {c['bg']}; width: 9px; }}
    """


# --------------------------------------------------------------------------
# dock title-bar restyle (monkeypatch pyqtgraph's DockLabel)
# --------------------------------------------------------------------------
def _patch_dock_labels():
    from pyqtgraph.dockarea.Dock import DockLabel

    def updateStyle(self):
        r = "8px"
        if self.dim:
            fg, bg, border = C["faint"], C["surface"], C["border"]
        else:
            fg, bg, border = C["text"], C["surface2"], C["accent"]
        if self.orientation == "vertical":
            self.vStyle = f"""DockLabel {{
                background-color:{bg}; color:{fg};
                border-top-left-radius:{r}; border-bottom-left-radius:{r};
                border-top-right-radius:0px; border-bottom-right-radius:0px;
                border-width:0px; border-right:2px solid {border};
                padding-top:6px; padding-bottom:6px;
                font-size:{self.fontSize}; font-weight:600; }}"""
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = f"""DockLabel {{
                background-color:{bg}; color:{fg};
                border-top-left-radius:{r}; border-top-right-radius:{r};
                border-bottom-left-radius:0px; border-bottom-right-radius:0px;
                border-width:0px; border-bottom:2px solid {border};
                padding-left:10px; padding-right:10px; padding-top:5px;
                padding-bottom:5px;
                font-size:{self.fontSize}; font-weight:600; }}"""
            self.setStyleSheet(self.hStyle)

    DockLabel.updateStyle = updateStyle


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def apply_theme(app):
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True,
                        background=C["surface"], foreground=C["muted"])
    _patch_dock_labels()
    app.setStyle("Fusion")
    font = QtGui.QFont("Inter", 10)
    app.setFont(font)
    app.setStyleSheet(_qss())


def style_plot(pw, title=None):
    """Apply the sleek look to a PlotWidget."""
    pw.setBackground(C["surface"])
    pi = pw.getPlotItem()
    pi.showGrid(x=True, y=True, alpha=0.12)
    for ax in ("left", "bottom"):
        a = pi.getAxis(ax)
        a.setPen(pg.mkPen(C["border"], width=1))
        a.setTextPen(pg.mkPen(C["muted"]))
        a.setStyle(tickLength=-4)
    for ax in ("top", "right"):
        pi.showAxis(ax, False)
    if title:
        pi.setTitle(title, color=C["muted"], size="11pt")
    return pw


def style_imageview(iv, cmap="viridis"):
    """Declutter an ImageView and give it a modern colormap + dark chrome."""
    # hide the busy default buttons
    try:
        iv.ui.roiBtn.hide()
        iv.ui.menuBtn.hide()
    except Exception:
        pass
    # dark backgrounds (view may be a ViewBox or a PlotItem)
    view = iv.getView()
    try:
        vb = view.getViewBox() if isinstance(view, pg.PlotItem) else view
        vb.setBackgroundColor(C["surface"])
    except Exception:
        pass
    try:
        iv.ui.histogram.setBackground(C["surface"])
        iv.ui.histogram.axis.setPen(pg.mkPen(C["border"]))
        iv.ui.histogram.axis.setTextPen(pg.mkPen(C["muted"]))
    except Exception:
        pass
    # colormap
    try:
        iv.setColorMap(pg.colormap.get(cmap))
    except Exception:
        try:
            iv.setColorMap(pg.colormap.getFromMatplotlib(cmap))
        except Exception:
            pass
    # style plot axes of the underlying PlotItem
    vb_item = iv.getView()
    if isinstance(vb_item, pg.PlotItem):
        for ax in ("left", "bottom"):
            a = vb_item.getAxis(ax)
            a.setPen(pg.mkPen(C["border"], width=1))
            a.setTextPen(pg.mkPen(C["muted"]))
    return iv
