"""SigScope desktop application: metric computation, CSV export and plots."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import traceback

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

try:
    import core, normalize, theme
    from theme import C
except ModuleNotFoundError:
    from sigscope import core, normalize, theme
    from sigscope.theme import C


METRIC_REQUIREMENTS = {
    "amplitude": "amplitude",
    "circular_diff": "phase",
    "circular_raw": "phase",
    "pulsepair": "phase",
    "lag_decay": "phase",
    "spanfactor": "phase",
    "linearity": "phase",
    "chirp": "phase",
    "spectral": "phase",
}

OUTPUT_REQUIREMENTS = {
    # Pulse-pair power is useful with amplitude-only data; its coherence and
    # Doppler outputs need phase.
    ("pulsepair", "pp_power"): "amplitude",
}


def decode_channel_matrix(array):
    """Decode a numeric matrix, including 3-channel poured-value storage.

    A poured value is reconstructed as C0 + C1 + C2. For example,
    [255, 255, 190] becomes 700. The original H×W×3 array is returned as the
    optional visualization image.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float32), None
    if arr.ndim == 3 and arr.shape[-1] == 3:
        decoded = np.sum(arr, axis=-1, dtype=np.float32)
        return decoded, arr
    if arr.ndim == 3 and arr.shape[0] == 3:
        visual = np.moveaxis(arr, 0, -1)
        decoded = np.sum(visual, axis=-1, dtype=np.float32)
        return decoded, visual
    raise ValueError(
        "Expected a 2-D matrix or a 3-channel poured matrix shaped "
        f"[height, width, 3] / [3, height, width]; received {arr.shape}."
    )


def decode_stacked_complex(array):
    """Split common amplitude+phase stack layouts and decode each channel."""
    arr = np.asarray(array)
    if np.iscomplexobj(arr):
        if arr.ndim != 2:
            raise ValueError(
                f"A native complex array must be 2-D; received {arr.shape}."
            )
        return (
            np.abs(arr).astype(np.float32),
            np.angle(arr).astype(np.float32),
            None,
            None,
            "native complex",
        )

    amplitude_source = phase_source = None
    layout = ""
    if arr.ndim == 3 and arr.shape[-1] == 2:
        amplitude_source, phase_source = arr[..., 0], arr[..., 1]
        layout = "amplitude/phase in last axis"
    elif arr.ndim == 3 and arr.shape[0] == 2:
        amplitude_source, phase_source = arr[0], arr[1]
        layout = "amplitude/phase in first axis"
    elif arr.ndim == 3 and arr.shape[-1] == 6:
        amplitude_source, phase_source = arr[..., :3], arr[..., 3:]
        layout = "three poured amplitude channels + three poured phase channels"
    elif arr.ndim == 3 and arr.shape[0] == 6:
        moved = np.moveaxis(arr, 0, -1)
        amplitude_source, phase_source = moved[..., :3], moved[..., 3:]
        layout = "first-axis poured amplitude/phase channels"
    elif arr.ndim == 4 and arr.shape[0] == 2:
        amplitude_source, phase_source = arr[0], arr[1]
        layout = "first-axis amplitude/phase stack with poured channels"
    elif arr.ndim == 4 and arr.shape[-2:] == (2, 3):
        amplitude_source, phase_source = arr[..., 0, :], arr[..., 1, :]
        layout = "last-axis amplitude/phase stack with poured channels"
    elif arr.ndim == 4 and arr.shape[-2:] == (3, 2):
        amplitude_source, phase_source = arr[..., :, 0], arr[..., :, 1]
        layout = "interleaved poured amplitude/phase channels"
    else:
        raise ValueError(
            "Complex input must be a native 2-D complex matrix or a real "
            "amplitude+phase stack. Supported stack shapes include [H,W,2], "
            "[2,H,W], [H,W,6], [6,H,W], and [2,H,W,3]. "
            f"Received {arr.shape}."
        )

    amplitude, amplitude_visual = decode_channel_matrix(amplitude_source)
    phase, phase_visual = decode_channel_matrix(phase_source)
    if amplitude.shape != phase.shape:
        raise ValueError(
            f"Decoded amplitude shape {amplitude.shape} does not match phase {phase.shape}."
        )
    return amplitude, phase, amplitude_visual, phase_visual, layout


class BatchWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, z, axis, selections):
        super().__init__()
        self.z = z
        self.axis = axis
        self.selections = selections

    def run(self):
        try:
            series = {}
            for stat_key, outputs in self.selections.items():
                result = core.STATS[stat_key]["gfn"](self.z, self.axis)
                for output in outputs:
                    series[f"{stat_key}.{output}"] = np.asarray(
                        result[output], dtype=np.float64
                    )
            self.done.emit(series)
        except Exception:
            self.failed.emit(traceback.format_exc())


class VisualizationCard(QtWidgets.QFrame):
    remove_requested = QtCore.pyqtSignal(object)

    def __init__(self, number):
        super().__init__()
        self.setObjectName("vizCard")
        self.number = number
        self.series = {}
        self.index_label = "index"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"Visualization {number}")
        title.setObjectName("vizTitle")
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.setMinimumWidth(230)
        self.metric_combo.currentIndexChanged.connect(self.refresh)
        remove = QtWidgets.QPushButton("Remove")
        remove.setObjectName("smallGhost")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.metric_combo)
        header.addWidget(remove)
        layout.addLayout(header)

        self.plot = pg.PlotWidget()
        theme.style_plot(self.plot)
        self.plot.setMinimumHeight(260)
        layout.addWidget(self.plot, 1)
        self.empty_label = QtWidgets.QLabel(
            "Choose a computed metric above. If the list is empty, compute metrics first."
        )
        self.empty_label.setObjectName("plotEmpty")
        self.empty_label.setAlignment(QtCore.Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)

    def set_series(self, series, index_label):
        selected = self.metric_combo.currentData()
        self.series = series
        self.index_label = index_label
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItem("Choose metric to show…", None)
        for name in sorted(series):
            self.metric_combo.addItem(name, name)
        if selected in series:
            self.metric_combo.setCurrentIndex(self.metric_combo.findData(selected))
        self.metric_combo.blockSignals(False)
        self.refresh()

    def refresh(self):
        self.plot.clear()
        name = self.metric_combo.currentData()
        self.empty_label.setVisible(not name)
        self.plot.setVisible(bool(name))
        if not name or name not in self.series:
            return
        values = self.series[name]
        self.plot.plot(values, pen=pg.mkPen(C["accent"], width=1.7))
        self.plot.setLabel("bottom", self.index_label, color=C["muted"])
        self.plot.setLabel("left", name, color=C["muted"])
        self.plot.getPlotItem().setTitle(name, color=C["text"], size="11pt")


class SigScope(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SigScope — metric workspace")
        self.resize(1600, 950)
        self.setMinimumSize(1120, 700)

        self._amp_raw = None
        self._phase_raw = None
        self._amp_visual = None
        self._phase_visual = None
        self.z = None
        self.amp = None
        self.phase = None
        self.source_names = {}
        self.source_notes = {}
        self.computed_series = {}
        self.pending_csv = None
        self.worker = None
        self.visualizations = []
        self._viz_counter = 0

        self._build_ui()
        self._update_data_state()

    def _build_ui(self):
        wrapper = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(wrapper)
        outer.setContentsMargins(14, 12, 14, 10)
        outer.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("SigScope")
        title.setObjectName("workspaceTitle")
        subtitle = QtWidgets.QLabel(
            "Load signal channels, choose metrics, export CSV, and build your own plots."
        )
        subtitle.setObjectName("workspaceSub")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.data_badge = QtWidgets.QLabel("NO DATA")
        self.data_badge.setObjectName("dataBadge")
        header.addWidget(self.data_badge)
        outer.addLayout(header)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_compute_tab(), "1  Compute & Export")
        self.tabs.addTab(self._build_visualization_tab(), "2  Stat Visualizations")
        outer.addWidget(self.tabs, 1)
        self.setCentralWidget(wrapper)
        self.statusBar().showMessage("Load amplitude, phase, or a complex matrix to begin.")

    def _panel(self, title, description=""):
        frame = QtWidgets.QFrame()
        frame.setObjectName("workspacePanel")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(9)
        label = QtWidgets.QLabel(title)
        label.setObjectName("panelHeading")
        layout.addWidget(label)
        if description:
            note = QtWidgets.QLabel(description)
            note.setObjectName("panelDescription")
            note.setWordWrap(True)
            layout.addWidget(note)
        return frame, layout

    def _build_compute_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls = QtWidgets.QWidget()
        controls.setMinimumWidth(340)
        controls.setMaximumWidth(440)
        control_layout = QtWidgets.QVBoxLayout(controls)
        control_layout.setContentsMargins(0, 0, 8, 0)
        control_layout.setSpacing(10)

        data_panel, data_layout = self._panel(
            "1. Load data",
            "Load amplitude and phase separately, or load one native/stacked complex file. Three-channel poured matrices are decoded automatically.",
        )
        load_amp = QtWidgets.QPushButton("Load amplitude (.npy)")
        load_amp.clicked.connect(lambda: self._load("amplitude"))
        load_phase = QtWidgets.QPushButton("Load phase (.npy)")
        load_phase.clicked.connect(lambda: self._load("phase"))
        load_complex = QtWidgets.QPushButton("Load complex or amplitude+phase stack (.npy)")
        load_complex.setObjectName("primary")
        load_complex.clicked.connect(lambda: self._load("complex"))
        data_layout.addWidget(load_amp)
        data_layout.addWidget(load_phase)
        data_layout.addWidget(load_complex)
        self.axis_check = QtWidgets.QCheckBox("Axis 0 contains range bins")
        self.axis_check.setChecked(True)
        self.axis_check.toggled.connect(self._rebuild_signal)
        data_layout.addWidget(self.axis_check)
        self.data_info = QtWidgets.QLabel("No channels loaded")
        self.data_info.setObjectName("dataInfo")
        self.data_info.setWordWrap(True)
        data_layout.addWidget(self.data_info)
        clear_data = QtWidgets.QPushButton("Clear loaded data")
        clear_data.setObjectName("smallGhost")
        clear_data.clicked.connect(self._clear_data)
        data_layout.addWidget(clear_data)
        control_layout.addWidget(data_panel)

        metrics_panel, metrics_layout = self._panel(
            "2. Choose metrics",
            "Only metrics supported by the loaded channels are enabled. Tick every output you want in the CSV.",
        )
        quick = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("Select available")
        select_all.setObjectName("smallGhost")
        select_all.clicked.connect(lambda: self._set_all_metrics(True))
        clear_all = QtWidgets.QPushButton("Clear selection")
        clear_all.setObjectName("smallGhost")
        clear_all.clicked.connect(lambda: self._set_all_metrics(False))
        quick.addWidget(select_all)
        quick.addWidget(clear_all)
        metrics_layout.addLayout(quick)
        self.metric_tree = QtWidgets.QTreeWidget()
        self.metric_tree.setHeaderHidden(True)
        self.metric_tree.setRootIsDecorated(True)
        self.metric_tree.itemChanged.connect(self._metric_selection_changed)
        metrics_layout.addWidget(self.metric_tree, 1)
        self.selection_label = QtWidgets.QLabel("0 outputs selected")
        self.selection_label.setObjectName("selectionSummary")
        metrics_layout.addWidget(self.selection_label)
        control_layout.addWidget(metrics_panel, 1)

        export_panel, export_layout = self._panel(
            "3. Compute and export",
            "The CSV contains one row per range bin (row analysis) or time sample (column analysis).",
        )
        self.analysis_axis = QtWidgets.QComboBox()
        self.analysis_axis.addItem("Row analysis — metric along time", 1)
        self.analysis_axis.addItem("Column analysis — metric along range", 0)
        export_layout.addWidget(self.analysis_axis)
        self.compute_button = QtWidgets.QPushButton("Compute selected metrics & save CSV…")
        self.compute_button.setObjectName("computeButton")
        self.compute_button.clicked.connect(self._compute_and_export)
        export_layout.addWidget(self.compute_button)
        self.last_export = QtWidgets.QLabel("No CSV created yet")
        self.last_export.setObjectName("panelDescription")
        self.last_export.setWordWrap(True)
        export_layout.addWidget(self.last_export)
        control_layout.addWidget(export_panel)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)

        preview_panel, preview_layout = self._panel(
            "Loaded data preview",
            "This is the matrix that will be analyzed. There is no hidden row-selection mode: computation always uses the full matrix.",
        )
        preview_toolbar = QtWidgets.QHBoxLayout()
        preview_toolbar.addWidget(QtWidgets.QLabel("Show:"))
        self.preview_mode = QtWidgets.QComboBox()
        self.preview_mode.currentIndexChanged.connect(self._refresh_preview)
        preview_toolbar.addWidget(self.preview_mode)
        preview_toolbar.addStretch()
        self.preview_shape = QtWidgets.QLabel("—")
        self.preview_shape.setObjectName("selectionSummary")
        preview_toolbar.addWidget(self.preview_shape)
        preview_layout.addLayout(preview_toolbar)
        self.image_view = pg.ImageView(view=pg.PlotItem())
        theme.style_imageview(self.image_view, "viridis")
        view = self.image_view.getView()
        view.invertY(True)
        view.setAspectLocked(False)
        view.setLabel("bottom", "time / impulses")
        view.setLabel("left", "range bins")
        preview_layout.addWidget(self.image_view, 1)
        splitter.addWidget(preview_panel)
        splitter.setSizes([390, 1050])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)
        self._build_metric_tree()
        return tab

    def _build_visualization_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        toolbar = QtWidgets.QHBoxLayout()
        text = QtWidgets.QVBoxLayout()
        heading = QtWidgets.QLabel("Custom statistic visualizations")
        heading.setObjectName("panelHeading")
        description = QtWidgets.QLabel(
            "Add as many plot spaces as needed. Each space can show any metric from the latest computation."
        )
        description.setObjectName("panelDescription")
        text.addWidget(heading)
        text.addWidget(description)
        toolbar.addLayout(text)
        toolbar.addStretch()
        add = QtWidgets.QPushButton("+ Add empty visualization")
        add.setObjectName("primary")
        add.clicked.connect(self._add_visualization)
        toolbar.addWidget(add)
        layout.addLayout(toolbar)

        self.viz_container = QtWidgets.QWidget()
        self.viz_grid = QtWidgets.QGridLayout(self.viz_container)
        self.viz_grid.setContentsMargins(0, 4, 0, 4)
        self.viz_grid.setSpacing(10)
        self.viz_grid.setColumnStretch(0, 1)
        self.viz_grid.setColumnStretch(1, 1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.viz_container)
        layout.addWidget(scroll, 1)
        self._add_visualization()
        return tab

    def _build_metric_tree(self):
        self.metric_tree.blockSignals(True)
        self.metric_tree.clear()
        for stat_key, meta in core.STATS.items():
            requirement = METRIC_REQUIREMENTS.get(stat_key, "complex")
            parent = QtWidgets.QTreeWidgetItem([meta["label"]])
            parent.setData(0, QtCore.Qt.UserRole, stat_key)
            parent.setData(0, QtCore.Qt.UserRole + 1, requirement)
            parent.setToolTip(0, f"Requires {requirement} data")
            self.metric_tree.addTopLevelItem(parent)
            for output in meta["outputs"]:
                child = QtWidgets.QTreeWidgetItem([output])
                child.setData(0, QtCore.Qt.UserRole, output)
                child_requirement = OUTPUT_REQUIREMENTS.get(
                    (stat_key, output), requirement
                )
                child.setData(0, QtCore.Qt.UserRole + 1, child_requirement)
                child.setToolTip(0, f"Requires {child_requirement} data")
                child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(0, QtCore.Qt.Unchecked)
                parent.addChild(child)
            parent.setExpanded(False)
        self.metric_tree.blockSignals(False)
        self._update_metric_availability()

    def _load(self, kind):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Load {kind}", "", "NumPy arrays (*.npy)"
        )
        if not path:
            return
        try:
            array = np.load(path, mmap_mode="r")
            if kind == "complex":
                amp, phase, amp_visual, phase_visual, layout = decode_stacked_complex(array)
            else:
                decoded, visual = decode_channel_matrix(array)
                counterpart = self._phase_raw if kind == "amplitude" else self._amp_raw
                if counterpart is not None and counterpart.shape != decoded.shape:
                    raise ValueError(
                        f"Decoded shape {decoded.shape} does not match the already loaded "
                        f"channel {counterpart.shape}."
                    )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Could not load data", str(exc))
            return

        if kind == "complex":
            self._amp_raw = np.asarray(amp, dtype=np.float32)
            self._phase_raw = np.asarray(phase, dtype=np.float32)
            self._amp_visual = amp_visual
            self._phase_visual = phase_visual
            self.source_names = {"complex": Path(path).name}
            self.source_notes = {"complex": layout}
        elif kind == "amplitude":
            previous_complex = self.source_names.get("complex")
            self._amp_raw = decoded
            self._amp_visual = visual
            if previous_complex and self._phase_raw is not None:
                self.source_names["phase"] = f"{previous_complex} (derived phase)"
            self.source_names["amplitude"] = Path(path).name
            self.source_names.pop("complex", None)
            self.source_notes.pop("complex", None)
            self.source_notes["amplitude"] = (
                "decoded C0 + C1 + C2" if visual is not None else "2-D matrix"
            )
        else:
            previous_complex = self.source_names.get("complex")
            self._phase_raw = decoded
            self._phase_visual = visual
            if previous_complex and self._amp_raw is not None:
                self.source_names["amplitude"] = f"{previous_complex} (derived amplitude)"
            self.source_names["phase"] = Path(path).name
            self.source_names.pop("complex", None)
            self.source_notes.pop("complex", None)
            self.source_notes["phase"] = (
                "decoded C0 + C1 + C2" if visual is not None else "2-D matrix"
            )
        self._rebuild_signal()

    def _rebuild_signal(self, *_):
        if self._amp_raw is None and self._phase_raw is None:
            self.z = self.amp = self.phase = None
            self._update_data_state()
            return
        amp = self._amp_raw
        phase = self._phase_raw
        if not self.axis_check.isChecked():
            amp = amp.T if amp is not None else None
            phase = phase.T if phase is not None else None
        if amp is not None and phase is not None:
            z = np.asarray(amp, dtype=np.complex64).copy()
            z *= np.exp(np.asarray(phase, dtype=np.float32) * np.complex64(1j))
        elif amp is not None:
            # Amplitude-only metrics operate correctly on real values, so do
            # not allocate a needless full-size complex copy.
            z = np.asarray(amp, dtype=np.float32)
        else:
            z = np.exp(
                np.asarray(phase, dtype=np.float32) * np.complex64(1j)
            ).astype(np.complex64, copy=False)
        self.z = np.ascontiguousarray(z)
        self.amp = np.asarray(amp, dtype=np.float32) if amp is not None else None
        self.phase = np.asarray(phase, dtype=np.float32) if phase is not None else None
        self.computed_series = {}
        self._update_data_state()

    def _clear_data(self):
        self._amp_raw = self._phase_raw = None
        self._amp_visual = self._phase_visual = None
        self.source_names = {}
        self.source_notes = {}
        self.computed_series = {}
        self.z = self.amp = self.phase = None
        self.image_view.clear()
        self._update_data_state()
        self._refresh_visualizations()

    def _update_data_state(self):
        has_amp = self._amp_raw is not None
        has_phase = self._phase_raw is not None
        if has_amp and has_phase:
            label, badge = "Amplitude + phase available", "FULL COMPLEX SIGNAL"
        elif has_amp:
            label, badge = "Amplitude only", "AMPLITUDE ONLY"
        elif has_phase:
            label, badge = "Phase only", "PHASE ONLY"
        else:
            label, badge = "No channels loaded", "NO DATA"
        lines = [label]
        for channel, filename in self.source_names.items():
            note = self.source_notes.get(channel)
            lines.append(
                f"{channel}: {filename}" + (f"\n  {note}" if note else "")
            )
        if self.z is not None:
            lines.append(f"z[range, time] = {self.z.shape[0]} × {self.z.shape[1]}")
            self.preview_shape.setText(f"{self.z.shape[0]} × {self.z.shape[1]}")
        else:
            self.preview_shape.setText("—")
        self.data_info.setText("\n".join(lines))
        self.data_badge.setText(badge)
        self.compute_button.setEnabled(self.z is not None)
        self._update_metric_availability()
        self._update_preview_options()
        self._refresh_preview()

    def _update_preview_options(self):
        current = self.preview_mode.currentData()
        self.preview_mode.blockSignals(True)
        self.preview_mode.clear()
        if self._amp_visual is not None:
            self.preview_mode.addItem("Amplitude source (3 channels)", "amp_rgb")
        if self.amp is not None:
            self.preview_mode.addItem("Amplitude (dB)", "amp_db")
            self.preview_mode.addItem("Amplitude (linear)", "amp")
        if self._phase_visual is not None:
            self.preview_mode.addItem("Phase source (3 channels)", "phase_rgb")
        if self.phase is not None:
            self.preview_mode.addItem("Phase", "phase")
        index = self.preview_mode.findData(current)
        if index >= 0:
            self.preview_mode.setCurrentIndex(index)
        self.preview_mode.blockSignals(False)

    def _refresh_preview(self, *_):
        if self.z is None or self.preview_mode.count() == 0:
            return
        mode = self.preview_mode.currentData()
        if mode in ("amp_rgb", "phase_rgb"):
            visual = self._amp_visual if mode == "amp_rgb" else self._phase_visual
            image = visual if self.axis_check.isChecked() else np.swapaxes(visual, 0, 1)
            self.image_view.setImage(image, autoLevels=True, autoRange=True)
            view = self.image_view.getView()
            view.setLabel("bottom", "time / impulses")
            view.setLabel("left", "range bins")
            return
        if mode == "phase":
            image, levels = self.phase, (-np.pi, np.pi)
        elif mode == "amp":
            image = self.amp
            levels = normalize.clip_levels(image)
        else:
            image = normalize.to_db(self.amp)
            levels = normalize.clip_levels(image)
        self.image_view.setImage(
            image, levels=levels, autoRange=True, autoHistogramRange=False
        )
        view = self.image_view.getView()
        view.setLabel("bottom", "time / impulses")
        view.setLabel("left", "range bins")

    def _requirement_available(self, requirement):
        if requirement == "amplitude":
            return self._amp_raw is not None
        if requirement == "phase":
            return self._phase_raw is not None
        return self._amp_raw is not None and self._phase_raw is not None

    def _update_metric_availability(self):
        if not hasattr(self, "metric_tree"):
            return
        self.metric_tree.blockSignals(True)
        for i in range(self.metric_tree.topLevelItemCount()):
            parent = self.metric_tree.topLevelItem(i)
            requirement = parent.data(0, QtCore.Qt.UserRole + 1)
            child_availability = []
            for j in range(parent.childCount()):
                child = parent.child(j)
                child_requirement = child.data(0, QtCore.Qt.UserRole + 1)
                child_available = self._requirement_available(child_requirement)
                child_availability.append(child_available)
                child.setDisabled(not child_available)
                child.setToolTip(
                    0,
                    f"Requires {child_requirement} data"
                    + ("" if child_available else " — not currently loaded"),
                )
                if not child_available:
                    child.setCheckState(0, QtCore.Qt.Unchecked)
            available = any(child_availability)
            parent.setDisabled(not available)
            parent.setToolTip(
                0,
                f"Requires {requirement} data" + ("" if available else " — not currently loaded"),
            )
        self.metric_tree.blockSignals(False)
        self._metric_selection_changed()

    def _set_all_metrics(self, checked):
        self.metric_tree.blockSignals(True)
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for i in range(self.metric_tree.topLevelItemCount()):
            parent = self.metric_tree.topLevelItem(i)
            if parent.isDisabled():
                continue
            for j in range(parent.childCount()):
                child = parent.child(j)
                if not child.isDisabled():
                    child.setCheckState(0, state)
        self.metric_tree.blockSignals(False)
        self._metric_selection_changed()

    def _selected_metrics(self):
        selected = {}
        for i in range(self.metric_tree.topLevelItemCount()):
            parent = self.metric_tree.topLevelItem(i)
            if parent.isDisabled():
                continue
            outputs = [
                parent.child(j).data(0, QtCore.Qt.UserRole)
                for j in range(parent.childCount())
                if not parent.child(j).isDisabled()
                and parent.child(j).checkState(0) == QtCore.Qt.Checked
            ]
            if outputs:
                selected[parent.data(0, QtCore.Qt.UserRole)] = outputs
        return selected

    def _metric_selection_changed(self, *_):
        count = sum(len(outputs) for outputs in self._selected_metrics().values())
        self.selection_label.setText(f"{count} output{'s' if count != 1 else ''} selected")

    def _compute_and_export(self):
        selections = self._selected_metrics()
        if self.z is None:
            QtWidgets.QMessageBox.information(self, "No data", "Load data first.")
            return
        if not selections:
            QtWidgets.QMessageBox.information(
                self, "No metrics selected", "Tick at least one metric output first."
            )
            return
        default_name = "sigscope_metrics.csv"
        if self.source_names:
            stem = Path(next(iter(self.source_names.values()))).stem
            default_name = f"{stem}_metrics.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save computed metrics", default_name, "CSV files (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        self.pending_csv = Path(path)
        axis = self.analysis_axis.currentData()
        self.compute_button.setEnabled(False)
        self.compute_button.setText("Computing…")
        self.statusBar().showMessage("Computing selected metrics…")
        self.worker = BatchWorker(self.z, axis, selections)
        self.worker.done.connect(self._batch_done)
        self.worker.failed.connect(self._batch_failed)
        self.worker.start()

    def _batch_done(self, series):
        try:
            if not series:
                raise ValueError("No metric output was produced.")
            lengths = {len(values) for values in series.values()}
            if len(lengths) != 1:
                raise ValueError("Selected metrics returned incompatible output lengths.")
            count = lengths.pop()
            index_label = (
                "range_bin" if self.analysis_axis.currentData() == 1 else "time_index"
            )
            names = list(series)
            with self.pending_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([index_label] + names)
                for index in range(count):
                    writer.writerow([index] + [series[name][index] for name in names])
        except Exception as exc:
            self._batch_failed(str(exc))
            return

        self.computed_series = series
        self.last_index_label = index_label
        self.compute_button.setEnabled(True)
        self.compute_button.setText("Compute selected metrics & save CSV…")
        self.last_export.setText(
            f"Saved {len(series)} metric columns × {count} rows\n{self.pending_csv}"
        )
        self.statusBar().showMessage(f"CSV saved: {self.pending_csv}")
        self.tabs.setTabText(1, f"2  Stat Visualizations ({len(series)} metrics)")
        self._refresh_visualizations()

    def _batch_failed(self, message):
        self.compute_button.setEnabled(self.z is not None)
        self.compute_button.setText("Compute selected metrics & save CSV…")
        self.statusBar().showMessage("Computation failed")
        QtWidgets.QMessageBox.critical(self, "Computation failed", str(message)[:5000])

    def _add_visualization(self):
        self._viz_counter += 1
        card = VisualizationCard(self._viz_counter)
        card.remove_requested.connect(self._remove_visualization)
        self.visualizations.append(card)
        card.set_series(
            self.computed_series, getattr(self, "last_index_label", "index")
        )
        self._reflow_visualizations()

    def _remove_visualization(self, card):
        if card not in self.visualizations:
            return
        self.visualizations.remove(card)
        card.setParent(None)
        card.deleteLater()
        self._reflow_visualizations()

    def _reflow_visualizations(self):
        while self.viz_grid.count():
            self.viz_grid.takeAt(0)
        for index, card in enumerate(self.visualizations):
            self.viz_grid.addWidget(card, index // 2, index % 2)
        self.viz_grid.setRowStretch(max(1, (len(self.visualizations) + 1) // 2), 1)

    def _refresh_visualizations(self):
        for card in self.visualizations:
            card.set_series(
                self.computed_series, getattr(self, "last_index_label", "index")
            )


def main():
    app = QtWidgets.QApplication(sys.argv)
    theme.apply_theme(app)
    window = SigScope()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
