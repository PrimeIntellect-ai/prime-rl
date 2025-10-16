"""
Minimalist dark mode PyQt6 visualizer for comparing inference vs trainer logprobs.
Single monolith view with hover showing both distributions.
"""

import sys
import json
import math
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QFileDialog, QLabel, QListWidget, QSplitter, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QPalette, QTextCursor
from transformers import AutoTokenizer


class SequenceViewer(QTextEdit):
    """Text viewer with hover showing both inference and trainer logprobs"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMouseTracking(True)
        self.token_positions = []  # List of (start, end, inf_lp, train_lp, token_id)
        
        font = QFont("Monaco", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#00ff00"))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #00ff00;
                border: none;
                padding: 10px;
            }
        """)
    
    def set_sequence(self, tokens, masks, inference_lp, trainer_lp, tokenizer, prob_threshold=0.1):
        """Render sequence with color-coded masks and highlight discrepancies"""
        self.clear()
        self.token_positions = []
        cursor = QTextCursor(self.document())
        
        char_pos = 0
        for i, (token_id, mask, inf_lp, train_lp) in enumerate(zip(tokens, masks, inference_lp, trainer_lp)):
            decoded = tokenizer.decode([token_id])
            fmt = QTextCharFormat()
            
            # Check for large probability difference
            has_discrepancy = False
            if mask == 1:
                prob_inf = math.exp(inf_lp) if inf_lp > -100 else 0.0
                prob_train = math.exp(train_lp) if train_lp > -100 else 0.0
                prob_diff = abs(prob_inf - prob_train)
                if prob_diff > prob_threshold:
                    has_discrepancy = True
            
            if has_discrepancy:
                fmt.setForeground(QColor("#ffff00"))
                fmt.setBackground(QColor("#332200"))
            elif mask == 1:
                fmt.setForeground(QColor("#00ff00"))
            else:
                fmt.setForeground(QColor("#333333"))
            
            cursor.insertText(decoded, fmt)
            
            end_pos = char_pos + len(decoded)
            self.token_positions.append((char_pos, end_pos, inf_lp, train_lp, token_id))
            char_pos = end_pos
    
    def mouseMoveEvent(self, event):
        cursor = self.cursorForPosition(event.pos())
        pos = cursor.position()
        
        for start, end, inf_lp, train_lp, tok_id in self.token_positions:
            if start <= pos < end:
                prob_inf = math.exp(inf_lp) if inf_lp > -100 else 0.0
                prob_train = math.exp(train_lp) if train_lp > -100 else 0.0
                prob_diff = abs(prob_inf - prob_train)
                
                tooltip = f"Token ID: {tok_id}\n\n"
                tooltip += f"INFERENCE:  p={prob_inf:.6f}  log(p)={inf_lp:.4f}\n"
                tooltip += f"TRAINER:    p={prob_train:.6f}  log(p)={train_lp:.4f}\n"
                tooltip += f"\nΔp = {prob_diff:.6f}"
                
                self.setToolTip(tooltip)
                return
        
        self.setToolTip("")
        super().mouseMoveEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sequences = []
        self.tokenizer = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Logprob Comparison")
        self.setGeometry(100, 100, 1400, 900)
        
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#00ff00"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#00ff00"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#111111"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#00ff00"))
        self.setPalette(palette)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Left sidebar
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #000000; border-right: 1px solid #222222;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(5)
        
        # Controls
        load_btn = QPushButton("Load JSONL")
        load_btn.clicked.connect(self.load_jsonl)
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #111111;
                color: #00ff00;
                border: 1px solid #00ff00;
                padding: 5px;
                font-family: Monaco, monospace;
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #003300; }
        """)
        sidebar_layout.addWidget(load_btn)
        
        # Threshold
        threshold_label = QLabel("Highlight Δp >")
        threshold_label.setStyleSheet("color: #00ff00; font-family: Monaco, monospace; font-size: 9pt;")
        sidebar_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(50)
        self.threshold_slider.setValue(10)
        self.threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #222222;
                height: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff00;
                width: 10px;
                margin: -3px 0;
            }
        """)
        self.threshold_slider.valueChanged.connect(self.update_viewer)
        sidebar_layout.addWidget(self.threshold_slider)
        
        self.threshold_value_label = QLabel("0.10")
        self.threshold_value_label.setStyleSheet("color: #00ff00; font-family: Monaco, monospace; font-size: 9pt; text-align: center;")
        sidebar_layout.addWidget(self.threshold_value_label)
        
        # Stats
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #00ff00; font-family: Monaco, monospace; font-size: 8pt; padding-top: 10px;")
        sidebar_layout.addWidget(self.stats_label)
        
        # Sequence list
        self.seq_list = QListWidget()
        self.seq_list.setStyleSheet("""
            QListWidget {
                background-color: #000000;
                color: #00ff00;
                border: 1px solid #222222;
                font-family: Monaco, monospace;
                font-size: 8pt;
            }
            QListWidget::item {
                padding: 3px;
            }
            QListWidget::item:selected {
                background-color: #003300;
            }
        """)
        self.seq_list.itemSelectionChanged.connect(self.update_viewer)
        sidebar_layout.addWidget(self.seq_list)
        
        self.status_label = QLabel("Loading tokenizer...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #00ff00; font-family: Monaco, monospace; font-size: 8pt;")
        sidebar_layout.addWidget(self.status_label)
        
        sidebar.setFixedWidth(250)
        layout.addWidget(sidebar)
        
        # Main viewer
        self.viewer = SequenceViewer()
        layout.addWidget(self.viewer)
        
        # Load tokenizer
        QApplication.processEvents()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.status_label.setText("Ready")
    
    def load_jsonl(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open JSONL", "", "JSONL Files (*.jsonl);;All Files (*)"
        )
        if not path:
            return
        
        try:
            self.sequences = []
            with open(path, 'r') as f:
                for line in f:
                    self.sequences.append(json.loads(line))
            
            self.seq_list.clear()
            for seq in self.sequences:
                meta = seq['metadata']
                label = f"s{seq['step']} r{seq['rank']} #{seq['seq_idx']}\n{meta['masked_tokens']}/{meta['total_tokens']}"
                self.seq_list.addItem(label)
            
            self.status_label.setText(f"{len(self.sequences)} sequences loaded")
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
    
    def update_viewer(self):
        selected = self.seq_list.selectedIndexes()
        if len(selected) == 0:
            return
        
        prob_threshold = self.threshold_slider.value() / 100.0
        self.threshold_value_label.setText(f"{prob_threshold:.2f}")
        
        idx = selected[0].row()
        seq = self.sequences[idx]
        
        tokens = seq['token_ids']
        masks = seq['mask']
        inference_lp = seq['inference_logprobs']
        trainer_lp = seq['trainer_logprobs']
        
        self.viewer.set_sequence(tokens, masks, inference_lp, trainer_lp, self.tokenizer, prob_threshold)
        
        # Compute stats
        masked_positions = [i for i, m in enumerate(masks) if m == 1]
        if masked_positions:
            inference_masked = [inference_lp[i] for i in masked_positions]
            trainer_masked = [trainer_lp[i] for i in masked_positions]
            
            # Mismatch KL
            kl_terms = []
            for t_lp, i_lp in zip(trainer_masked, inference_masked):
                log_ratio = t_lp - i_lp
                kl = math.exp(log_ratio) - log_ratio - 1
                kl_terms.append(kl)
            
            total_kl = sum(kl_terms)
            avg_kl = total_kl / len(masked_positions)
            
            # Prob diffs
            prob_diffs = []
            for t_lp, i_lp in zip(trainer_masked, inference_masked):
                p1 = math.exp(t_lp) if t_lp > -100 else 0.0
                p2 = math.exp(i_lp) if i_lp > -100 else 0.0
                prob_diffs.append(abs(p1 - p2))
            
            num_large = sum(1 for d in prob_diffs if d > prob_threshold)
            avg_prob = sum(prob_diffs) / len(prob_diffs)
            max_prob = max(prob_diffs)
            
            # First discrepancy
            first_idx = None
            for idx_pos, d in enumerate(prob_diffs):
                if d > prob_threshold:
                    first_idx = masked_positions[idx_pos]
                    break
            
            stats = f"Masked: {len(masked_positions)}\n\n"
            stats += f"KL: {total_kl:.4f}\n"
            stats += f"Avg KL: {avg_kl:.4f}\n\n"
            stats += f"Δp > {prob_threshold:.2f}:\n{num_large} tokens\n\n"
            stats += f"Avg Δp: {avg_prob:.4f}\n"
            stats += f"Max Δp: {max_prob:.4f}\n"
            if first_idx is not None:
                stats += f"\n1st @ pos {first_idx}"
            
            self.stats_label.setText(stats)
        else:
            self.stats_label.setText("No masked tokens")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()