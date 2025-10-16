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
    QTextBrowser, QPushButton, QFileDialog, QLabel, QListWidget, QSlider, QSplitter
)
from PyQt6.QtCore import Qt, QPoint, QRect, QTimer
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QPalette, QTextCursor, QKeySequence, QShortcut
from transformers import AutoTokenizer


class SelectionInfoOverlay(QLabel):
    """Persistent overlay showing token info at selection location"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_font_size = 10
        self.update_style()
        self.setWordWrap(False)
        self.hide()
        self.setWindowFlags(Qt.WindowType.ToolTip)
    
    def update_style(self):
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(0, 0, 0, 220);
                color: #00ff00;
                border: 2px solid #00ff00;
                padding: 8px;
                font-family: Monaco, monospace;
                font-size: {self.base_font_size}pt;
            }}
        """)


class SequenceViewer(QTextBrowser):
    """Text viewer - shows persistent info for token with most SELECTED characters"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setMouseTracking(True)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | 
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.token_data = []
        self.base_font_size = 11
        self.parent_window = parent
        self.last_tooltip_token_idx = None
        self.selection_pos = None
        
        self.update_font()
        
        # Create overlay
        self.info_overlay = SelectionInfoOverlay(self)
        
        # Set up palette with visible selection colors
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Base, QColor("#000000"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#00ff00"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#0066cc"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QTextBrowser {
                background-color: #000000;
                color: #00ff00;
                border: none;
                padding: 10px;
                selection-background-color: #0066cc;
                selection-color: #ffffff;
            }
        """)
        
        # Timer for checking selection
        self.selection_timer = QTimer()
        self.selection_timer.timeout.connect(self.check_selection)
        self.selection_timer.start(100)
    
    def update_font(self):
        font = QFont("Monaco", self.base_font_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
    
    def set_sequence(self, tokens, masks, inference_lp, trainer_lp, tokenizer, prob_threshold=0.1):
        """Render sequence with color-coded masks and highlight discrepancies"""
        self.clear()
        self.token_data = []
        
        # Decode all tokens
        decoded_tokens = []
        for i, (token_id, mask, inf_lp, train_lp) in enumerate(zip(tokens, masks, inference_lp, trainer_lp)):
            decoded = tokenizer.decode([token_id])
            
            has_discrepancy = False
            if mask == 1:
                prob_inf = math.exp(inf_lp) if inf_lp > -100 else 0.0
                prob_train = math.exp(train_lp) if train_lp > -100 else 0.0
                prob_diff = abs(prob_inf - prob_train)
                if prob_diff > prob_threshold:
                    has_discrepancy = True
            
            decoded_tokens.append({
                'text': decoded,
                'token_id': token_id,
                'mask': mask,
                'inf_lp': inf_lp,
                'train_lp': train_lp,
                'has_discrepancy': has_discrepancy
            })
        
        # Build document
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        
        char_pos = 0
        for token_idx, token_info in enumerate(decoded_tokens):
            decoded = token_info['text']
            mask = token_info['mask']
            has_discrepancy = token_info['has_discrepancy']
            
            fmt = QTextCharFormat()
            if has_discrepancy:
                fmt.setForeground(QColor("#ffff00"))
                fmt.setBackground(QColor("#332200"))
            elif mask == 1:
                fmt.setForeground(QColor("#00ff00"))
            else:
                fmt.setForeground(QColor("#333333"))
            
            cursor.insertText(decoded, fmt)
            
            char_end = char_pos + len(decoded)
            self.token_data.append({
                'token_id': token_info['token_id'],
                'inf_lp': token_info['inf_lp'],
                'train_lp': token_info['train_lp'],
                'text': decoded,
                'char_start': char_pos,
                'char_end': char_end
            })
            
            char_pos = char_end
        
        cursor.endEditBlock()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
    
    def get_token_with_most_selected_chars(self):
        """Find which token has the most SELECTED characters"""
        cursor = self.textCursor()
        
        if not cursor.hasSelection():
            return None, None
        
        sel_start = cursor.selectionStart()
        sel_end = cursor.selectionEnd()
        
        # Store selection position for overlay placement
        cursor_at_start = QTextCursor(self.document())
        cursor_at_start.setPosition(sel_start)
        self.selection_pos = self.cursorRect(cursor_at_start)
        
        # Count selected characters per token
        token_votes = {}
        
        for token_idx, token_info in enumerate(self.token_data):
            char_start = token_info['char_start']
            char_end = token_info['char_end']
            
            overlap_start = max(sel_start, char_start)
            overlap_end = min(sel_end, char_end)
            
            if overlap_start < overlap_end:
                selected_chars = overlap_end - overlap_start
                token_votes[token_idx] = selected_chars
        
        if not token_votes:
            return None, None
        
        best_token_idx = max(token_votes, key=token_votes.get)
        return best_token_idx, self.token_data[best_token_idx]
    
    def check_selection(self):
        """Check current selection and show persistent info for token with most selected chars"""
        token_idx, token_info = self.get_token_with_most_selected_chars()
        
        if token_idx is None:
            # No selection - hide overlay
            self.info_overlay.hide()
            self.last_tooltip_token_idx = None
            return
        
        # Generate tooltip text
        tok_id = token_info['token_id']
        inf_lp = token_info['inf_lp']
        train_lp = token_info['train_lp']
        
        prob_inf = math.exp(inf_lp) if inf_lp > -100 else 0.0
        prob_train = math.exp(train_lp) if train_lp > -100 else 0.0
        prob_diff = abs(prob_inf - prob_train)
        
        log_ratio = train_lp - inf_lp
        mismatch_kl = math.exp(log_ratio) - log_ratio - 1
        
        text_repr = repr(token_info['text'])[1:-1]
        if len(text_repr) > 30:
            text_repr = text_repr[:27] + "..."
        
        info_text = f"Token #{token_idx}  ID: {tok_id}\n"
        info_text += f"Text: '{text_repr}'\n"
        info_text += f"Chars: [{token_info['char_start']}:{token_info['char_end']}]\n\n"
        info_text += f"INFERENCE:  p={prob_inf:.6f}  log(p)={inf_lp:.4f}\n"
        info_text += f"TRAINER:    p={prob_train:.6f}  log(p)={train_lp:.4f}\n"
        info_text += f"\nΔp = {prob_diff:.6f}\n"
        info_text += f"Mismatch KL = {mismatch_kl:.6f}"
        
        self.info_overlay.setText(info_text)
        self.info_overlay.adjustSize()
        
        # Position overlay BELOW the selection to avoid overlap
        if self.selection_pos:
            overlay_x = self.selection_pos.left()
            overlay_y = self.selection_pos.bottom() + 5  # 5px below selection
            
            # Make sure overlay doesn't go off screen to the right
            overlay_width = self.info_overlay.width()
            viewport_width = self.viewport().width()
            if overlay_x + overlay_width > viewport_width:
                overlay_x = viewport_width - overlay_width - 10
            
            # Convert to global coordinates
            global_pos = self.mapToGlobal(QPoint(overlay_x, overlay_y))
            self.info_overlay.move(global_pos)
        
        self.info_overlay.show()
        self.info_overlay.raise_()
        self.last_tooltip_token_idx = token_idx


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sequences = []
        self.tokenizer = None
        self.zoom_level = 1.0
        self.init_ui()
        self.setup_shortcuts()
    
    def zoom_in(self):
        self.zoom_level *= 1.1
        self.apply_zoom()
    
    def zoom_out(self):
        self.zoom_level /= 1.1
        self.apply_zoom()
    
    def apply_zoom(self):
        base_sidebar_font = int(9 * self.zoom_level)
        base_stats_font = int(8 * self.zoom_level)
        base_viewer_font = int(11 * self.zoom_level)
        base_overlay_font = int(10 * self.zoom_level)
        
        self.viewer.base_font_size = base_viewer_font
        self.viewer.update_font()
        
        # Update overlay font size
        self.viewer.info_overlay.base_font_size = base_overlay_font
        self.viewer.info_overlay.update_style()
        
        self.load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #111111;
                color: #00ff00;
                border: 1px solid #00ff00;
                padding: 5px;
                font-family: Monaco, monospace;
                font-size: {base_sidebar_font}pt;
            }}
            QPushButton:hover {{ background-color: #003300; }}
        """)
        
        self.threshold_label.setStyleSheet(f"color: #00ff00; font-family: Monaco, monospace; font-size: {base_sidebar_font}pt;")
        self.threshold_value_label.setStyleSheet(f"color: #00ff00; font-family: Monaco, monospace; font-size: {base_sidebar_font}pt; text-align: center;")
        self.stats_label.setStyleSheet(f"color: #00ff00; font-family: Monaco, monospace; font-size: {base_stats_font}pt; padding-top: 10px;")
        self.status_label.setStyleSheet(f"color: #00ff00; font-family: Monaco, monospace; font-size: {base_stats_font}pt;")
        
        self.seq_list.setStyleSheet(f"""
            QListWidget {{
                background-color: #000000;
                color: #00ff00;
                border: 1px solid #222222;
                font-family: Monaco, monospace;
                font-size: {base_stats_font}pt;
            }}
            QListWidget::item {{
                padding: 3px;
            }}
            QListWidget::item:selected {{
                background-color: #003300;
            }}
        """)
    
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
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #222222;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #00ff00;
            }
        """)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #000000;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        sidebar_layout.setSpacing(5)
        
        self.load_btn = QPushButton("Load Folder")
        self.load_btn.clicked.connect(self.load_folder)
        self.load_btn.setStyleSheet("""
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
        sidebar_layout.addWidget(self.load_btn)
        
        self.threshold_label = QLabel("Highlight Δp >")
        self.threshold_label.setStyleSheet("color: #00ff00; font-family: Monaco, monospace; font-size: 9pt;")
        sidebar_layout.addWidget(self.threshold_label)
        
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
        
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #00ff00; font-family: Monaco, monospace; font-size: 8pt; padding-top: 10px;")
        sidebar_layout.addWidget(self.stats_label)
        
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
        
        splitter.addWidget(sidebar)
        
        self.viewer = SequenceViewer(parent=self)
        splitter.addWidget(self.viewer)
        
        splitter.setSizes([250, 1150])
        
        layout.addWidget(splitter)
        
        QApplication.processEvents()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.status_label.setText("Ready")
    
    def setup_shortcuts(self):
        zoom_in = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        zoom_in.activated.connect(self.zoom_in)
        
        zoom_in_alt = QShortcut(QKeySequence("Ctrl+="), self)
        zoom_in_alt.activated.connect(self.zoom_in)
        
        zoom_out = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        zoom_out.activated.connect(self.zoom_out)
    
    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with JSONL files")
        if not folder:
            return
        
        try:
            self.sequences = []
            self.seq_list.clear()
            
            folder_path = Path(folder)
            jsonl_files = sorted(folder_path.glob("*.jsonl"))
            
            if not jsonl_files:
                self.status_label.setText("No JSONL files found in folder")
                return
            
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        seq = json.loads(line)
                        seq['_source_file'] = jsonl_file.name
                        self.sequences.append(seq)
            
            for seq in self.sequences:
                meta = seq['metadata']
                source = seq.get('_source_file', 'unknown')
                label = f"{source}\ns{seq['step']} r{seq['rank']} #{seq['seq_idx']}\n{meta['masked_tokens']}/{meta['total_tokens']}"
                self.seq_list.addItem(label)
            
            self.status_label.setText(f"{len(self.sequences)} sequences from {len(jsonl_files)} files")
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
        
        masked_positions = [i for i, m in enumerate(masks) if m == 1]
        if masked_positions:
            inference_masked = [inference_lp[i] for i in masked_positions]
            trainer_masked = [trainer_lp[i] for i in masked_positions]
            
            kl_terms = []
            for t_lp, i_lp in zip(trainer_masked, inference_masked):
                log_ratio = t_lp - i_lp
                kl = math.exp(log_ratio) - log_ratio - 1
                kl_terms.append(kl)
            
            total_kl = sum(kl_terms)
            avg_kl = total_kl / len(masked_positions)
            
            prob_diffs = []
            for t_lp, i_lp in zip(trainer_masked, inference_masked):
                p1 = math.exp(t_lp) if t_lp > -100 else 0.0
                p2 = math.exp(i_lp) if i_lp > -100 else 0.0
                prob_diffs.append(abs(p1 - p2))
            
            num_large = sum(1 for d in prob_diffs if d > prob_threshold)
            avg_prob = sum(prob_diffs) / len(prob_diffs)
            max_prob = max(prob_diffs)
            
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