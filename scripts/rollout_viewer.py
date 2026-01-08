#!/usr/bin/env python3
"""Interactive TUI for viewing prime-rl rollouts."""

from pathlib import Path

import msgspec
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static


class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    examples: list[TrainingSample]
    temperature: float
    step: int
    run_idx: int | None = None


def load_batch(path: Path) -> TrainingBatch:
    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)
    with open(path, "rb") as f:
        return decoder.decode(f.read())


def get_steps(rollouts_dir: Path) -> list[int]:
    if not rollouts_dir.exists():
        return []
    return sorted(
        [int(d.name.split("_")[1]) for d in rollouts_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    )


def parse_chatml_from_ids(token_ids: list[int], tokenizer) -> list[tuple[str, str]] | None:
    """Parse ChatML format from token IDs. Returns None if not ChatML."""
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if start_id == tokenizer.unk_token_id or end_id == tokenizer.unk_token_id:
        return None
    messages = []
    i = 0
    while i < len(token_ids):
        if token_ids[i] == start_id:
            j = i + 1
            while j < len(token_ids) and token_ids[j] != end_id:
                j += 1
            segment = tokenizer.decode(token_ids[i + 1 : j], skip_special_tokens=False)
            if "\n" in segment:
                role, content = segment.split("\n", 1)
                messages.append((role.strip(), content.strip()))
            i = j + 1
        else:
            i += 1
    return messages if messages else None


def format_chat_display(messages: list[tuple[str, str]]) -> Text:
    """Format parsed chat messages for display."""
    text = Text()
    role_styles = {"system": "bold yellow", "user": "bold cyan", "assistant": "bold green"}
    for i, (role, content) in enumerate(messages):
        style = role_styles.get(role, "bold white")
        text.append(f"┌─ {role.upper()} ", style=style)
        text.append("─" * (60 - len(role)), style="dim")
        text.append("\n")
        text.append(content)
        if i < len(messages) - 1:
            text.append("\n\n")
    return text


def group_samples_by_prompt(samples: list[TrainingSample]) -> dict[int, list[tuple[int, TrainingSample]]]:
    """Group samples by prompt hash."""
    groups = {}
    for idx, sample in enumerate(samples):
        h = hash(tuple(sample.prompt_ids[:50]))
        if h not in groups:
            groups[h] = []
        groups[h].append((idx, sample))
    return groups


def sparkline(values: list[float], width: int = 20) -> str:
    """Generate a sparkline from values."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    if len(values) <= width:
        buckets = values
    else:
        bucket_size = len(values) / width
        buckets = []
        for i in range(width):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            buckets.append(sum(values[start:end]) / (end - start))
    return "".join(blocks[int((v - mn) / rng * (len(blocks) - 1))] for v in buckets)


class RolloutViewer(App):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("j", "prev_sample", "Prev"),
        Binding("k", "next_sample", "Next"),
        Binding("u", "prev_group", "Prev Problem"),
        Binding("o", "next_group", "Next Problem"),
        Binding("i", "prev_step", "Prev Step"),
        Binding("m", "next_step", "Next Step"),
        Binding("c", "toggle_chat", "Chat"),
        Binding("e", "export", "Export"),
        Binding("b", "sort_best", "Best First"),
        Binding("w", "sort_worst", "Worst First"),
    ]

    CSS = """
    Screen { layout: horizontal; }
    #sidebar { width: 34; height: 100%; layout: horizontal; }
    #step-panel { width: 14; border: solid green; height: 100%; }
    #group-panel { width: 20; border: solid yellow; height: 100%; }
    .sidebar-title { text-align: center; text-style: bold; padding: 0 1; }
    #main { width: 1fr; height: 100%; }
    #stats { height: 3; border: solid gray; padding: 0 1; }
    #sample-container { height: 1fr; border: solid blue; padding: 1; }
    ListView > ListItem.--highlight { background: $accent; }
    #nav-help { height: 1; dock: bottom; background: $surface; color: $text-muted; }
    """

    current_step = reactive(0)
    current_group_idx = reactive(0)
    current_sample_idx = reactive(0)
    sort_mode = reactive("default")

    def __init__(self, rollouts_dir: str, model_name: str):
        super().__init__()
        self.rollouts_dir = Path(rollouts_dir)
        self.model_name = model_name
        self.tokenizer = None
        self.batch = None
        self.steps = []
        self.groups: dict[int, list[tuple[int, TrainingSample]]] = {}
        self.group_keys: list[int] = []
        self.current_group_samples: list[tuple[int, TrainingSample]] = []
        self._original_group_order: list[tuple[int, TrainingSample]] = []
        self.chat_format = True

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Horizontal(id="sidebar"):
                with Vertical(id="step-panel"):
                    yield Label("Steps", classes="sidebar-title")
                    yield ListView(id="step-list")
                with Vertical(id="group-panel"):
                    yield Label("Problems", classes="sidebar-title")
                    yield ListView(id="group-list")
            with Vertical(id="main"):
                yield Static(id="stats")
                yield ScrollableContainer(Static(id="sample-view"), id="sample-container")
        yield Static("j/k: samples | u/o: problems | i/m: steps | b/w: sort | c: chat | e: export", id="nav-help")
        yield Footer()

    def on_mount(self) -> None:
        self.notify("Loading tokenizer...")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.refresh_steps()
        if self.steps:
            self.current_step = self.steps[-1]
            self.load_step(self.current_step)

    def refresh_steps(self) -> None:
        self.steps = get_steps(self.rollouts_dir)
        step_list = self.query_one("#step-list", ListView)
        step_list.clear()
        for step in self.steps:
            step_list.append(ListItem(Label(f"{step}")))

    def _get_score(self, sample: TrainingSample) -> float:
        return sample.reward if sample.reward is not None else (sample.advantage or 0)

    def load_step(self, step: int) -> None:
        batch_path = self.rollouts_dir / f"step_{step}" / "rollouts.bin"
        if not batch_path.exists():
            self.notify(f"Step {step} not found!", severity="error")
            return
        self.batch = load_batch(batch_path)
        self.groups = group_samples_by_prompt(self.batch.examples)
        self.group_keys = sorted(
            self.groups.keys(), key=lambda h: max(self._get_score(s) for _, s in self.groups[h]), reverse=True
        )
        self._update_group_list()
        self.current_group_idx = 0
        self.current_sample_idx = 0
        self._load_group(0)
        self._update_stats()
        self._update_sample_view()

    def _update_group_list(self) -> None:
        group_list = self.query_one("#group-list", ListView)
        group_list.clear()
        for i, h in enumerate(self.group_keys):
            group = self.groups[h]
            best = max(self._get_score(s) for _, s in group)
            group_list.append(ListItem(Label(f"P{i + 1} ({len(group)}) {best:.2f}")))

    def _load_group(self, idx: int) -> None:
        if not self.group_keys or idx >= len(self.group_keys):
            self.current_group_samples = []
            self._original_group_order = []
            return
        self._original_group_order = list(self.groups[self.group_keys[idx]])
        self.current_group_samples = list(self._original_group_order)
        self.current_group_samples.sort(key=lambda x: self._get_score(x[1]), reverse=True)

    def _update_stats(self) -> None:
        stats = self.query_one("#stats", Static)
        if not self.batch:
            return
        n_groups = len(self.group_keys)
        n_attempts = len(self.current_group_samples)
        if self.current_group_samples:
            scores = [self._get_score(s) for _, s in self.current_group_samples]
            spark = sparkline(sorted(scores), 10)
            stats.update(
                f"Step {self.current_step} | Problem {self.current_group_idx + 1}/{n_groups} | "
                f"{n_attempts} attempts | {min(scores):.2f}[dim]{spark}[/]{max(scores):.2f}"
            )
        else:
            stats.update(f"Step {self.current_step} | No samples")

    def _update_sample_view(self) -> None:
        if not self.current_group_samples or not self.batch:
            return
        self.query_one("#sample-container", ScrollableContainer).scroll_home(animate=False)

        orig_idx, sample = self.current_group_samples[self.current_sample_idx]
        adv = sample.advantage or 0.0
        prompt = self.tokenizer.decode(sample.prompt_ids, skip_special_tokens=False)
        completion = self.tokenizer.decode(sample.completion_ids, skip_special_tokens=False)

        text = Text()
        adv_style = "bold green" if adv > 0.1 else "bold red" if adv < -0.1 else "bold yellow"
        text.append(f"Sample {self.current_sample_idx + 1}/{len(self.current_group_samples)}", style="bold")
        text.append(f" (#{orig_idx})\n", style="dim")
        if sample.reward is not None:
            reward_style = "bold green" if sample.reward > 0.5 else "bold red" if sample.reward < 0.5 else "bold yellow"
            text.append("Reward: ", style="dim")
            text.append(f"{sample.reward:.4f}", style=reward_style)
            text.append(" | Adv: ", style="dim")
            text.append(f"{adv:+.4f}", style=adv_style)
        else:
            text.append("Advantage: ", style="dim")
            text.append(f"{adv:+.4f}", style=adv_style)
        text.append(f" | {len(sample.prompt_ids)}+{len(sample.completion_ids)} toks\n", style="dim")

        lps = sample.completion_logprobs
        avg_lp = sum(lps) / len(lps) if lps else 0
        min_lp = min(lps) if lps else 0
        teacher_info = ""
        if sample.teacher_logprobs:
            tlps = sample.teacher_logprobs
            avg_tlp = sum(tlps) / len(tlps) if tlps else 0
            teacher_info = f" | Teacher avg: {avg_tlp:.2f}"
        text.append(f"Logprobs: avg={avg_lp:.2f}, min={min_lp:.2f}{teacher_info}\n\n", style="dim")

        if self.chat_format:
            parsed = parse_chatml_from_ids(sample.prompt_ids + sample.completion_ids, self.tokenizer)
            if parsed:
                text.append_text(format_chat_display(parsed))
            else:
                text.append("[dim]Not ChatML format[/]\n\n")
                text.append("─── PROMPT ───\n", style="bold cyan")
                text.append(prompt[-1500:] if len(prompt) > 1500 else prompt, style="dim")
                text.append("\n\n─── COMPLETION ───\n", style="bold magenta")
                text.append(completion[:3000] if len(completion) > 3000 else completion)
        else:
            text.append("─── PROMPT ───\n", style="bold cyan")
            text.append(prompt[-1500:] if len(prompt) > 1500 else prompt, style="dim")
            text.append("\n\n─── COMPLETION ───\n", style="bold magenta")
            text.append(completion[:3000] if len(completion) > 3000 else completion)

        self.query_one("#sample-view", Static).update(text)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None:
            return
        if event.list_view.id == "step-list":
            self._select_step_by_index(idx)
        elif event.list_view.id == "group-list":
            self._select_group_by_index(idx)

    def action_refresh(self) -> None:
        self.refresh_steps()
        if self.steps:
            self.current_step = self.steps[-1]
            self.load_step(self.current_step)
        self.notify("Refreshed!")

    def action_next_sample(self) -> None:
        if self.current_sample_idx < len(self.current_group_samples) - 1:
            self.current_sample_idx += 1
            self._update_sample_view()

    def action_prev_sample(self) -> None:
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self._update_sample_view()

    def _select_step_by_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.steps):
            return
        step_list = self.query_one("#step-list", ListView)
        step_list.index = idx
        self.current_step = self.steps[idx]
        self.load_step(self.current_step)

    def action_next_step(self) -> None:
        try:
            idx = self.steps.index(self.current_step)
            self._select_step_by_index(idx + 1)
        except ValueError:
            pass

    def action_prev_step(self) -> None:
        try:
            idx = self.steps.index(self.current_step)
            self._select_step_by_index(idx - 1)
        except ValueError:
            pass

    def _select_group_by_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.group_keys):
            return
        group_list = self.query_one("#group-list", ListView)
        group_list.index = idx
        self.current_group_idx = idx
        self._load_group(idx)
        self.current_sample_idx = 0
        self._update_stats()
        self._update_sample_view()

    def action_next_group(self) -> None:
        self._select_group_by_index(self.current_group_idx + 1)

    def action_prev_group(self) -> None:
        self._select_group_by_index(self.current_group_idx - 1)

    def action_toggle_chat(self) -> None:
        self.chat_format = not self.chat_format
        self._update_sample_view()
        self.notify(f"Chat format: {'on' if self.chat_format else 'off'}")

    def action_sort_best(self) -> None:
        if self.sort_mode == "best":
            self.sort_mode = "default"
            self.current_group_samples = list(self._original_group_order)
        else:
            self.sort_mode = "best"
            self.current_group_samples.sort(key=lambda x: self._get_score(x[1]), reverse=True)
        self.current_sample_idx = 0
        self._update_sample_view()
        self.notify("Sorted: best first" if self.sort_mode == "best" else "Original order")

    def action_sort_worst(self) -> None:
        if self.sort_mode == "worst":
            self.sort_mode = "default"
            self.current_group_samples = list(self._original_group_order)
        else:
            self.sort_mode = "worst"
            self.current_group_samples.sort(key=lambda x: self._get_score(x[1]), reverse=False)
        self.current_sample_idx = 0
        self._update_sample_view()
        self.notify("Sorted: worst first" if self.sort_mode == "worst" else "Original order")

    def action_export(self) -> None:
        if not self.current_group_samples:
            self.notify("No sample to export!", severity="warning")
            return
        from datetime import datetime

        orig_idx, sample = self.current_group_samples[self.current_sample_idx]
        prompt = self.tokenizer.decode(sample.prompt_ids, skip_special_tokens=False)
        completion = self.tokenizer.decode(sample.completion_ids, skip_special_tokens=False)

        adv = sample.advantage or 0.0
        lps = sample.completion_logprobs
        avg_lp = sum(lps) / len(lps) if lps else 0
        min_lp = min(lps) if lps else 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path.cwd() / f"sample_s{self.current_step}_p{self.current_group_idx}_{ts}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Exported from rollout_viewer\n")
            f.write(f"# Step: {self.current_step}\n")
            f.write(f"# Problem: {self.current_group_idx + 1}/{len(self.group_keys)}\n")
            f.write(f"# Sample: {self.current_sample_idx + 1}/{len(self.current_group_samples)} (orig #{orig_idx})\n")
            f.write(f"# Timestamp: {ts}\n\n")
            f.write(f"Advantage: {adv:+.4f}\n")
            if sample.reward is not None:
                f.write(f"Reward: {sample.reward:.4f}\n")
            f.write(f"Prompt tokens: {len(sample.prompt_ids)}\n")
            f.write(f"Completion tokens: {len(sample.completion_ids)}\n")
            f.write(f"Logprobs: avg={avg_lp:.2f}, min={min_lp:.2f}\n\n")
            f.write("=" * 80 + "\nPROMPT\n" + "=" * 80 + "\n")
            f.write(prompt)
            f.write("\n\n" + "=" * 80 + "\nCOMPLETION\n" + "=" * 80 + "\n")
            f.write(completion)
        self.notify(f"Exported to {path.name}")


def find_latest_rollouts() -> str:
    cwd = Path.cwd()
    for d in cwd.iterdir():
        if d.is_dir() and d.name.startswith("outputs"):
            for run_dir in d.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run_"):
                    rollouts = run_dir / "rollouts"
                    if rollouts.exists():
                        return str(rollouts)
    return "outputs/run_default/rollouts"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    RolloutViewer(args.dir or find_latest_rollouts(), args.model).run()


if __name__ == "__main__":
    main()
