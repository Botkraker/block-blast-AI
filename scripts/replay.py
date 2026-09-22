"""Replay a recorded episode in the terminal or export it as a GIF.

uv run python scripts/replay.py data/replays/greedy_seed1000000.json
uv run python scripts/replay.py data/replays/greedy_seed1000000.json --gif out.gif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from blockblast.visualization import export_gif, load_record, render_ansi, replay_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--gif", type=Path, default=None, help="write a GIF instead of printing")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between ANSI frames")
    args = parser.parse_args()

    record = load_record(args.record)
    if args.gif:
        print(f"wrote {export_gif(record, args.gif, fps=args.fps)}")
        return
    state = None
    for i, state in enumerate(replay_states(record)):
        print(f"--- move {i}/{len(record.actions)}\n{render_ansi(state)}\n")
        if args.delay:
            time.sleep(args.delay)
    assert state is not None
    status = "OK" if state.score == record.final_score else "MISMATCH"
    print(f"final score {state.score} (recorded {record.final_score}) {status}")


if __name__ == "__main__":
    main()
