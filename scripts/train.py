"""Train MaskablePPO. Example: uv run python scripts/train.py train.total_timesteps=1000000"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from blockblast.utils.config import to_app_config
from blockblast.utils.logger import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Imported here, not at the top: on Windows each env worker process re-runs this
    # module's top-level imports, and the training stack would load torch (~0.5 GB) in
    # every worker.
    from blockblast.training.train import train

    setup_logging()
    app = to_app_config(cfg)
    if app.agent.name != "maskable_ppo":
        raise SystemExit("only agent=maskable_ppo is trainable")
    path = train(app)
    print(f"model saved: {path}")


if __name__ == "__main__":
    main()
