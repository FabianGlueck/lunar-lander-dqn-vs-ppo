"""Environment-Factory: erzeugt geseedete, gekapselte LunarLander-Instanzen."""

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from lunarlander import config


def make_env(seed: int, render_mode: str | None = None, **env_kwargs) -> gym.Env:
    """Baut eine LunarLander-Umgebung mit fest gesetztem Seed.

    Args:
        seed: Startwert für Reset *und* Aktionsraum → reproduzierbare Läufe.
        render_mode: z. B. "rgb_array" für Frames (GIF), sonst None (kein Rendering).
        **env_kwargs: zusätzliche Parameter, die an `gym.make` durchgereicht werden —
            z. B. `enable_wind=True`, `wind_power=15.0`, `turbulence_power=1.5`,
            `gravity=-10.0`. Zum Experimentieren; fürs Training/den Vergleich weglassen.

    Returns:
        Eine `Monitor`-gewrappte Env. Der Monitor zeichnet Episoden-Returns/-Längen
        auf, die Stable-Baselines3 fürs Logging und die Evaluation braucht.
    """
    env = gym.make(config.ENV_ID, render_mode=render_mode, **env_kwargs)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env
