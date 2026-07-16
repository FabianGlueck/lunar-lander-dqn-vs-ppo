import optuna
from lunarlander import tune
from lunarlander.train import train


class _RecordingTrial:
    """Fake Optuna trial that records report() calls and never prunes."""

    def __init__(self):
        self.reported = []

    def report(self, value, step):
        self.reported.append((value, step))

    def should_prune(self):
        return False


class _PruningTrial:
    """Fake Optuna trial that always asks to prune."""

    def report(self, value, step):
        pass

    def should_prune(self):
        return True


def test_trial_eval_callback_reports_intermediate_values(tmp_path):
    trial = _RecordingTrial()

    def factory(eval_env, eval_freq):
        return tune.TrialEvalCallback(eval_env, trial, n_eval_episodes=2, eval_freq=eval_freq)

    train("dqn", {"learning_starts": 100}, seed=0, timesteps=1500,
          log_dir=tmp_path, eval_callback_fn=factory)
    assert len(trial.reported) >= 1


def test_trial_eval_callback_sets_is_pruned_and_stops(tmp_path):
    trial = _PruningTrial()
    holder = {}

    def factory(eval_env, eval_freq):
        cb = tune.TrialEvalCallback(eval_env, trial, n_eval_episodes=2, eval_freq=eval_freq)
        holder["cb"] = cb
        return cb

    train("dqn", {"learning_starts": 100}, seed=0, timesteps=1500,
          log_dir=tmp_path, eval_callback_fn=factory)
    assert holder["cb"].is_pruned is True


def test_sample_hyperparams_dqn():
    study = optuna.create_study()
    trial = study.ask()
    params = tune.sample_hyperparams(trial, "dqn")
    assert "learning_rate" in params
    assert "gamma" in params


def test_sample_hyperparams_ppo():
    study = optuna.create_study()
    trial = study.ask()
    params = tune.sample_hyperparams(trial, "ppo")
    assert "n_steps" in params
    assert "clip_range" in params


def test_run_study_tiny(tmp_path):
    db = tmp_path / "t.db"
    study = tune.run_study("dqn", n_trials=2, db_path=db,
                           timesteps=1500, study_name="tiny")
    assert len(study.trials) == 2
    assert study.best_value is not None


def test_best_params_reads_from_study(tmp_path):
    db = tmp_path / "t.db"
    study = tune.run_study("dqn", n_trials=2, db_path=db,
                           timesteps=1500, study_name="tiny")
    params = tune.best_params("dqn", db_path=db, study_name="tiny")
    assert params == study.best_params
    assert isinstance(params, dict)


def test_sample_hyperparams_ppo_extended_widens_options():
    study = optuna.create_study()
    trial = study.ask()
    p = tune.sample_hyperparams_ppo_extended(trial)
    assert "clip_range" in p and "n_steps" in p
    # die ausgereizten Dimensionen sind erweitert:
    assert 512 in trial.distributions["n_steps"].choices
    assert 30 in trial.distributions["n_epochs"].choices
    assert [256, 256] in trial.distributions["net_arch"].choices


def test_run_study_with_custom_sample_fn(tmp_path):
    db = tmp_path / "v2.db"
    study = tune.run_study("ppo", n_trials=2, db_path=db, timesteps=1500,
                           study_name="v2", sample_fn=tune.sample_hyperparams_ppo_extended)
    assert len(study.trials) == 2
    assert study.best_value is not None
