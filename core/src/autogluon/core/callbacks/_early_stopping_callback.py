from logging import Logger
from typing import Tuple

from ..trainer import AbstractTrainer
from ._abstract_callback import AbstractCallback


class EarlyStoppingCallback(AbstractCallback):
    """
    A simple early stopping callback.

    Will early stop AutoGluon's training process after `patience` number of models fitted sequentially without improvement to score_val.

    Parameters
    ----------
    patience : int, default = 10
        The number of models fit in a row without improvement in score_val before early stopping the training process.
    verbose : bool, default = False
        If True, will log a stopping message when early stopping triggers.
    """

    def __init__(self, patience: int = 10, verbose: bool = False):
        self.patience = patience
        self.last_improvement = 0
        self.score_best = None
        self.verbose = verbose

    def before_fit(self, logger: Logger, **kwargs) -> Tuple[bool, bool]:
        early_stop = self._early_stop()
        if self.verbose and early_stop:
            msg = f"Stopping trainer fit due to callback early stopping. Reason: No score_val improvement in the past {self.last_improvement} models."
            self._log(logger, 20, msg=msg)
        return early_stop, False

    def after_fit(self, trainer: AbstractTrainer, logger: Logger, **kwargs) -> bool:
        self._calc_new_best(trainer=trainer)
        early_stop = self._early_stop()
        if self.verbose and early_stop:
            msg = f"Stopping trainer fit due to callback early stopping. Reason: No score_val improvement in the past {self.last_improvement} models."
            self._log(logger, 20, msg=msg)
        return early_stop

    def _calc_new_best(self, trainer: AbstractTrainer):
        leaderboard = trainer.leaderboard()
        if len(leaderboard) == 0:
            score_cur = None
        else:
            score_cur = leaderboard["score_val"].max()
        if score_cur is None:
            self.last_improvement += 1
        elif self.score_best is None or score_cur > self.score_best:
            self.score_best = score_cur
            self.last_improvement = 0
        else:
            self.last_improvement += 1

    def _early_stop(self):
        if self.last_improvement >= self.patience:
            return True
        else:
            return False

    def _log(self, logger: Logger, level, msg: str):
        msg = f"{self.__class__.__name__}: {msg}"
        logger.log(
            level,
            msg,
        )
