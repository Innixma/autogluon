from __future__ import annotations

import copy
import logging
import time
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm.basic import Booster, _ConfigAliases, _log_info, _log_warning
from lightgbm.callback import EarlyStopException, _format_eval_result, CallbackEnv, _EarlyStoppingCallback

from autogluon.common.utils.lite import disable_if_lite_mode
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.utils.early_stopping import SimpleES, ESWrapperOOF
from autogluon.core.utils.utils import get_pred_from_proba

logger = logging.getLogger(__name__)


class _EarlyStoppingCustomCallback(_EarlyStoppingCallback):
    def __init__(
        self,
        stopping_rounds: int | tuple | None = None,
        metrics_to_use: list | None = None,
        first_metric_only: bool = False,
        start_time: float | None = None,
        time_limit: float | None = None,
        ignore_dart_warning: bool = False,
        enable_mem_check: bool = True,
        enable_es_oof: bool = False,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        stopping_metric: Callable | None = None,
        problem_type: str | None = None,
        verbose: bool = True,
    ):
        """
        AutoGluon's custom early stopping callback for LightGBM.
        This extends LightGBM's native early stopping callback with the following extra functionality:

        1. Can specify time_limit for early stopping.
        2. Can early stop when nearing out-of-memory.
        3. Can specify a dynamic early stopping patience.
        4. Can early stop in dart mode.
        5. [Experimental] Can enable leave-one-out early stopping.

        Parameters
        ----------
        stopping_rounds : int | tuple, optional
            The stopping rounds for early stopping.
            If None, early stopping via performance is disabled.
        metrics_to_use : list, optional
            If specified, will use the provided metrics for determining early stopping.
            Required if stopping_rounds is not None.
        first_metric_only : bool = False
            If True, only the first metric in `metrics_to_use` is considered for early stopping.
        start_time : float, optional
            If specified, will early stop the model if the time limit is exceeded.
            Elapsed time is calculated as `time.time() - start_time`.
        time_limit : float, optional
            If specified, will early stop the model if the time limit is exceeded.
            Elapsed time is calculated as `time.time() - start_time`.
        ignore_dart_warning : bool, default False
            If True, enables early stopping with dart mode.
            If False and dart mode is enabled, the callback will be disabled.
        enable_mem_check : bool, default True
            If True, will check that memory usage is within safe values.
            If there is a risk of running out-of-memory, early stopping will be triggered.
        enable_es_oof : bool, default False
            Experimental, enables leave-one-out early stopping.
        X_val : pd.DataFrame, optional
            Required for enable_es_oof=True
            The validation data to be passed to `model.predict`
        y_val : np.ndarray, optional
            Required for enable_es_oof=True
            The validation data ground truth.
        stopping_metric : Callable, optional
            Required for enable_es_oof=True
        problem_type : str, optional
            Required for enable_es_oof=True
        verbose : bool, default True
            If True, logs additional information during fit.
        """
        super().__init__(
            stopping_rounds=1,  # Dummy value
            first_metric_only=first_metric_only,
            verbose=verbose,
            min_delta=0.0,  # min_delta != 0 is not implemented
        )
        self.stopping_rounds = stopping_rounds

        self.best_trainloss = []  # stores training losses at corresponding best_iter
        self.indices_to_check = []
        self.es = []
        self.init_mem_rss = None
        self.init_mem_avail = None
        self.score_oof = 0
        self.early_stop = False
        self.enable_mem_check = enable_mem_check
        self.enable_es_oof = enable_es_oof

        self.metrics_to_use = metrics_to_use
        self.start_time = start_time
        self.time_limit = time_limit
        self.ignore_dart_warning = ignore_dart_warning
        self.X_val = X_val
        self.y_val = y_val
        self.problem_type = problem_type

        self.early_stop_idx = None

        self.mem_status = None

        if self.stopping_rounds is not None:
            if isinstance(self.stopping_rounds, int):
                self.es_template = SimpleES(patience=self.stopping_rounds)
            else:
                self.es_template = self.stopping_rounds[0](**self.stopping_rounds[1])
            if self.enable_es_oof:
                assert self.X_val is not None, f"X_val must be specified when `enable_es_oof=True`"
                assert self.y_val is not None, f"y_val must be specified when `enable_es_oof=True`"
                assert stopping_metric is not None, f"stopping_metric must be specified when `enable_es_oof=True`"
                assert self.problem_type is not None, f"problem_type must be specified when `enable_es_oof=True`"
                self.early_stop_oof = False
                self.es_wrapper_oof: ESWrapperOOF = ESWrapperOOF(
                    es=copy.deepcopy(self.es_template),
                    score_func=stopping_metric,
                    best_is_later_if_tie=False,
                )
            else:
                self.early_stop_oof = True
                self.es_wrapper_oof = None
        else:
            self.es_template = None
            self.es_wrapper_oof = None
            self.early_stop_oof = True

    @property
    def enable_es(self) -> bool:
        return self.stopping_rounds is not None

    def _init(self, env: CallbackEnv) -> None:
        self._init_og(env=env)

        if self.metrics_to_use is None:
            for i in range(len(env.evaluation_result_list)):
                self.indices_to_check.append(i)
                if self.first_metric_only:
                    break
        else:
            for i, eval in enumerate(env.evaluation_result_list):
                if (eval[0], eval[1]) in self.metrics_to_use:
                    self.indices_to_check.append(i)
                    if self.first_metric_only:
                        break

        if self.enable_mem_check:
            self.mem_status = ResourceManager.get_process()

            @disable_if_lite_mode()
            def _init_mem():
                self.init_mem_rss = self.mem_status.memory_info().rss
                self.init_mem_avail = ResourceManager.get_available_virtual_mem()

            _init_mem()

    def _init_og(self, env: CallbackEnv) -> None:
        """
        The original LightGBM code for `_EarlyStopingCallback._init`, only changing the `is_dart` logic to not disable ES in dart mode, and init `self.es`
        Code snapshot from LightGBM version 4.3.0
        """
        if self.enable_es:
            self._init_es(env=env)

        self._reset_storages()

    def _init_es(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None or env.evaluation_result_list == []:
            raise ValueError(
                "For early stopping, at least one dataset and eval metric is required for evaluation"
            )

        if not self.ignore_dart_warning:
            is_dart = any(env.params.get(alias, "") == 'dart' for alias in _ConfigAliases.get("boosting"))
            if is_dart:
                self.enabled = False
                _log_warning('Early stopping is not available in dart mode')
                return

        # validation sets are guaranteed to not be identical to the training data in cv()
        if isinstance(env.model, Booster):
            only_train_set = (
                len(env.evaluation_result_list) == 1
                and self._is_train_set(
                    ds_name=env.evaluation_result_list[0][0],
                    eval_name=env.evaluation_result_list[0][1].split(" ")[0],
                    env=env
                )
            )
            if only_train_set:
                self.enabled = False
                _log_warning('Only training set found, disabling early stopping.')
                return

        n_metrics = len({m[1] for m in env.evaluation_result_list})
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all(t >= 0 for t in self.min_delta):
                raise ValueError('Values for early stopping min_delta must be non-negative.')
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info('Disabling min_delta for early stopping.')
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(f'Using {self.min_delta[0]} as min_delta for all metrics.')
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError('Must provide a single value for min_delta or as many as metrics.')
                if self.first_metric_only and self.verbose:
                    _log_info(f'Using only {self.min_delta[0]} as early stopping min_delta.')
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError('Early stopping min_delta must be non-negative.')
            if self.min_delta > 0 and n_metrics > 1 and not self.first_metric_only and self.verbose:
                _log_info(f'Using {self.min_delta} as min_delta for all metrics.')
            deltas = [self.min_delta] * n_datasets * n_metrics

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        self.first_metric = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            self.es.append(copy.deepcopy(self.es_template))
            self.best_iter.append(0)
            self.best_score_list.append(None)
            if eval_ret[3]:  # greater is better
                self.best_score.append(float('-inf'))
                self.cmp_op.append(partial(self._gt_delta, delta=delta))
            else:
                self.best_score.append(float('inf'))
                self.cmp_op.append(partial(self._lt_delta, delta=delta))

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError(
                "early_stopping() callback enabled but no evaluation results found. This is probably a bug in LightGBM. "
                "Please report it at https://github.com/microsoft/LightGBM/issues"
            )

        if self.enable_es_oof and not self.early_stop_oof:
            self._update_es_oof(env=env)

        if self.enable_es:
            self._update_es(env=env)

        if self.enable_es and self.early_stop and self.early_stop_oof:
            if self.verbose:
                eval_result_str = '\t'.join([_format_eval_result(x, show_stdv=True) for x in self.best_score_list[self.early_stop_idx]])
                logger.log(15, f"Early stopping, best iteration is:\n[{self.best_iter[self.early_stop_idx] + 1}]\t{eval_result_str}")
            raise EarlyStopException(self.best_iter[self.early_stop_idx], self.best_score_list[self.early_stop_idx])

        if self.time_limit is not None:
            self._time_early_stop(env=env)

        # TODO: Identify optimal threshold values for early_stopping based on lack of memory
        if self.enable_mem_check and (env.iteration % 10 == 0):
            self._mem_early_stop(env=env)

    def _update_es(self, env: CallbackEnv):
        # self.best_score_list is initialized to an empty list
        first_time_updating_best_score_list = (self.best_score_list == [])
        for i in self.indices_to_check:
            if self.early_stop:
                continue
            score = env.evaluation_result_list[i][2]
            is_best_iter = False
            if first_time_updating_best_score_list or self.cmp_op[i](score, self.best_score[i]):
                is_best_iter = True
                self.best_score[i] = score
                self.best_iter[i] = env.iteration
                if first_time_updating_best_score_list:
                    self.best_score_list[i] = env.evaluation_result_list
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            if self.first_metric_only and self.first_metric != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if self._is_train_set(
                ds_name=env.evaluation_result_list[i][0],
                eval_name=eval_name_splitted[0],
                env=env
            ):
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)

            self.early_stop = self.es[i].update(cur_round=env.iteration, is_best=is_best_iter)
            self._final_iteration_check(env, eval_name_splitted, i)
            if self.early_stop:
                self.early_stop_idx = i

    # TODO: Be smart about how to calculate y_pred_val, can exploit the fact that we already had predictions from a past call?
    #  Boosting methods should be able to take pred outputs from iter N and use them to speed up pred of same data on iter N+1
    #  How to do this with LightGBM?
    def _update_es_oof(self, env: CallbackEnv):
        # FIXME Calling `env.model.predict` leads to warnings about "No further splits with positive gain, best gain: -inf"
        #  Unsure why this is being logged later in fit only if we do a predict call here...
        #  I haven't figured out a way to supress these warnings, no matter what I do. Even setting predictor verbosity=0 doesn't stop it.
        y_pred_val = env.model.predict(self.X_val, num_iteration=env.iteration + 1)  # Predict on latest iter
        if self.es_wrapper_oof.score_func.needs_pred or self.es_wrapper_oof.score_func.needs_quantile:
            y_score = get_pred_from_proba(y_pred_val, problem_type=self.problem_type)
        else:
            y_score = y_pred_val

        es_oof_output = self.es_wrapper_oof.update(y=self.y_val, y_score=y_score, cur_round=env.iteration, y_pred_proba=y_pred_val)
        self.early_stop_oof = es_oof_output.early_stop

    def get_y_pred_val_oof(self) -> np.ndarray:
        return self.es_wrapper_oof.y_pred_proba_val_best_oof

    def _time_early_stop(self, env: CallbackEnv):
        time_elapsed = time.time() - self.start_time
        time_left = self.time_limit - time_elapsed
        time_per_iter = time_elapsed / (env.iteration + 1)
        if time_left <= time_per_iter:
            extra_msg, best_iter, best_score_list = self._best_iter_msg(env=env)
            logger.log(
                20,
                f"\tRan out of time, early stopping on iteration {env.iteration + 1}.{extra_msg}"
            )
            raise EarlyStopException(best_iter, best_score_list)

    @disable_if_lite_mode()
    def _mem_early_stop(self, env: CallbackEnv):
        available = ResourceManager.get_available_virtual_mem()
        cur_rss = self.mem_status.memory_info().rss

        if cur_rss < self.init_mem_rss:
            self.init_mem_rss = cur_rss
        estimated_model_size_mb = (cur_rss - self.init_mem_rss) / (1024 ** 2)
        available_mb = available / (1024 ** 2)

        model_size_memory_ratio = estimated_model_size_mb / available_mb
        if self.verbose or (model_size_memory_ratio > 0.25):
            logger.debug("Available Memory: " + str(available_mb) + " MB")
            logger.debug("Estimated Model Size: " + str(estimated_model_size_mb) + " MB")

        early_stop_mem = False
        if model_size_memory_ratio > 1.0:
            logger.warning("Warning: Large GBM model size may cause OOM error if training continues")
            logger.warning("Available Memory: " + str(available_mb) + " MB")
            logger.warning("Estimated GBM model size: " + str(estimated_model_size_mb) + " MB")
            early_stop_mem = True

        # TODO: We will want to track size of model as well, even if we early stop before OOM, we will still crash when saving if the model is large enough
        if available_mb < 512:  # Less than 500 MB
            logger.warning("Warning: Low available memory may cause OOM error if training continues")
            logger.warning("Available Memory: " + str(available_mb) + " MB")
            logger.warning("Estimated GBM model size: " + str(estimated_model_size_mb) + " MB")
            early_stop_mem = True

        if early_stop_mem:
            logger.warning(
                "Warning: Early stopped GBM model prior to optimal result to avoid OOM error. Please increase available memory to avoid subpar model quality."
            )

            extra_msg, best_iter, best_score_list = self._best_iter_msg(env=env)
            logger.log(
                20,
                f"\tLow on memory, early stopping on iteration {env.iteration + 1}.{extra_msg}",
            )

            raise EarlyStopException(best_iter, best_score_list)

    def _best_iter_msg(self, env: CallbackEnv) -> tuple[str, int, list]:
        if self.enable_es:
            i = self.indices_to_check[0]
            extra_msg = f" Best iteration is:\n\t[%d]\t%s" % (
                self.best_iter[i] + 1, "\t".join([_format_eval_result(x, show_stdv=False) for x in self.best_score_list[i]])
            )
            best_iter = self.best_iter[i]
            best_score_list = self.best_score_list[i]
        else:
            extra_msg = ""
            best_iter = env.iteration
            best_score_list = []
        return extra_msg, best_iter, best_score_list
