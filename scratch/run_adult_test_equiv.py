from __future__ import annotations

import pandas as pd

import numpy as np
from autogluon.tabular import TabularPredictor
from autogluon_benchmark.tasks.task_wrapper import OpenMLTaskWrapper


# FIXME: Don't hardcode predict_proba calls, base it off of problem type
def verify_es_oof_correctness(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    task: OpenMLTaskWrapper,
    model_key: str,
    model_hyperparameters: dict | None = None,
    eval_metric: str | None = None,
    n_samples_to_test_equiv: int = 10,
    fit_transform_val: bool = False,
    delete_from_disk: bool = True,
    verbosity: int = 0,
):
    """
    Verifies that ES-OOF logic is working correctly.
    This checks that the ES-OOF validation predictions are identical to the predictions obtained for a given sample if it is held-out from training.

    Parameters
    ----------
    train_data
    val_data
    task
    model_key
    model_hyperparameters
    eval_metric
    n_samples_to_test_equiv
    fit_transform_val
    delete_from_disk
    verbosity
    """

    if model_hyperparameters is None:
        model_hyperparameters = {}

    init_kwargs = dict(
        label=task.label,
        eval_metric=eval_metric,
        verbosity=verbosity,
        learner_kwargs=dict(fit_transform_val=fit_transform_val),
    )

    predictor = TabularPredictor(**init_kwargs)

    predictor = predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters={
            model_key: [
                {"ag.es_oof": True, "ag_args": {"name": model_key}, **model_hyperparameters},
            ],
        },
        fit_weighted_ensemble=False,
    )

    model = predictor._trainer.load_model(model_key)
    oof = model.y_pred_proba_val_oof_
    assert oof is not None, f"Model's `y_pred_proba_val_oof_` variable was not set!"

    predictor_norm = TabularPredictor(**init_kwargs)

    # fit with ag.es_oof = False
    predictor_norm = predictor_norm.fit(
        train_data=train_data,
        tuning_data=val_data,
        hyperparameters={
            model_key: [
                {"ag.es_oof": False, "ag_args": {"name": model_key}, **model_hyperparameters},
            ],
        },
        fit_weighted_ensemble=False,
    )

    model_norm = predictor_norm._trainer.load_model(model_key)
    assert model_norm.y_pred_proba_val_oof_ is None, f"Model Norm's `y_pred_proba_val_oof_` variable was set!"

    y_pred_proba = predictor.predict_proba(val_data, model=model_key, as_multiclass=False)

    y_pred_proba_norm = predictor_norm.predict_proba(val_data, model=model_key, as_multiclass=False)
    assert y_pred_proba.equals(y_pred_proba_norm), f"Difference in y_pred_proba when ag.es_oof is True vs False!"

    y_pred_proba = y_pred_proba.to_frame(name="val")
    y_pred_proba["oof"] = oof

    model_info = predictor.info()["model_info"][model_key]

    y_pred_proba_1_holdout = []
    predictors_holdout = []

    for i in range(n_samples_to_test_equiv):
        val_data_1 = val_data.iloc[[i]]
        val_data_n = val_data.drop(index=val_data_1.index.to_list())

        predictor_holdout = TabularPredictor(**init_kwargs)

        predictor_holdout = predictor_holdout.fit(
            train_data=train_data,
            tuning_data=val_data_n,
            hyperparameters={
                model_key: [
                    {"ag_args": {"name": model_key}, **model_hyperparameters},
                ],
            },
            fit_weighted_ensemble=False,
        )

        y_pred_proba_1 = predictor_holdout.predict_proba(val_data_1, model=model_key, as_multiclass=False)
        y_pred_proba_1_holdout.append(y_pred_proba_1)
        predictors_holdout.append(predictor_holdout)

    y_pred_proba_1_holdout = pd.concat(y_pred_proba_1_holdout, axis=0)

    y_pred_proba["holdout"] = y_pred_proba_1_holdout

    y_pred_proba["delta"] = y_pred_proba["holdout"] - y_pred_proba["oof"]
    y_pred_proba = y_pred_proba.head(n_samples_to_test_equiv)

    y_pred_proba["isclose"] = np.isclose(y_pred_proba["delta"], 0, atol=2.e-7)
    y_pred_proba["diff_from_val"] = np.invert(np.isclose(y_pred_proba["val"], y_pred_proba["oof"], atol=2.e-7))

    print(f"model: {model_key}")
    print(y_pred_proba)

    X_internal = predictor._trainer.load_X()

    for i, predictor_holdout in enumerate(predictors_holdout):
        model_holdout_info = predictor_holdout.info()["model_info"][model_key]

        X_internal_2 = predictor_holdout._trainer.load_X()

        training_data_identical = X_internal.equals(X_internal_2)

        if not training_data_identical:
            print(f"Aligned training data preprocessed for predictor_holdout {i+1}: {training_data_identical}")
            for c in X_internal.columns:
                column_identical = X_internal[c].equals(X_internal_2[c])
                print(f"\t{column_identical}\t{c}")
                if not column_identical and X_internal[c].dtype.name == 'category':
                    print(f"\t\t{X_internal[c].cat.categories}")
                    print(f"\t\t{X_internal_2[c].cat.categories}")

        # Checks that model hyperparameters aren't altered based on the validation data
        assert model_info["hyperparameters"] == model_holdout_info["hyperparameters"]

    if delete_from_disk:
        predictor.delete_from_disk(verbose=False, dry_run=False)
        predictor_norm.delete_from_disk(verbose=False, dry_run=False)
        for p in predictors_holdout:
            p.delete_from_disk(verbose=False, dry_run=False)

    assert y_pred_proba["isclose"].all()

    if (y_pred_proba["diff_from_val"] == False).all():
        print(
            "WARNING: LOO predictions are identical to normal validation, "
            "and thus this test is not verifying the correctness of the ES-OOF logic. "
            "Try a different split / model hyperparameters! "
            "Some models such as CatBoost don't perfectly implement ES-OOF for practical reasons. "
            "For this test, you can set early stopping to a large value to avoid this imperfect implementation causing a test failure. "
            "If this issue remains regardless of split / hyperparameters, then ES-OOF might be incompatible with the model. "
            "For example, maybe the model doesn't use early stopping."
        )


if __name__ == '__main__':
    task = OpenMLTaskWrapper.from_name(dataset="adult")

    train_data, test_data = task.get_train_test_split_combined(
        fold=0,
        train_size=100,
    )

    from autogluon.core.utils import generate_train_test_split_combined

    label = task.label
    problem_type = task.problem_type
    eval_metric = "log_loss"

    train_data, val_data = generate_train_test_split_combined(train_data, label=label, problem_type=problem_type, test_size=20)

    model_types = [
        "GBM",
        "CAT",
        "NN_TORCH",
    ]

    model_hyperparameters_dict = {
        "GBM": {
            "num_boost_round": 500,
            "ag.early_stop": 5,
        },
        "CAT": {
            "iterations": 50,
            "ag.early_stop": 50,
            "learning_rate": 0.3,
        },
        "NN_TORCH": {
            "num_epochs": 100,
            "ag.early_stop": 5,
            "num_layers": 3,
            "hidden_size": 64,
            "learning_rate": 0.002,
        },
    }

    n_samples_to_test_equiv = 5

    for model_type in model_types:
        import time
        ts = time.time()
        verify_es_oof_correctness(
            train_data=train_data,
            val_data=val_data,
            task=task,
            model_key=model_type,
            model_hyperparameters=model_hyperparameters_dict.get(model_type, None),
            eval_metric=eval_metric,
            n_samples_to_test_equiv=n_samples_to_test_equiv,
            # fit_transform_val=True,  # If this is specified, it will fail on NN_TORCH.
        )
        te = time.time()
        print(te - ts)
