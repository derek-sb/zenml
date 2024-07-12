# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from typing_extensions import Annotated

from zenml import ArtifactConfig, step
from zenml.logger import get_logger

from zenml.integrations.neptune.experiment_trackers.run_state import get_neptune_run
import numpy as np
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

@step(experiment_tracker="neptune_experiment_tracker")
def model_trainer(
    dataset_trn: pd.DataFrame,
    model_type: str = "sgd",
    target: Optional[str] = "target",
    max_iter: int = 1000,
    tol: float = 1e-3,
    validation_fraction: float = 0.1
) -> Annotated[
    ClassifierMixin,
    ArtifactConfig(name="sklearn_classifier", is_model_artifact=True),
]:
    neptune_run = get_neptune_run()
    
    X = dataset_trn.drop(columns=[target])
    y = dataset_trn[target]

    if model_type == "sgd":
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_fraction, random_state=42)

        model = SGDClassifier(
            max_iter=1,
            tol=tol,
            random_state=42
        )
        
        train_scores = []
        val_scores = []

        for epoch in range(max_iter):
            model.partial_fit(X_train, y_train, classes=np.unique(y))
            
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
            
            neptune_run[f"metrics/train_score/epoch_{epoch}"] = train_score
            neptune_run[f"metrics/val_score/epoch_{epoch}"] = val_score
            neptune_run["metrics/train_accuracy"].append(train_score) 
            neptune_run["metrics/val_accuracy"].append(val_score) 

            # Check for early stopping
            if epoch > 5 and np.mean(val_scores[-5:]) < np.mean(val_scores[-6:-1]) - tol:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"SGD training completed after {epoch + 1} iterations")
        logger.info(f"Final training score: {train_scores[-1]}")
        logger.info(f"Final validation score: {val_scores[-1]}")

        # Log final scores
        neptune_run["metrics/final_train_score"] = train_scores[-1]
        neptune_run["metrics/final_val_score"] = val_scores[-1]


        # model = SGDClassifier()
    elif model_type == "rf":
        model = RandomForestClassifier()
        model.fit(
            dataset_trn.drop(columns=[target]),
            dataset_trn[target],
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")
    logger.info(f"Training model {model}...")

    # model.fit(
    #     dataset_trn.drop(columns=[target]),
    #     dataset_trn[target],
    # )
    return model
