# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HiVT prediction module.

This package wraps the HiVT trajectory-prediction model behind a clean,
model-agnostic interface so that it can serve as a drop-in replacement
for any existing prediction module.

Public API
----------
HiVTConfig
    Dataclass holding all model hyper-parameters.  Use the
    ``HiVTConfig.hivt_64()`` or ``HiVTConfig.hivt_128()`` factory
    methods to obtain the two standard configurations.

HiVTPredictor
    High-level wrapper around the :class:`~models.hivt.HiVT` Lightning
    module.  Provides ``predict`` / ``predict_agent`` methods and a
    ``from_checkpoint`` factory for loading pretrained weights.

Example
-------
>>> from prediction import HiVTConfig, HiVTPredictor
>>>
>>> # Build a fresh model
>>> config = HiVTConfig.hivt_64()
>>> predictor = HiVTPredictor(config)
>>> predictions, scores = predictor.predict(data)
>>>
>>> # Load pretrained weights
>>> predictor = HiVTPredictor.from_checkpoint("checkpoints/HiVT-64.ckpt")
>>> predictions, scores = predictor.predict(data)
"""
from prediction.config import HiVTConfig
from prediction.predictor import HiVTPredictor

__all__ = ['HiVTConfig', 'HiVTPredictor']
