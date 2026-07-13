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
"""Evaluation entry point for the HiVT prediction module.

This script mirrors the top-level ``eval.py`` but imports the model
through the ``prediction`` package, making it the canonical way to
evaluate HiVT when it is used as a drop-in prediction module.

Example
-------
.. code-block:: bash

    python -m prediction.evaluate \\
        --root /path/to/dataset \\
        --ckpt_path checkpoints/HiVT-64.ckpt \\
        --gpus 1
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.data import DataLoader

from datasets import ArgoverseV1Dataset
from prediction.predictor import HiVTPredictor


def main() -> None:
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of the Argoverse dataset.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to a Lightning checkpoint (.ckpt).')
    args = parser.parse_args()

    predictor = HiVTPredictor.from_checkpoint(
        checkpoint_path=args.ckpt_path, parallel=True)

    val_dataset = ArgoverseV1Dataset(
        root=args.root, split='val',
        local_radius=predictor.local_radius)
    dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.validate(predictor.model, dataloader)


if __name__ == '__main__':
    main()
