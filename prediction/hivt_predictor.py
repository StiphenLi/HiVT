from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Iterable, List, Optional, Tuple


class HiVTPredictor:

    def __init__(self,
                 checkpoint_path: str,
                 parallel: bool = True,
                 device: Optional[str] = None,
                 model_loader: Optional[Callable[..., Any]] = None) -> None:
        if not checkpoint_path:
            raise ValueError('checkpoint_path must not be empty')

        if model_loader is None:
            from models.hivt import HiVT
            model_loader = HiVT.load_from_checkpoint

        self.model = model_loader(checkpoint_path=checkpoint_path, parallel=parallel)
        if hasattr(self.model, 'eval'):
            self.model.eval()
        if device is not None and hasattr(self.model, 'to'):
            self.model.to(device)

    def predict(self, data: Any) -> Tuple[Any, Any]:
        context = nullcontext()
        try:
            import torch
            context = torch.no_grad()
        except ImportError:
            pass

        with context:
            outputs = self.model(data)

        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise ValueError('HiVT model must return a tuple: (y_hat, pi)')
        return outputs

    def predict_batch(self, batch: Iterable[Any]) -> List[Tuple[Any, Any]]:
        return [self.predict(data) for data in batch]
