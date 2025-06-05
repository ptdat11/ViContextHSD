from typing import Literal, Callable


class EarlyStopper:
    def __init__(
        self,
        patience: int = 5,
        smoothing_alpha: float | None = None,
        policy: Literal["min", "max"] = "max"
    ):
        self.patience = patience
        self.smoothing_alpha = smoothing_alpha if smoothing_alpha is not None else 1.0
        self.policy = policy
        self.best_score = None
        self.waited = 0
        self.smoothed_score = None

    def __call__(
        self,
        score: float,
        on_improvement: Callable[[float], None] = None
    ) -> bool:
        # Apply exponential smoothing if alpha < 1.0
        if self.smoothed_score is None:
            self.smoothed_score = score
        else:
            alpha = self.smoothing_alpha
            self.smoothed_score = alpha * score + (1 - alpha) * self.smoothed_score

        # Determine comparison direction
        current = -self.smoothed_score if self.policy == "min" else self.smoothed_score
        best = -self.best_score if (self.best_score is not None and self.policy == "min") else self.best_score
        print(current, best)

        # Check for improvement
        if self.best_score is None or current > best:
            self.best_score = self.smoothed_score
            self.waited = 0
            if on_improvement is not None:
                on_improvement(score)  # Note: pass raw score here
        else:
            self.waited += 1

        return self.waited >= self.patience