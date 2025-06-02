from typing import Literal, Callable

class EarlyStopper:
    def __init__(self, patience: int = 5, policy: Literal["min", "max"] = "max"):
        self.patience = patience
        self.policy = policy
        self.best_score = None
        self.waited = 0

    def __call__(
        self, 
        score: float, 
        on_improvement: Callable[[float], None] = None
    ):
        if self.policy == "min":
            _score = -score
            _best_score = -self.best_score
        else:
            _score = score
            _best_score = self.best_score

        if self.best_score is None or _score > _best_score:
            if on_improvement is not None:
                on_improvement(score)

            self.best_score = score
            self.waited = 0
        else:
            self.waited += 1

        return self.waited >= self.patience