class BaseMetric:
    def __init__(self):
        self.score = 0
        self.n = 0

    def clean(self):
        self.score = self.n = 0

    def evaluate(self):
        return self.score / self.n


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

        self.clean()

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        preds = preds.sigmoid().sum(-1).round().long()
        targets = targets.sum(-1).long()

        self.score += (targets == preds).sum().item()
        self.n += len(preds)


class Score(BaseMetric):
    def __init__(self):
        super().__init__()

        self.error = [
            [0.0, 0.1, 0.7, 1.0],
            [0.1, 0.0, 0.3, 0.7],
            [0.7, 0.3, 0.0, 0.3],
            [1.0, 0.7, 0.3, 0.0],
        ]

        self.clean()

    def update(self, preds, targets):
        assert len(preds) == len(targets)

        preds = preds.sigmoid().sum(-1).round().long()
        targets = targets.sum(-1).long()

        for p, t in zip(preds, targets):
            self.score += 1 - self.error[t][p]

        self.n += len(preds)
