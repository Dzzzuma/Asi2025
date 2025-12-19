class DummyModel:
    def predict(self, X):
        return [1.0] * len(X)
