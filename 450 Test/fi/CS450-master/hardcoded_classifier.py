class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, data):
        targets = []

        for row in data:
            targets.append(self.predict_one(row))

        return targets

    def predict_one(self, row):
        # The hard coded model always predicts 0
        return 0


class HardCodedClassifier:
    def __init__(self):
        pass

    def fit(self, data, targets):
        return HardCodedModel()
