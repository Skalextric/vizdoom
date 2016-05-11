class QLearningAgent:
    def __init__(self, actions, extractor, state):
        self.actions = actions
        self.weights = {}
        self.extractor = extractor

    def getActions(self):
        return self.actions

    def getQValue(self, state, action):
        features = self.extractor(state, action)
        q_value = 0.0
        for key in features:
            feature_value = features[key]
            if key in self.weights:
                q_value += self.weights[key] * feature_value
            else:
                self.weights[key] = 0.0
                q_value += self.weights[key] * feature_value
        return q_value


