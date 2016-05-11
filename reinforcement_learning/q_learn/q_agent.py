class QLearningAgent:
    def __init__(self, extractor):
        self.weights = {}
        self.extractor = extractor
        self.qvalues = {}

        # Learning rate
        self.alpha = 0.5

    def getQValue(self, state, action):
        features = self.extractor.getFeatures(state, action)
        q_value = 0.0
        for key in features:
            feature_value = features[key]
            if key in self.weights:
                q_value += self.weights[key] * feature_value
            else:
                self.weights[key] = 0.0
                q_value += self.weights[key] * feature_value
        return q_value

    def computeValueFromQValues(self, state):
        value_list = []
        for action in self.getLegalActions(state):
            value_list.append(self.getQValue(state, action))
        if len(value_list) == 0:
            value = 0.0
        else:
            value = max(value_list)
        return value

    def computeActionFromQValues(self, state):
        max_value = -float('inf')
        best_action = None
        for action in self.getLegalActions(state):
            actual_value = self.getQValue(state, action)
            if actual_value > max_value:
                max_value = actual_value
                best_action = action
        return best_action

    def update(self, state, action, nextState, reward):
        diff = (reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action))
        features = self.extractor.getFeatures(state, action)
        for key in self.weights:
            self.weights[key] = self.weights[key] + self.alpha * diff * features[key]

    def getLegalActions(self, state):
        return state.legalActions
