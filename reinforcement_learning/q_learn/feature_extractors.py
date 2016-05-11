
class FeatureExtractor:
    def getFeatures(self, state, action):
        raise Exception('Not implemented function')


class SimpleExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        features = {}
        distance = state.get_x_distance()

        features['x_distance'] = distance

        return features
