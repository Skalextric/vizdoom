class AuxGameState:
    observations = {'x_distance': 0}

    def __init__(self, actions):
        self.legalActions = actions

    def get_x_distance(self):
        return self.observations['x_distance']
    def set_x_distance(self, value):
        self.observations['x_distance'] = value