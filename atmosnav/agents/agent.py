from .controllers import Controller
from .dynamics import Dynamics

class Agent:
    def __init__(self, controller: Controller, dynamics: Dynamics):
        self.controller = controller
        self.dynamics = dynamics