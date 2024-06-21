from atmosnav import Agent
print(Agent.__class__)
from atmosnav.agents.controllers import Controller
from atmosnav.agents.dynamics import Dynamics

print(Agent.__class__, Controller.__class__, Dynamics.__class__)