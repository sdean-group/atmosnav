from atmosnav import *

# 


class ActionSupplier(abc.ABC):
    @abc.abstractmethod
    def get_action(self, t, state):
        """"""

def lerp(a, b, t):
    return a + t * (b - a)

class Plan(ActionSupplier):
    def __init__(self, plan_array, start_time):
        self.plan_array = plan_array
        self.start_time = start_time


    def get_action(self, time, state):
        idx = (time - self.start_time) // self.waypoint_time_step
        theta = (time - self.start_time - idx * self.waypoint_time_step) / float(self.waypoint_time_step)
        return lerp(self.plan_array[idx], self.plan_array[(idx+1)], theta), self
    
    def tree_flatten(self):
        children = (self.start_time, )  # arrays / dynamic values
        aux_data = {'waypoint_time_step': self.waypoint_time_step}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return Plan(children[0], aux_data['waypoint_time_step'])


class BradleyAgent(ActionSupplier):
    # def learn()
    def learn(self, reward, observation): pass
    def get_action(self):pass


class JannaAgent:
    def learn(self, odfjasdlkfj, adskfjaskjld) -> 'action': pass
    def get_action_supplier(): pass

# def bradley_train(...): pass

# def janna_train(...): pass

def run(start_time, time_elapsed, airborne, action_supplier, wind):
    time = start_time
    log = []

    while time < start_time + time_elapsed:
        wind_vector = wind.get_direction()
        action = action_supplier.get_action(time, airborne.state)
        airborne, info = airborne.step(time, action, wind_vector)

        # update log
        time += 60*10

    return log




def step_balloon(airborne, time, action, wind, reward_fn):
    wind_vector = wind.get_direction()
    airborne, info, done = airborne.step(time, action, wind_vector)
    reward = reward_fn(airborne.state, wind_vector, ...)
    return time + 60*10, reward, airborne, done


def train(start_time, reward_fn, airborne, agent, wind):
    time = start_time
    done = False
    reward = 123456789
    action = begin_episode() #? 
    while not done:
        time, reward, airborne, done = step_balloon(time, reward_fn, airborne, action)
        action = agent.get_action_and_learn(reward, airborne.state)





class BalloonEnv(env.gym):

    def step(self):
        airborne  = airborne.step(action)
        obs = airborne.state
        reward = calculate_reward(obs)

        # agent.