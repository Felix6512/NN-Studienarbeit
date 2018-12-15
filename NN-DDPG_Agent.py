import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from matplotlib import pyplot as plt

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        """self.max_speed=8
        self.max_out=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_out, high=self.max_out, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        
        """
        #self.fig, self.ax = plt.subplots(1)
        #self.plot, = self.ax.plot(0,0)
        #self.ax.relim()
        #self.ax.set_autoscale_on(True)
        #self.ax.autoscale_view(True, True, True)
        self.time_list = []
        self.state_list = []
        
        self.sollwert = 0
        
        self.last_out = 0
        
        self.time = 0
        self.winkel = []
        self.time_array = []
        self.winkel_ges = []
        
        self.T = 5
        self.dt = 1
        self.K = 1
        
        self.done = 0
        self.state = np.array([0, 0])               # Winkel des Systems
        self.last_theta = 0
        
        self.max_speed=8
        self.max_out=1.
        self.dt=1
        self.viewer = None

        high = np.array([1.1, 1.1, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_out, high=self.max_out, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        """th, thdot = self.state # th := theta

#        g = 10.
#        m = 1.
#        l = 1.
#        dt = self.dt

        #u = np.clip(u, -self.max_torque, self.max_torque)[0]
        #self.last_u = u # for rendering
        costs = 1 # angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        #newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        #newth = th + newthdot*dt
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        #self.state = np.array([0, 0]) #self.state = np.array([newth, newthdot])
        return np.array([0,0,0]), -costs, False, {} #1. = self.obs"""
        
        """ Reglereingriff zur regelung der Strecke"""
        u = u[0]
        print(u)
        T = self.T
        dt = self.dt
        K = self.K
        th, thdot = self.state
        
        for i in np.arange(0, 100):
            if (self.time == (1000 * i)):
                self.sollwert = np.random.random()*2-1
        
        #soll = np.sin(self.time/1000)
        #temp_th = th #für berechnung von thdot
        costs = 0
        for i in np.arange(0, 10):
            #self.last_theta = th
            """output = (self.T/self.dt+ 1)**(-1) * (self.K * regelwert - self.last_out) +self.last_out
            self.last_out = output"""
            th = (T/dt + 1)**(-1) * (K * u - self.last_theta) + self.last_theta
            self.last_theta = th
            if th>1:print(th)
            if th<-1:print(th)
            
            self.winkel.append(th)
            self.time_array.append(self.time)
            self.time += 1    
            self.winkel_ges.append(self.sollwert)
        self.state = np.array([th, thdot])
        th = np.clip(th, -1, 1)
        costs = (self.sollwert - th)**2 /2
        costs = np.clip(costs, -1, 1)
        #u = np.clip(u, -self.K, self.K)  # -K <= Regelwert u <= K
        self.last_u = u
        
        
        return np.array([th, th, self.sollwert]), -costs, False, {} #[th,(th-temp_th)/dt,th]

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.array([0, 0]) #self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return np.array([0,0,0]) #self.obs

#    def _get_obs(self):
#        theta, thetadot = self.state
#        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        return
#        if self.viewer is None:
#            from gym.envs.classic_control import rendering
#            self.viewer = rendering.Viewer(500,500)
#            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
#            rod = rendering.make_capsule(1, .2)
#            rod.set_color(.8, .3, .3)
#            self.pole_transform = rendering.Transform()
#            rod.add_attr(self.pole_transform)
#            self.viewer.add_geom(rod)
#            axle = rendering.make_circle(.05)
#            axle.set_color(0,0,0)
#            self.viewer.add_geom(axle)
#            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
#            self.img = rendering.Image(fname, 1., 1.)
#            self.imgtrans = rendering.Transform()
#            self.img.add_attr(self.imgtrans)
#
#        self.viewer.add_onetime(self.img)
#        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
#        if self.last_u:
#            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
#
#        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    def plot(self):
        plt.plot(self.time_array, self.winkel)
        plt.plot(self.time_array, self.winkel_ges)
        plt.gca().set_ylim(-2,2)
        plt.grid()
        #print(winkel[-1])
        plt.show()
        print(self.state)
        #print(t.shape)
        
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


# Get the environment and extract the number of actions.
env = PendulumEnv() #gym.make('Pendulum-v0')



np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=5000, visualize=False, verbose=1, nb_max_episode_steps=200) #steps 50000

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format('Pendulum-V0'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=2, visualize=False, nb_max_episode_steps=1000)
env.close()
env.plot()