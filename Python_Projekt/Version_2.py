#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import math
from gym import logger
from gym import spaces
from gym.utils import seeding
import numpy as np

from matplotlib import pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from keras.utils import plot_model

class strecke(gym.Env):
    def __init__(self):
        self.fig, self.ax = plt.subplots(1)
        self.plot, = self.ax.plot(0,0)
        self.ax.relim()
        self.ax.set_autoscale_on(True)
        self.ax.autoscale_view(True, True, True)
        self.time_list = []
        self.state_list = []
        
        self.sollwert = 1
        
        self.last_out = 0
        self.time = 0
        self.T = 1
        self.dt = 1
        self.K = 5
        self.done = 0
        self.state = 0               # Winkel des Systems
        
        high = np.array([1.])
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
#        
#        high = np.array([15])        # Beobachtungsbereichsdefinition
#        # Anzahl möglicher Eingabewerte für die Strecke
#        self.action_space = spaces.Box(np.array([-10.]), np.array([10.]))
#        # Beobachtungsbereich
#        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ Reglereingriff zur regelung der Strecke""" 
        for i in np.arange(0,10):   # strecke soll 10 mal berechnet werden mit input "action"
            output = (self.T/self.dt+ 1)**(-1) * (self.K * action - self.state) +self.state
            self.time += 1          # wir zählen hoch
        self.time_list.append(self.time)
            
        # Ausgabewert aufbauen, spaeter mit Winkelgesch.
        self.state = output     
        self.state_list.append(self.state)
        if (self.time > 200):      
            self.done = 1
        else:
            self.done = 0
            
        #if self.state > (10 + self.sollwert) or self.state < (self.sollwert-)
        reward = 1/self.state**2#(self.sollwert / (self.sollwert - self.state))**2     # lies mal nach du faule sau!
        return np.array([self.state]), reward, self.done, {}
    
    def reset(self):
        """ Startzustand herstellen """
        self.sollwert = 1#, = self.np_random.uniform(low=-10, high=10, size=(1,))
        print(self.sollwert)
        self.state = 0
        np.array(self.state)
        self.state_list = []
        self.time_list = []
        self.time = 0
        return np.array([self.state])
    
    def render(self):
#        self.plot.set_data(self.time_list, self.state_list)
#        plt.gcf().canvas.flush_events()
#        plt.draw()
        pass


def main():
    #ENV_NAME = 'CartPole-v0'
    
    # Get the environment and extract the number of actions available in the Cartpole problem
    #env = CartPoleEnv()  #gym.make(ENV_NAME)
    #env=gym.make('CartPole-v0')
    env = strecke()
#    env.reset()
#    for _ in range(1000): # run for 1000 steps
#        env.render()
#        action = env.action_space.sample() # pick a random action
#        env.step(action) # take action
    #np.random.seed(123)
    #env.seed()
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]
#    nb_actions = env.action_space.n #Anzahl von Aktionen abrufen
    
    # as first layer in a sequential model:
#    model = Sequential()
#    model.add(Dense(32, input_shape=(1,)))
#    model.add(Activation('linear'))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)
    
    # after the first layer, you don't need to specify
    # the size of the input anymore:
#    model.add(Dense(32))
#    model.add(Activation('linear'))
#    model.add(Dense(16))
#    model.add(Activation('relu'))
#    model.add(Dense(1))
#    model.add(Activation('linear'))
#    print(model.summary())
#    plot_model(model, to_file='model.png')
    
    # Next, we build a very simple model.
    actor = Sequential()
    print((1,)+env.observation_space.shape)
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(1))
    actor.add(Activation('relu'))
    actor.add(Dense(3))
    actor.add(Activation('relu'))
    actor.add(Dense(3))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())
    
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    print(env.observation_space.shape)
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(1)(x)
    x = Activation('relu')(x)
    x = Dense(3)(x)
    x = Activation('relu')(x)
    x = Dense(3)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())
    
    memory = SequentialMemory(limit=50000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    ddpg = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=.99, target_model_update=1e-3)
   # print(dqn.nb_actions)
    ddpg.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
    
    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot. 
    ddpg.fit(env, nb_steps=5000, visualize=False, verbose=2, nb_max_episode_steps=200)
    
    ddpg.test(env, nb_episodes=5, visualize=False)
    plt.plot(env.time_list, env.state_list)
    plt.show()
    env.close()
    

if __name__ == "__main__":
    main()