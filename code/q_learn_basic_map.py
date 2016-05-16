#!/usr/bin/python
#####################################################################
# This script presents different formats of the screen buffer.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../examples/config/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.

# To see the scenario description go to "../../scenarios/README.md"
#
#####################################################################
# from __future__ import print_function
from code.vizdoom import *
from time import sleep
from random import choice
import cv2
import aux.utilities as utilities

# Learn
from reinforcement_learning.q_learn import q_agent, feature_extractors

game = DoomGame()

# Use other config file if you wish.
game.load_config("../configs/vizdoom_opencv.cfg")

game.set_screen_format(ScreenFormat.RGB24)

# game.set_mode(Mode.SPECTATOR)

game.init()

left = [1, 0, 0]
right = [0, 1, 0]
attack = [0, 0, 1]
actions = {'left': left, 'right': right, 'attack': attack}
#actions = {'left': left, 'right': right}

episodes = 30
# sleep time in ms
sleep_time = 1
learning =  True
# Own doom_agent and gamestate
doom_agent = q_agent.QLearningAgent(feature_extractors.BasicMapExtractor(), actions.keys())

for i in range(episodes):
    # print("Episode #" + str(i + 1))
    # Not needed for the first episode but the loop is nicer.

    if i > 20:
        learning = False
        print 'Learning finised'
    else:
        print 'Currently learning'

    game.new_episode()
    while not game.is_episode_finished():

        # Gets the state and possibly to something with it
        game_state = game.get_state()
        misc = game_state.game_variables

        imgs, features = doom_agent.extractor.getFeatures(game_state, ret_img=True)
        # cv2.imshow('Doom Buffer 0', imgs[0])
        # cv2.waitKey(sleep_time)

        # Get random action
        # action = choice(actions.keys())
        ###Cheating code, not learning!!!###
        ##x_distance = features['x_distance']
        ##w = features['target_size'][0]
        ##action = utilities.cheat_basic(x_distance, w)


        # Makes an action and save the reward.
        action = doom_agent.getAction(game_state)
        r = game.make_action(actions[action])
        nextGameState = game.get_state()

        if learning:
            if not game.is_episode_finished():
                doom_agent.update(game_state, action, r, nextState=nextGameState)
            else:
                doom_agent.update(game_state, action, r)
        #print doom_agent.weights

        '''
        print("State #" + str(s.number))
        print("Game Variables:", misc)
        print("Last Reward:", r)
        print("=====================")
        '''
        sleep(0.02)
        #raw_input()
    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
cv2.destroyAllWindows()
