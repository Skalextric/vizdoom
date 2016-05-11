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
from __future__ import print_function
from vizdoom import *
from time import sleep
from time import time
from random import choice
import cv2
import mahotas
import numpy as np
import aux.utilities as utilities
import aux.aux_game_state as aux_state

# Learn
from reinforcement_learning.q_learn import q_agent, feature_extractors

game = DoomGame()

# Use other config file if you wish.
game.load_config("../configs/vizdoom_opencv.cfg")

game.set_screen_format(ScreenFormat.RGB24)

#game.set_mode(Mode.SPECTATOR)

game.init()

left = [1, 0, 0]
right = [0, 1, 0]
attack = [0, 0, 1]
actions = {'left': left, 'right': right, 'attack': attack}

episodes = 10
# sleep time in ms
sleep_time = 20

# Own doom_agent and gamestate
doom_agent = q_agent.QLearningAgent(feature_extractors.SimpleExtractor())
auxstate = aux_state.AuxGameState(actions)

for i in range(episodes):
    print("Episode #" + str(i + 1))
    # Not needed for the first episdoe but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():

        # Gets the state and possibly to something with it
        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        # Gray8 shape is not cv2 compliant
        if game.get_screen_format() in [ScreenFormat.GRAY8, ScreenFormat.DEPTH_BUFFER8]:
            img = img.reshape(img.shape[1], img.shape[2], 1)

        '''OPENCV '''

        blueLower = np.array([0, 0, 0], dtype='uint8')
        blueUpper = np.array([255, 0, 0], dtype='uint8')

        blue = cv2.inRange(img, blueLower, blueUpper)
        blurred = cv2.GaussianBlur(blue, (3, 3), 0)

        (_, contours, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = utilities.points_of_contours(contours)
        (x, y, w, h) = cv2.boundingRect(points)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rectangle_center = (w / 2 + x, h / 2 + y)
        cv2.circle(img, rectangle_center, 3, (0, 2550, 0))

        mid = (img.shape[1] // 2, img.shape[0] // 2)
        x_distance = rectangle_center[0] - mid[0]
        #if x_distance < 0 the enemy is left, else right
        auxstate.set_x_distance(x_distance)


        ###Displaying images###
        #cv2.imshow('Blue', blue)
        cv2.imshow('Doom Buffer', img)
        cv2.waitKey(sleep_time)

        '''END OPENCV'''

        #Get random action
        action = choice(actions.keys())
        ###Cheating code, not learning!!!###
        action = utilities.cheat_basic(x_distance, w)
        # Makes an action and save the reward.
        r = game.make_action(actions[action])

        '''
        print("State #" + str(s.number))
        print("Game Variables:", misc)
        print("Last Reward:", r)
        print("=====================")
        '''
        sleep(.05)
    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")
cv2.destroyAllWindows()
