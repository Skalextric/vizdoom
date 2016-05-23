#!/usr/bin/python

#####################################################################
# This script presents how to make use of game variables to implement
# shaping using health_guided.wad scenario
# Health_guided scenario is just like health_gathering 
# (see "../../scenarios/README.md") but for each collected medkit global
# variable number 1 in acs script (coresponding to USER1) is increased
# by 100.0. It is not considered a part of reward but will possibly
# reduce learning time.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# 
#####################################################################
from vizdoom import *
from random import choice
import itertools as it
from time import sleep
import cv2
from matplotlib import pyplot

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("../configs/health_gathering.cfg")

game.init()

# Creates all possible actions.
actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))

episodes = 10
sleep_time = 0.028
last_total_shaping_reward = 0

cross = cv2.imread("../images/rsz_medikit.jpg")
sift = cv2.xfeatures2d.SIFT_create()
(kps_cross, descs_cross) = sift.detectAndCompute(cross, None)


for i in range(episodes):

    print("Episode #" + str(i + 1))
    # Not needed for the first episdoe but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():
        # Gets the state and possibly to something with it
        s = game.get_state()
        img = s.image_buffer
        misc = s.game_variables

        (kps, descs) = sift.detectAndCompute(img, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs_cross, descs, k=2)


        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        if len(good) > 0:
            print good[0][0]

        img3 = cv2.drawMatchesKnn(cross, kps_cross, img, kps, good, img)

        cv2.imshow('cross', img3)
        cv2.waitKey(2)

        '''
        ******* Template match *********
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 120, 255)
        detected_cross = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(detected_cross)
        (startX, startY) = (maxLoc[0], maxLoc[1])
        (endX, endY) = (startX + template.shape[0], startY + template.shape[1])
        cv2.rectangle(img, (startX, startY), (endY, endX), (0, 255, 0), 2)
        cv2.imshow('edged', edged)
        cv2.imshow('img', img)
        cv2.waitKey(2)
        '''

        # Makes a random action and save the reward.
        r = game.make_action(choice(actions))

        # Retrieve the shaping reward
        sr = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
        sr = sr - last_total_shaping_reward
        last_total_shaping_reward += sr

        print("State #" + str(s.number))
        print("Health:", misc[0])
        print("Last Reward:", r)
        print("Last Shaping Reward:", sr)
        print("=====================")

        # Sleep some time because processing is too fast to watch.
        if sleep_time > 0:
            sleep(sleep_time)

    print("Episode finished!")
    print("total reward:", game.get_total_reward())
    print("************************")

game.close()
