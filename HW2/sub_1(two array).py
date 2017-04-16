from __future__ import print_function
import numpy as np
from tkinter import *
import time

WORLD_SIZE = 4
REWARD = -1.0
ACTION_PROB = []

world = np.zeros((WORLD_SIZE, WORLD_SIZE))

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

for i in range(0, WORLD_SIZE):
    ACTION_PROB.append([])
    for j in range(0, WORLD_SIZE):
        ACTION_PROB[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))

nextState = []
for i in range(0, WORLD_SIZE):
    nextState.append([])
    for j in range(0, WORLD_SIZE):
        next = dict()
        if i == 0:
            next['U'] = [i, j]
        else:
            next['U'] = [i - 1, j]

        if i == WORLD_SIZE - 1:
            next['D'] = [i, j]
        else:
            next['D'] = [i + 1, j]

        if j == 0:
            next['L'] = [i, j]
        else:
            next['L'] = [i, j - 1]

        if j == WORLD_SIZE - 1:
            next['R'] = [i, j]
        else:
            next['R'] = [i, j + 1]

        nextState[i].append(next)

states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        if (i == 0 and j == 0) or (i == WORLD_SIZE - 1 and j == WORLD_SIZE - 1):
            continue
        else:
            states.append([i, j])


def policy_improvement():
    for i, j in states:
        max = -99999.0
        max_cnt = 0.0
        for action0 in actions:
            newPosition = nextState[i][j][action0]
            if max < world[newPosition[0], newPosition[1]]:
                max = world[newPosition[0], newPosition[1]]

        for action1 in actions:
            newPosition = nextState[i][j][action1]
            if max == world[newPosition[0], newPosition[1]]:
                max_cnt += 1.0

        for action2 in actions:
            newPosition = nextState[i][j][action2]
            if max == world[newPosition[0], newPosition[1]]:
                ACTION_PROB[i][j][action2] = 1.0/max_cnt
            else:
                ACTION_PROB[i][j][action2] = 0.0

idx = 0
# for figure 4.1
start_time = time.time()
while True:
    # keep iteration until convergence
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i, j in states:
        for action in actions:
            newPosition = nextState[i][j][action]
            # bellman equation
            newWorld[i, j] += ACTION_PROB[i][j][action] * (REWARD + world[newPosition[0], newPosition[1]])

    if np.sum(np.abs(world - newWorld)) < 1e-4:
        print('Random Policy')
        print(newWorld)
        break
    world = newWorld

    policy_improvement()

    if idx == 0 or idx == 1 or idx == 3 or idx == 10:
        print(world)
        master = Tk()
        w = Canvas(master, width=900, height=900)
        w.pack()
        i_cnt = 0
        for i in range(20, 220, 50):
            j_cnt = 0
            for j in range(20, 220, 50):
                w.create_rectangle(j, i, j + 50, i + 50)

                if (j_cnt == 0 and i_cnt == 0) or (j_cnt == 3 and i_cnt == 3):
                    pass
                else:
                    # ['L', 'U', 'R', 'D']
                    x1 = (j + (j + 50)) / 2
                    y1 = (i + (i + 50)) / 2

                    if ACTION_PROB[i_cnt][j_cnt]['L'] > 0:
                        w.create_line(x1, y1, x1 - 20, y1, arrow=LAST)
                    if ACTION_PROB[i_cnt][j_cnt]['U'] > 0:
                        w.create_line(x1, y1, x1, y1 - 20, arrow=LAST)
                    if ACTION_PROB[i_cnt][j_cnt]['R'] > 0:
                        w.create_line(x1, y1, x1 + 20, y1, arrow=LAST)
                    if ACTION_PROB[i_cnt][j_cnt]['D'] > 0:
                        w.create_line(x1, y1, x1, y1 + 20, arrow=LAST)

                j_cnt += 1
            i_cnt += 1

        mainloop()
        w.delete()
    idx += 1
duration = time.time() - start_time
print('Two array version time: %.3fsec' % duration)