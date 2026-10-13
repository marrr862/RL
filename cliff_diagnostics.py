import json
import gymnasium as gym
import numpy as np
import sarsa_ql_assignment as m

def reaches_goal(Q):
    state, seen = 36, set()
    path = [state]
    while state != 47 and state not in seen:
        seen.add(state)
        row, col = divmod(state, 12)
        dr, dc = [(-1,0),(0,1),(1,0),(0,-1)][int(Q[state].argmax())]
        row, col = max(0,min(3,row+dr)),max(0,min(11,col+dc))
        state = row * 12 + col
        if 37 <= state <= 46:
            state = 36
        path.append(state)
    return state == 47, path

result = {}
for name, algo in [('SARSA',m.sarsa_learning),('Q-learning',m.q_learning)]:
    first = []
    def callback(ep,Q,epsilon):
        ok,path = reaches_goal(Q)
        if ok and not first:
            first.append(ep)
    env=gym.make('CliffWalking-v1')
    Q,_=algo(env,num_episodes=500,seed=0,on_episode_end=callback)
    result[name]={'first_goal_episode':first[0] if first else None,'final_path':reaches_goal(Q)[1],
                  'above_cliff_values':Q[24:36].max(axis=1).tolist()}
    env.close()
print(json.dumps(result,indent=2))
