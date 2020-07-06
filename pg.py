
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
from Env import *

import numpy as np

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time


relay_net_slow, relay_net_fast= create_example_env()
State,Reward=relay_net_slow.update_state()
d_state=len(State)
print("State dimension: " + str(d_state))
env=relay_net_slow
Action_space=env.get_possible_action_space()
    
#print(Action_space)



input_shape=[len(State)]  #total dimension of the observation
output_shape=len(Action_space)  #action space dim
steps_per_episode=100




print("input/output dims of the DQN: " + str((input_shape, output_shape)))

RL = PolicyGradient(
    n_actions=25,
    n_features=4,
    learning_rate=0.003,
    reward_decay=0.95,
    # output_graph=True,
)

for i_episode in range(100):
    

    env,dummy= create_example_env()
    state,reward=env.update_state()
    state=np.array(state)


        
    for i in range(steps_per_episode):

        action_id = RL.choose_action(state)
        action_space=env.get_possible_action_space()
        action=action_space[action_id]
        
        env.apply_action(action)

        next_state,reward=env.update_state()
        next_state=np.array(next_state)
         

        RL.store_transition(state, action_id, reward)
        state=next_state
        #print(state)
        
        
    ep_rs_sum = sum(RL.ep_rs)

    if 'running_reward' not in globals():
        running_reward = ep_rs_sum
    else:
        running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
    if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
    print("episode:", i_episode, "  reward:", int(running_reward))

    vt = RL.learn()
    if i_episode == 0:
        plt.plot(vt)    # plot the episode vt
        plt.xlabel('episode steps')
        plt.ylabel('normalized state-action value')
        plt.show()

    
            

        
    
