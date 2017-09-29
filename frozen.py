import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.reset()

NUM_ACTIONS=env.action_space.n
NUM_STATES= env.observation_space.n
q_table=np.zeros((NUM_STATES, NUM_ACTIONS))
"""print q_table"""


# Q-Learning Parameters

G=.99
L=.75

rL= []


#Learning
episodes=1500

for i_episode in range(episodes):
	state= env.reset()
	
	rall=0

	for t in range(100):
	    env.render()
	    # action = env.action_space.sample()
	    action=np.argmax(q_table[state,:] + np.random.randn(1,NUM_ACTIONS) * 1./(i_episode+1))
	    

	    next_state, reward, done, info = env.step(action)

	    rall+=reward
	    # update Q table
	    q_table[state,action]+= L * ( reward + G * (np.max(q_table[next_state,:]))-q_table[state,action])

	    state=next_state
	    if done:
	        print("Episode finished after {} timesteps".format(t+1))
	        break
	rL.append(rall)



print "score" + str(sum(rL)/episodes)
print "G:%f L:%f" %(G,L)
print "Resulting Q Table"
print q_table


