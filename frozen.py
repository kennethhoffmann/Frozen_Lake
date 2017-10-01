import gym
import numpy as np
import argparse
from scipy.optimize import minimize

env = gym.make('FrozenLake-v0')
env.reset()

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.n
"""print q_table"""

# Q-Learning Parameters
num_episodes = 1500
def run_episodes(G, learning_rate):
    rL= []
    q_table=np.zeros((NUM_STATES, NUM_ACTIONS), dtype=object)

    for i_episode in range(num_episodes):
        state = env.reset()
        rall = 0

        for t in range(100):
            env.render()
            # action = env.action_space.sample()
            action=np.argmax(q_table[state,:] + np.random.randn(1,NUM_ACTIONS) * 1. / (i_episode+1))
            #print "action" + str(action)
            next_state, reward, done, info = env.step(action)
            rall += reward

            # update Q table
            q_table[state, action] += learning_rate * (reward + G * (np.max(q_table[next_state, :])) - q_table[state,action])
            state = next_state

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        rL.append(rall)
    score=sum(rL) / num_episodes
    print "score" + str( sum(rL) / num_episodes)
    print "G:%f L:%f" % (G, learning_rate)
    print "Resulting Q Table"
    print q_table
    return score
# if __name__ == '__main__':
    """
    take input from the command line and run your game
    @see https://docs.python.org/3/library/argparse.html
    for more examples

    parser = argparse.ArgumentParser()
    parser.add_argument('g', type=float,
                        default=1.0, help='gamma')
    parser.add_argument('l', type=float,
                        default=0.5, help='learning rate')
    parser.add_argument('e', type=int, default=1500,
                        help='number of episodes')
    args = parser.parse_args()

    G = args.g
    learning_rate = args.l
    num_episodes  = args.e
    run_episodes(G, learning_rate, num_episodes)
    """

    #run_episodes(0.5 , 0.5, 1500)

#run_episodes(0.5 , 0.5)
x0=np.array([.75, .75])
res = minimize(run_episodes, x0, args=(2,))

