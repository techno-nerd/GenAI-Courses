import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
#Meausure of optimising future reward - Value of future reward vs current reward
DISCOUNT = 0.95
EPISODES = 10000
epsilon = 0.5 #How much randomness, how much exploration to do - Higher = more random
start_epsilon_decay = 1
end_epsilon_decay = EPISODES // 2
epsilon_decay_value = epsilon/(end_epsilon_decay-start_epsilon_decay)

#Size of the Q-table - Break the range between high and low to 20 buckets
DISCRETE_OS = [20] * len(env.observation_space.high)
OS_window_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / OS_window_size
    return tuple(discrete_state.astype(int))

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS + [env.action_space.n]))


for episode in range(0, EPISODES+1):

    if episode%1000 == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())

    done  = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, info = env.step(action)
        
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = ((1 - LEARNING_RATE) * current_q) + (LEARNING_RATE*(reward + DISCOUNT*max_future_q))

            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Task completed on Episode {episode}")
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    
    if end_epsilon_decay >= episode >= start_epsilon_decay:
        epsilon -= epsilon_decay_value

env.close()