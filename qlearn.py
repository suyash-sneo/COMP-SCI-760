import numpy as np

ALPHA = 0.5
GAMMA = 0.8
EPSILON = 0.5
NUM_STEPS = 200

NUM_STATE = 2           # 0-A; 1-B
NUM_ACTIONS = 2         # 0-stay; 1-move
rewards = np.array([[1, 0], [1, 0]])        # A-stay, A-move, B-stay, B-move

# Get next state given current state and action taken
def step(state, action):
    next_state = state
    if action == 1:
        next_state = 1-state
    return next_state

def get_greedy_action(q, state):
    actions = q[state]
    if actions[1] >= actions[0]:
        return 1
    return 0

def get_epsilon_greedy_action(q, state):
    rv = np.random.uniform()
    if rv <= EPSILON:
        return np.random.choice(np.arange(2))
    else:
        return np.argmax(q[state])

def initialize():
    q = np.zeros((2, 2))
    initial_state = 0           # A
    return q, initial_state, get_epsilon_greedy_action

def main():
    q, state, get_next_action = initialize()
    for _ in range(NUM_STEPS):
        action = get_next_action(q, state)
        next_state = step(state, action)
        reward = rewards[state][action]
        q_max = np.max(q[next_state])
        print("q_max:",q_max)
        q[state][action] = ((1-ALPHA)*q[state][action]) + (ALPHA * (reward + (GAMMA * q_max)))
        print("step:", step, "; state:", state, "; action:", action, "; q_val:", q[state][action])
        state = next_state
    
    print(q)

main()