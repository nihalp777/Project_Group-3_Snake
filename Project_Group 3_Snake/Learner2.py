import random
import json
import matplotlib.pyplot as plt

import dataclasses

@dataclasses.dataclass
class GameState:
    distance: tuple
    position: tuple
    surroundings: str
    food: tuple


class Learner(object):
    def __init__(self, display_width, display_height, block_size):
        # Game parameters
        self.display_width = display_width
        self.display_height = display_height
        self.block_size = block_size
        
        # Learning parameters
        self.epsilon = 0.1
        self.lr = 0.7
        self.discount = .5

        # State/Action history
        self.qvalues = self.LoadQvalues()
        self.history = []
        
        # Episode rewards and losses
        self.episode_rewards = []
        self.episode_losses = []  # Initialize the episode_losses list

        # Action space
        self.actions = {
            0: 'left',
            1: 'right',
            2: 'up',
            3: 'down'
        }
        self.actions_inv_dict = {
            'left': 0,
            'right': 1,
            'up': 2,
            'down': 3
        }

    def Reset(self):
        self.history = []
        self.episode_rewards = []

    def LoadQvalues(self, path="qvalues.json"):
        with open(path, "r") as f:
            qvalues = json.load(f)
        return qvalues

    def SaveQvalues(self, path="qvalues.json"):
        with open(path, "w") as f:
            json.dump(self.qvalues, f)
            
    def act(self, snake, food):
        state = self._GetState(snake, food)

        # Epsilon greedy
        rand = random.uniform(0,1)
        if rand < self.epsilon:
            action_key = random.choices(list(self.actions.keys()))[0]
        else:
            state_scores = self.qvalues[self._GetStateStr(state)]
            action_key = state_scores.index(max(state_scores))
        action_val = self.actions[action_key]
        
        # Remember the actions it took at each state
        self.history.append({
            'state': state,
            'action': action_key
            })
        return action_val
    
    def UpdateQValues_1step(self, state, next_state, action_key, reward):
        # TODO: Check for validity of q-learning and update if needed
        state_str = self._GetStateStr(state)
        new_state_str = self._GetStateStr(next_state)
        action = self.actions_inv_dict[action_key]
        self.qvalues[state_str][action] = (1-self.lr) * (self.qvalues[state_str][action]) + self.lr * (reward + self.discount*max(self.qvalues[new_state_str])) # Bellman equation
        cumulative_reward = 0
        cumulative_reward += reward
        self.episode_rewards.append(cumulative_reward)
    
    def UpdateQValues(self, reason):
        # TODO: Check what algorithmm you're using here? Looks like Monte Carlo, not QLearning
        # TODO IF MonteCarlo, double check the update rule6
        history = self.history[::-1]
        cumulative_reward = 0
        reward = 0
        for i, h in enumerate(history[:-1]):
            if reason: # Snake Died -> Negative reward
                sN = history[0]['state']
                aN = history[0]['action']
                state_str = self._GetStateStr(sN)
                reward = -1
                self.qvalues[state_str][aN] = (1-self.lr) * self.qvalues[state_str][aN] + self.lr * reward # Bellman equation - there is no future state since game is over
                reason = None
            else:
                s1 = h['state'] # current state
                s0 = history[i+1]['state'] # previous state
                a0 = history[i+1]['action'] # action taken at previous state
                
                x1 = s0.distance[0] # x distance at current state
                y1 = s0.distance[1] # y distance at current state
    
                x2 = s1.distance[0] # x distance at previous state
                y2 = s1.distance[1] # y distance at previous state
                
                if s0.food != s1.food: # Snake ate a food, positive reward
                    reward = 1
                elif (abs(x1) > abs(x2) or abs(y1) > abs(y2)): # Snake is closer to the food, positive reward
                    reward = 1 # TODO: suggest trying 0.001 and -0.001
                else:
                    reward = -1 # Snake is further from the food, negative reward
                    
                state_str = self._GetStateStr(s0)
                new_state_str = self._GetStateStr(s1)
                self.qvalues[state_str][a0] = (1-self.lr) * (self.qvalues[state_str][a0]) + self.lr * (reward + self.discount*max(self.qvalues[new_state_str])) # Bellman equation
                
            # Append episode reward
            cumulative_reward += reward
        # TODO This is not correct because the reward should be calculated externally
        # TODO The reward should be provided every step and accumulated externally and saved every episode, which is not working with the current mix of Monte carlo and QLearning
        self.episode_rewards.append(cumulative_reward)


    def _GetState(self, snake, food):
        snake_head = snake[-1]  
        dist_x = food[0] - snake_head[0]
        dist_y = food[1] - snake_head[1]

        if dist_x > 0:
            pos_x = '1' # Food is to the right of the snake
        elif dist_x < 0:
            pos_x = '0' # Food is to the left of the snake
        else:
            pos_x = 'NA' # Food and snake are on the same X file

        if dist_y > 0:
            pos_y = '3' # Food is below snake
        elif dist_y < 0:
            pos_y = '2' # Food is above snake
        else:
            pos_y = 'NA' # Food and snake are on the same Y file

        sqs = [
            (snake_head[0]-self.block_size, snake_head[1]),   
            (snake_head[0]+self.block_size, snake_head[1]),         
            (snake_head[0],                  snake_head[1]-self.block_size),
            (snake_head[0],                  snake_head[1]+self.block_size),
        ]
        
        surrounding_list = []
        for sq in sqs:
            if sq[0] < 0 or sq[1] < 0: # off screen left or top
                surrounding_list.append('1')
            elif sq[0] >= self.display_width or sq[1] >= self.display_height: # off screen right or bottom
                surrounding_list.append('1')
            elif sq in snake[:-1]: # part of tail
                surrounding_list.append('1')
            else:
                surrounding_list.append('0')
        surroundings = ''.join(surrounding_list)

        return GameState((dist_x, dist_y), (pos_x, pos_y), surroundings, food)

    def _GetStateStr(self, state):
        return str((state.position[0],state.position[1],state.surroundings))
            
    def CalculateLoss(self):
    # Placeholder for loss calculation logic
        return random.random()  # Example: use your actual loss calculation here

    def PlotEpisodeMetrics(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_losses)
        plt.title("Episode Losses")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.show()

# Example usage at the end of the script
if __name__ == "__main__":
    learner = Learner(1000, 800, 10)
    # Simulation of learning process
    learner.PlotEpisodeMetrics()