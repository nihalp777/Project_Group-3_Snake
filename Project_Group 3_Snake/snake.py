#%%
import pygame
import random
import Learner2 as Learner
import matplotlib.pyplot as plt
import time

pygame.init()

#%% CONSTANTS
YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

BLOCK_SIZE = 10 
DIS_WIDTH = 1000
DIS_HEIGHT = 800

QVALUES_N = 100
FRAMESPEED = 50

#%% Game 

def GameLoop(visualize=True):
    global dis
    
    if visualize:
        dis = pygame.display.set_mode((DIS_WIDTH, DIS_HEIGHT))
        pygame.display.set_caption('Snake')
    clock = pygame.time.Clock()

    # Starting position of snake
    x1 = DIS_WIDTH / 2
    y1 = DIS_HEIGHT / 2
    x1_change = 0
    y1_change = 0
    snake_list = [(x1,y1)]
    length_of_snake = 1

    # Create first food
    foodx = round(random.randrange(0, DIS_WIDTH - BLOCK_SIZE) / 10.0) * 10.0
    foody = round(random.randrange(0, DIS_HEIGHT - BLOCK_SIZE) / 10.0) * 10.0

    dead = False
    reason = None
    while not dead:
        # Get action from agent
        state = learner._GetState(snake_list, (foodx, foody))
        # TODO get reward based on your logic not only score
        reward = 0
        action = learner.act(snake_list, (foodx,foody))
        if action == "left":
            x1_change = -BLOCK_SIZE
            y1_change = 0
        elif action == "right":
            x1_change = BLOCK_SIZE
            y1_change = 0
        elif action == "up":
            y1_change = -BLOCK_SIZE
            x1_change = 0
        elif action == "down":
            y1_change = BLOCK_SIZE
            x1_change = 0

        # Move snake
        x1 += x1_change
        y1 += y1_change
        snake_head = (x1,y1)
        snake_list.append(snake_head)
        # Check if snake is off screen
        if x1 >= DIS_WIDTH or x1 < 0 or y1 >= DIS_HEIGHT or y1 < 0:
            reason = 'Screen'
            dead = True

        # Delete the last cell since we just added a head for moving, unless we ate a food
        if len(snake_list) > length_of_snake:
            del snake_list[0]
            
        # Check if snake hit tail
        if snake_head in snake_list[:-1]:
            reason = 'Tail'
            dead = True

        # Check if snake ate food
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, DIS_WIDTH - BLOCK_SIZE) / 10.0) * 10.0
            foody = round(random.randrange(0, DIS_HEIGHT - BLOCK_SIZE) / 10.0) * 10.0
            length_of_snake += 1
            reward = 1

        # Draw food, snake and update score
        if visualize:
            dis.fill(BLUE)
            DrawFood(foodx, foody)
            DrawSnake(snake_list)
            DrawScore(length_of_snake - 1)
            pygame.display.update()

        # Update Q Table
        learner.UpdateQValues(reason)
        # next_state = learner._GetState(snake_list, (foodx, foody))
        # learner.UpdateQValues_1step(state, next_state, action, reward)
        
        # Next Frame
        clock.tick(FRAMESPEED)
    time.sleep(1)
    # print(f"Latest action: {action}")

    return length_of_snake - 1, reason

def DrawFood(foodx, foody):
    pygame.draw.rect(dis, GREEN, [foodx, foody, BLOCK_SIZE, BLOCK_SIZE])   

def DrawScore(score):
    font = pygame.font.SysFont("comicsansms", 35)
    value = font.render(f"Score: {score}", True, YELLOW)
    dis.blit(value, [0, 0])

def DrawSnake(snake_list):
    for x in snake_list:
        pygame.draw.rect(dis, BLACK, [x[0], x[1], BLOCK_SIZE, BLOCK_SIZE])



#%%
game_count = 1

learner = Learner.Learner(DIS_WIDTH, DIS_HEIGHT, BLOCK_SIZE)

# Lists to store the scores and episode numbers for the learning curve plot
episode_nums = []
scores = []

game_count = 1
while True:
    learner.Reset()
    # TODO change this to be a decaying epsilon
    if game_count > 100:
        learner.epsilon = 0.1
    else:
        learner.epsilon = .15666
    score, reason = GameLoop(visualize=False)
    print(f"Games: {game_count}; Score: {score}; Reason: {reason}") # Output results of each game to console to monitor as agent is training
    
    # Append the episode number and score to the lists
    episode_nums.append(game_count)
    scores.append(score)
    
    game_count += 1
    if game_count % QVALUES_N == 0: # Save qvalues every qvalue_dump_n games
        print("Save Qvals")
        learner.SaveQvalues(path="class_test.json")
    if game_count % 20 == 0:
    # Plot the learning curve
        plt.plot(episode_nums, scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Snake Game Learning Curve")
        plt.grid(True)
        plt.pause(0.001)  # Pauses the plot briefly to update the display

    # Check for game termination condition (if any)
    if game_count > 1000:  # Specify the maximum number of games
        break

# Display the final learning curve plot
#plt.show()
learner.PlotEpisodeMetrics()