
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, rewards, mean_rewards):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score and Reward')
    plt.plot(scores, label='Score', color='c')  # Cyan for scores
    plt.plot(mean_scores, label='Mean Score', color='m')  # Magenta for mean scores
    plt.plot(rewards, label='Reward', color='y')  # Yellow for rewards
    plt.plot(mean_rewards, label='Mean Reward', color='k')  # Black for mean rewards
    plt.ylim(ymin=0)
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)

