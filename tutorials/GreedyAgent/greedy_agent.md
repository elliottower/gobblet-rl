# Tutorial: Greedy Agent
This tutorial provides a basic example of running the gobblet environment using greedy agents.

The agents are greedy in the sense that they will only choose actions which:
1. Wins the game 
2. Blocks the opponent from winning

The `depth` parameter controls the amount of turns in the future they are able to search through. 

For example, depth 2 means it will consider moves which will set the agent up to win with their next move, regardless of what the opponent does.

This script randomizes the first move for each agent, in order to add variety, and the underlying policy in `greedy_policy.py` additionally enforces that agents cannot repeat any of the previous 3 moves they have made (to avoid getting stuck in a loop).

## Usage:

1. (Optional) Create a virtual environment: `conda create -n gobblet python=3.10`
2. (Optional) Activate the virtual environment: `conda activate gobblet`
3. Install gobblet: run `pip install gobblet-rl` or run `pip install -e .` in the root directory
4. Install requirements for this tutorial: `cd tutorials/GreedyAgent && pip install -r requirements.txt`
5. Run `python tutorial_greedy.py`

