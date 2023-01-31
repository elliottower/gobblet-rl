# gobblet-rl

[![PyPI version](https://badge.fury.io/py/gobblet-rl.svg?branch=master)](https://badge.fury.io/py/gobblet-rl) 
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/elliottower/gobblet-rl/blob/main/LICENSE)

Interactive Multi-Agent Reinforcement Learning Environment for the [Gobblet](https://themindcafe.com.sg/wp-content/uploads/2018/07/Gobblet-Gobblers.pdf) board game  using [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) and [Pygame](https://github.com/pygame/pygame).

<p align="center">
  <img alt="Light" src="./gobblet.jpg" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="./gobblet_pygame.png" width="45%">
</p> 
 
## Installation

### Using pip (recommended)

```bash
pip install gobblet-rl
```

### Local

```bash
git clone hhttps://github.com/elliottower/gobblet-rl.git
cd gobblet-rl
pip install -e .
```

## Usage

### Setting up a basic environment

In a Python shell, run the following:

```python
import pettingzoo
from gobblet import gobblet_v1

env = gobblet_v1.env()
```

### Train a DQL agent with Tianshou

In the terminal, run the following:
```
python gobblet/example_tianshou.py
```

This will train a [DQN](https://tianshou.readthedocs.io/en/master/tutorials/dqn.html) model from Tianshou for 50 epochs, and then render the trained agent playing against a random agent in an example match. 


### Play an interactive game againt an agent

In the terminal, run the following:
```
python gobblet/example_interactive_pygame.py --render_mode="human" --agent_type="random_permissible"
```

### Command Line Arguments

#### Game Modes

The default game mode is human vs CPU, with the human playing as red and CPU as yellow. The ``--player`` argument allows you to play as yellow, with the CPU moving first.

The flag ``--no-cpu`` will launch a game with no CPU agents, taking interactive input for both agents. 

The flag ``--cpu-only`` will launch a game with two CPU agents, and takes no interactive input.


#### Render Modes

``--render_mode="human" `` will render the game board visually using pygame. Player 1 plays red and goes first, while player 2 plays yellow. 

<img src="https://raw.githubusercontent.com/elliottower/gobblet-rl/main/gobblet_pygame.png" width=30% height=30%>

When playing interactively, possible moves can be previewed by hovering the mouse over each square. To move a piece which is already placed, simply click on it.


``--render_mode="text"`` will render a 3x3 board only showing the topmost pieces (pieces which are covered by others, or 'gobbled', are hidden):
```
TURN: 2, AGENT: player_1, ACTION: 51, POSITION: 6, PIECE: 3
       |       |       
  -    |   -   |   -3  
_______|_______|_______
       |       |       
  -    |   -   |   +2  
_______|_______|_______
       |       |       
  -    |   -   |   -   
       |       |       
```

``--render_mode="text_full"`` will render three different 3x3 boards representing the small, medium and large pieces. This gives full information about pieces which are covered or 'gobbled' by other pieces. :
```
TURN: 3, AGENT: player_0, ACTION: 42, POSITION: 6, PIECE: 5
         SMALL                     MED                     LARGE           
       |       |                |       |                |       |       
  -    |   -   |   -       -    |   -   |   -       -    |   -   |   +5  
_______|_______|_______  _______|_______|_______  _______|_______|_______
       |       |                |       |                |       |       
  -    |   -   |   -2      -    |   -   |   -       -    |   -   |   -   
_______|_______|_______  _______|_______|_______  _______|_______|_______
       |       |                |       |                |       |       
  -    |   -   |   -       -    |   -   |   -       -    |   -   |   -6  
       |       |                |       |                |       |       
```


## Testing

Testing can be done via [pytest](http://doc.pytest.org/):

```bash
git clone https://github.com/elliottower/gobblet-rl.git
cd gobblet-rl
pytest
```
