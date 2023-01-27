# gobblet-rl

Multi-Agent Reinforcement Learning Environment for the [Gobblet](https://themindcafe.com.sg/wp-content/uploads/2018/07/Gobblet-Gobblers.pdf) board game board game using [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo).

![Gobblet game](gobblet.jpg?raw=true "Gobblet game")

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
from gobblet gobblet_v0

-rl
env = gobblet_v0.env()
```

### Train a basic agent with Tianshou

In the terminal, run the following:
```
python gobblet/example_tianshou.py
```

This will train a [DQN](https://tianshou.readthedocs.io/en/master/tutorials/dqn.html) model from Tianshou for 50 epochs, and then render the trained agent playing against a random agent in an example match. 


### Playing a game with a random agent

In the terminal, run the following:
```
python gobblet/example_random.py --render_mode="human" --agent_type="random"
```

#### Command-line arguments

``--render_mode="human"`` will render a 3x3 board only showing the topmost pieces (pieces which are covered by others, or 'gobbled', are hidden):
```
AGENT: player_1, ACTION: 50, POSITION: 5, PIECE: 6
       |       |       
  -    |   -   |   -   
_______|_______|_______
       |       |       
  -    |   -   |   -   
_______|_______|_______
       |       |       
  +5   |   -6  |   -   
       |       |       
```

``--render_mode="human_full"`` will render three different 3x3 boards representing the small, medium and large pieces. This gives full information about pieces which are covered or 'gobbled' by other pieces. :
```
AGENT: player_0, ACTION: 42, POSITION: 6, PIECE: 5
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
