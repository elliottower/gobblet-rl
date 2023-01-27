# gobblet-rl

This is an implementation of the Gobblet board game as a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) game.
The rules have been sourced from [here](https://themindcafe.com.sg/wp-content/uploads/2018/07/Gobblet-Gobblers.pdf).
An additional rule where games exceeding 44 moves end in a draw has been implemented, since the
original rules do not have explicit draw conditions.

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
from gobblet-rl import gobblet_v0
env = gobblet_v0.env()
```

## Testing

Testing can be done via [pytest](http://doc.pytest.org/):

```bash
git clone https://github.com/elliottower/gobblet-rl.git
cd gobblet-rl
pytest
```
