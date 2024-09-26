import random
from enum import Enum

class Color(Enum):
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"


class Player:
    def __init__(self, color, is_bot=True):
        self.color = color
        self.is_bot = is_bot

    def decide(self, game, playable_actions):
        raise NotImplementedError


class randomPlayer(Player):
    def decide(self, game, playable_actions):
        return playable_actions[random.randint(0, len(playable_actions) - 1)]


class humanPlayer(Player):
    def decide(self, game, playable_actions):
        for i, action in enumerate(playable_actions):
            print(f"{i}: {action.action_type} {action.value}")
        i = None
        while i is None or (i < 0 or i >= len(playable_actions)):
            print("Please enter a valid index:")
            try:
                x = input(">>> ")
                i = int(x)
            except ValueError:
                pass

        return playable_actions[i]
