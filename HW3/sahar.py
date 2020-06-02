import numpy as np

class State():
    def __init__(self, player_sum, dealer_sum):
        self.player_sum = player_sum
        self.dealer_sum = dealer_sum
    
    def 
    

class Game():
    def __init__(self):
        pass

    def step(self, action):
        if action == 'hit':
            card = self.get_card()
            self.player_sum += card
            if self.player_sum>21:
                return -1
            elif  self.player_sum == 21:
                return 1
            else:
                return 0
        else:
            return 0
    

    