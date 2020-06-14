import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king', 11]


class State():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.value = 0
        self.lastValue = 0

    def iterate(self, stateSpace):
        stick = self.calculateReward('stick', self.X, self.Y) + self.lastValue
        hit = 0
        for state in stateSpace:
            if self.isPossibleState(state):
                transProb = self.calcTransProb(state)
                hit += state.lastValue*transProb
        hit += self.calculateReward('hit', self.X, self.Y)
        self.lastValue = self.value
        self.value = max(hit, stick)
    def calculateReward(self, a, X, Y):
        if a == 'stick':
            dealer = Dealer(X, Y)
            return dealer.play()
        else:
            return 0

    def isPossibleState(self, state):
        stateX = state.X
        if np.abs(stateX - self.X) <= 11 and np.abs(stateX - self.X) > 1:
            return True
        else:
            return False

        pass

    def calcTransProb(self, state):
        stateX = state.X
        if np.abs(stateX - self.X) < 10 or np.abs(stateX - self.X) == 11:
            return 1/13
        elif np.abs(stateX - self.X) == 10:
            return 4/13
        else:
            return 0


class Dealer():
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X
        self.play()

    def play(self):
        nextCard = np.random.randint(2, 11+1)
        self.Y = self.Y + nextCard
        if self.Y > 21:
            return 1
        elif self.Y < 17:
            return self.play()
        else:
            if self.Y > self.X:
                return -1
            elif self.X > self.Y:
                return 1
            else:
                return 0


def printValueSpace(stateSpace):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    Y, X, value = [], [], []
    for state in stateSpace:
        Y.append(state.Y)
        X.append(state.X)
        value.append(state.value)
    ax.scatter3D(X, Y, value, c=value)
    plt.show()


if __name__ == '__main__':
    Y = [x+2 for x in range(10)]
    X = [x+4 for x in range(18)]
    stateSpace = []
    for x in X:
        for y in Y:
            stateSpace.append(State(x, y))
    printValueSpace(stateSpace)
    for i in range(20):
        for state in stateSpace:
            state.iterate(stateSpace)
    printValueSpace(stateSpace)

