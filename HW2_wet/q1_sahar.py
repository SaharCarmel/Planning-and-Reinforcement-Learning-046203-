import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'jack', 'queen', 'king', 11]


class State():
    def __init__(self, X, Y, gamma=0.7):
        self.X = X
        self.gamma = gamma
        self.Y = Y
        self.value = 0
        self.tempVal = 0
        self.lastValue = 0.1
        self.stick = 0
        self.hit = 0
        self.reward = self.calculateReward()

    def iterate(self, stateSpace):
        self.stick = self.gamma * self.lastValue + self.reward
        self.hit = 0
        if self.X != 21:
            for state in stateSpace:
                if self.isPossibleState(state):
                    transProb = self.calcTransProb(state)
                    self.hit += (self.gamma*(state.lastValue+state.reward))*transProb
        if self.X > 21:
            self.hit += -1
        self.value = max(self.hit, self.stick)
        self.tempVal = self.value

    def updateValue(self):
        self.lastValue = self.tempVal

    def calculateReward(self):
        _sum = 0 
        times = 1000
        for i in range(times):
            dealer = Dealer(self.X, self.Y)
            _sum += dealer.play()
        
        return _sum / times

    def isPossibleState(self, state):
        stateX = state.X
        if state.Y != self.Y:
            return False
        elif stateX <= self.X or self.X >=21:
            return False
        elif stateX - self.X <= 11 and stateX - self.X > 1:
            return True
        else:
            return False

    def calcTransProb(self, state):
        stateX = state.X
        if np.abs(stateX - self.X) < 10 or np.abs(stateX - self.X) == 11:
            return (1/13)
        elif np.abs(stateX - self.X) == 10:
            return (4/13)
        else:
            return 0


class Dealer():
    def __init__(self, X, Y):
        self.Y = Y
        self.X = X

    def play(self):
        nextCard = np.random.randint(2, 11+1)
        self.Y = self.Y + nextCard
        if self.X > 21:
            return -1
        elif self.X == 21:
            return 1
        elif self.Y > 21:
            return 1
        elif self.Y == 21:
            return -1
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
        if state.X <= 21:
            Y.append(state.Y)
            X.append(state.X)
            value.append(state.value)
    ax.plot_trisurf(X, Y, value, cmap=plt.cm.Spectral)
    # ax.scatter3D(X, Y, value, c=value)
    ax.set_title('The optimal value function V* as a function of (X,Y)')
    ax.set_xlabel('Player')
    ax.set_ylabel('Dealer')
    ax.view_init(37,-133)
    plt.show()


if __name__ == '__main__':
    Y = [x+2 for x in range(10)]
    X = [x+4 for x in range(27)]
    stateSpace = []
    for x in X:
        for y in Y:
            stateSpace.append(State(x, y))
    # printValueSpace(stateSpace)
    test = []
    for i in range(30):
        for state in stateSpace:
            state.iterate(stateSpace)
        for state in stateSpace:
            state.updateValue()
    printValueSpace(stateSpace)
    minVal = []
    for y in Y:
        tmpBool = True
        _tmpX = 22
        for state in stateSpace:
            if tmpBool:
                if state.Y == y and state.stick >= state.hit:
                    if state.X < _tmpX:
                        minVal.append(state.X)
                        tmpBool = False
                        _tmpX = state.X
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(111)
    ax.scatter(Y, minVal)
    ax.set_ylabel('Player sum (X)')
    ax.set_xlabel('Dealers showing (Y)')
    plt.ylim([4, 21])
    plt.show()
x = 2 