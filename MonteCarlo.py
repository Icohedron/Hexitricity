import datetime
from random import choice
import numpy as np

from _ast import Nonlocal

boardSize = 9
hexBoard = [[0]*(boardSize**2) for i in range(boardSize)]


class Board (object):
    def _init_ (self):	                    #Init 
        self.hexBoard = [[0]*(boardSize**2) for i in range(boardSize)]
    
    def setBoard(self, board):              #Done
        self.hexBoard = board
    
    def startState (self): 	                #Done
        return np.reshape(self.hexBoard,(1, len(self.hexBoard)))
    
    def getCurrentPlayer(self, state): 	    #Done
        plr1 = 0
        plr2 = 0
        
        for i in state:
            if i == 0:
                pass
            elif i == 1:
                plr1 += 1
            else:
                plr2 += 1
        if plr1 > plr2:
            return 2
        else:
            return 1
            
    def nextState(self, state, play): 	    #Done
        plr = getCurrentPlayer(state)
        state[play] = plr
        return state
    
    def getLegalPlays(self, stateHistory): 	#Done
        plays = []
        for i in range(len(stateHistory[-1])):
            if stateHistory[-1][i] == 0:
                plays.append(i)
        return plays
    
    def winner(self, stateHistory):         #Done
        
        board = np.reshape(stateHistory[-1], (boardSize, boardSize)).tolist()
        
        def shearchForWin():
            """Check if player has won."""
            wasHere = [[False]*boardSize for i in range(boardSize)]
            
            def recursiveCheck(x, y, depth):
                """The recursive check."""
                if (player == 1 and player == board[y][x]) and (y == boardSize-1):
                    return True
                elif (player == 2 and player == board[y][x]) and (x == boardSize-1):
                    return True
                if (board[y][x] != player) or (wasHere[y][x]):
                    return False
                wasHere[y][x] = True
        
                if(x != boardSize - 1):  # right
                    if(recursiveCheck(x+1, y, depth+1)):
                        return True
                if(y != boardSize - 1):  # down
                    if(recursiveCheck(x, y+1, depth+1)):
                        return True
                if(x != 0):  # left
                    if(recursiveCheck(x-1, y, depth+1)):
                        return True
                if(y != 0):  # up
                    if(recursiveCheck(x, y-1, depth+1)):
                        return True
                if((y != 0) and (x != boardSize - 1)):  # up-right
                    if(recursiveCheck(x+1, y-1, depth+1)):
                        return True
                if((y != boardSize - 1) and (x != 0)):  # down-left
                    if(recursiveCheck(x-1, y+1, depth+1)):
                        return True
                return False
            player = 1
            for x in range(boardSize):
                wasHere = [[False]*boardSize for i in range(boardSize)]
                if (recursiveCheck(x, 0, 1)):
                    # BlueWin
                    return 1
            player = 2
            for y in range(boardSize):
                wasHere = [[False]*boardSize for i in range(boardSize)]
                if (recursiveCheck(0, y, 1)):
                    # RedWin
                    return 2
            return 0
        return shearchForWin()
    
class monteCarlo(object):
    def __init__(self, board, **kwargs):
        
        self.board = board
        self.states = []
        
        seconds = kwargs.get('Time', 60)
        self.calculationTime = dateTime.timedelta(seconds = seconds)
        
        self.wins  = {}
        self.plays = {}
        
        self.c = kwargs.get('c', 1.4)
    
    def update (self, state):
        pass
    
    def getPlay (self):
        self.maxDepth = 0
        state = self.states[-1]
        player = self.board.currentPlayer(state)
        legal = self.board.getLegalPlays(self.states[:])
        
        if not legal:
            return
        elif len(legal) == 1:
            return legal[0]
        
        games = 0
        
        begin = dateTime.datetime.utcnow()
        while dateTime.dateTime.utcnow() - begin < self.calculationTime:
            self.runSimulation()
            games += 1
        
        moveStates = [(p, self.board.nextState(state, p)) for p in legal]
        
        print (games, " games in ", datetime.datetime.utcnow() - begin, " seconds.")
        
        percentWins, move = max(
                               (self.wins.get((player, s), 0) /
                                self.plays.get((player, s), 1), p)
                                for p, s in moveStates
                               )
        
        print("Max depth reached: ", self.maxDepth)
        return move
    
    def runSimulation(self):
        plays, wins = self.plays, self.wins
        
        visitedStates = set()
        statesCopy = self.states[:]
        state = statesCopy[-1]
        player = self.board.getCurrentPlayer(self, state)
        
        expand = True
        
        for t in range(self.maxMoves  + 1):
            legal = self.board.getLegalPlays(statesCopy)
            moveStates = [(p, self.board.nextState(state, p)) for p in legal]
            
            if all(plays.get((player, s)) for p, s in moveStates):
                logTotal = log(sum(plays[player, s] for p, s in moveStates))
                value, move, state = max(
                                        ((wins[(player, s)] / plays[(player, s)]) +
                                        self.c * sqrt(logTotal / plays[(player, s)]), p, s)
                                        for p, s in moveStates)
                
            else:
                move, state = choice(moveStates)
                
            statesCopy.append(state)
            
            
            if expand and (player, state) not in self.plays:
                expand = False
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0
                if t > self.maxDepth:
                    self.maxDepth = t
            
            visitedStates.add((player, state))
            
            winner = self.board.winner(statesCopy)
            if winner:
                break
            
        for player, state in visitedStates:
            if (player, state) not in self.plays:
                continue
            self.plays[(player, state)] += 1
            if player == winner:
                self.wins[(player, state)] += 1


