import datetime
from random import choice
from numpy.random import choice as npr

boardSize = 9
hexBoard = [[0]*boardSize for i in range(boardSize)]


class Board (object):
    def __init__(self):
        self
    def start (self):
        pass
    
    def getCurrentPlayer(self, state):
        pass
    
    def nextState(self, state, play):
        pass
    
    def getLegalPlays(self, stateHistory):
        pass
    
    def winner(self, stateHistory):
        def shearchForWin():
            """Check if player has won."""
            wasHere = [[False]*boardSize for i in range(boardSize)]
        
            def resetPath():
                nonlocal wasHere
                wasHere = [[False]*boardWidth for i in range(boardHeight)]
        
            def recursiveCheck(x, y, depth):
                """The recursive check."""
                if (player == 1 and player == board[y][x]) and (y == boardHeight-1):
                    path[y][x] = depth
                    return True
                elif (player == 2 and player == board[y][x]) and (x == boardWidth-1):
                    path[y][x] = depth
                    return True
                if (board[y][x] != player) or (wasHere[y][x]):
                    return False
                wasHere[y][x] = True
        
                if(x != boardWidth - 1):  # right
                    if(recursiveCheck(x+1, y, depth+1)):
                        path[y][x] = depth
                        return True
                if(y != boardHeight - 1):  # down
                    if(recursiveCheck(x, y+1, depth+1)):
                        path[y][x] = depth
                        return True
                if(x != 0):  # left
                    if(recursiveCheck(x-1, y, depth+1)):
                        path[y][x] = depth
                        return True
                if(y != 0):  # up
                    if(recursiveCheck(x, y-1, depth+1)):
                        path[y][x] = depth
                        return True
                if((y != 0) and (x != boardWidth - 1)):  # up-right
                    if(recursiveCheck(x+1, y-1, depth+1)):
                        path[y][x] = depth
                        return True
                if((y != boardHeight - 1) and (x != 0)):  # down-left
                    if(recursiveCheck(x-1, y+1, depth+1)):
                        path[y][x] = depth
                        return True
                return False
            player = 1
            for x in range(boardWidth):
                resetPath()
                if (recursiveCheck(x, 0, 1)):
                    # BlueWin
                    return path
            player = 2
            for y in range(boardHeight):
                resetPath()
                if (recursiveCheck(0, y, 1)):
                    # RedWin
                    return path
            return None
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
        self.maxPlay = 0
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
        
        return move
    
    def runSimulation(self):
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
                move, state = choice(legal)
                
            statesCopy.append(state)
            
            
            if expand and (player, state) not in self.plays:
                expand = False
                self.plays[(player, state)] = 0
                self.wins[(player, state)] = 0
            
            visitedStates.add((player, state))
            
            winner = self.board.winner()
            if winner:
                break
            
        for player, state in visitedStates:
            if (player, state) not in self.plays:
                continue
            self.plays[(player, state)] += 1
            if player == winner:
                self.wins[(player, state)] += 1


    