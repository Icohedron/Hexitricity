import datetime
import dill
import os.path
import sys
from random import choice
import numpy as np
import gc
import network_player as netPlay
import gym

from _ast import Nonlocal

boardSize = 9
inBoard = [[0]*(boardSize) for i in range(boardSize)]

path = "Saves/Trees/"
boardType = str(boardSize)

# network = netPlay.NetworkPlayer()

# Todo more comments
with netPlay.NetworkPlayer() as network:
    class Board (object):
        """Board class for MCTS."""
    
        def _init_ (self):                        #Change as needed
            self.hexBoard = [[0]*(boardSize**2) for i in range(boardSize)]
    
        def setBoard(self, newBrd):             #Done
            self.hexBoard = newBrd
    
        def startState (self):                     #Done
            return np.array(self.hexBoard).flatten().tolist()
    
        def getCurrentPlayer(self, state):         #Done
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
    
        def nextState(self, state, play):         #Done
            out = list(state)
            out[play] = self.getCurrentPlayer(out)
            return tuple(out)
    
        def getLegalPlays(self, state):         #Done
            plays = []
            for i in range(len(state)):
                if state[i] == 0:
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
        
        
        def getTensorState(self, state):
            blank   = [ [ 0 for i in range(9) ] for j in range(9) ]
            player1 = [ [ 0 for i in range(9) ] for j in range(9) ]
            player2 = [ [ 0 for i in range(9) ] for j in range(9) ]
            
            state2d = np.reshape(state, [9,9]).tolist()
            for y in range(9):
                for x in range(9):
                    if state2d[y][x] == 0:
                        blank[y][x] = 1
                    elif state2d[y][x] == 1:
                        player1[y][x] = 1
                        
                    else:
                        player2[y][x] = 1
                        
            # print(np.matrix(player1),"\n\n\n", np.matrix(player2),"\n\n\n", np.matrix(blank))
            return [player1, player2, blank]
        
        
        def miaState(self, state):
            output = [0]*81
            for a in range(2):
                for y in range(len(state[0])):
                    for x in range(len(state[0][0])):
                        if state[a][y][x] == 1:
                            if a == 0:
                                output[x + y * 9] = 1
                            elif a == 1:
                                output[x + y * 9] = 2
            return output
                    
    class monteCarlo(object):
    
        def __init__(self, board, **kwargs):
            """Initialize"""
            self.board = board
            self.states = []
    
            secondsCalc = kwargs.get('Time', 1)
            secondsNote = kwargs.get('Time', 600)
            self.calculationTime = datetime.timedelta(seconds = secondsCalc)
            self.notificationTime = datetime.timedelta(seconds = secondsNote)
    
            self.loadDicts(".none")
    
            self.c = kwargs.get('c', 1.4)
    
        def loadDicts(self, ext):
            """Load the node dictionaries."""
            global path, boardType
            gc.disable()
            if os.path.isfile(path + boardType + "-Plays"  + ext):
                print("Loading saved data")
                try:
                    self.plays = dill.load(open(path + boardType + "-Plays" + ext,"rb"))
                    self.wins = dill.load(open(path + boardType + "-Wins" + ext,"rb"))
                    print("Loaded")
                except:
                    print("Unable to load .dp, trying to load .bkp")
                    try:
                        self.plays = dill.load(open(path + boardType + "-Plays" + ".bkp","rb"))
                        self.wins = dill.load(open(path + boardType + "-Wins" + ".bkp","rb"))
                        print("Loaded")
                    except:
                        print("Failed to load the data. Ending program.")
                        gc.enable()
                        sys.exit()
                        quit()
            else:
                print("Creating new data")
                self.wins  = {}
                self.plays = {}
            gc.enable()
    
        def saveDicts(self, ext):
            """Save the node dictionaries."""
            global path, boardType
            gc.disable()
            print("Saving")
    
            dill.dump(self.plays, open(path + boardType + "-Plays" + ext,"wb"))
            dill.dump(self.wins, open(path + boardType + "-Wins" + ext,"wb"))
    
            print("Saved")
            gc.enable()
            print("Wins Recorded: ", len(self.wins))
            print("Plays Recorded: ", len(self.plays))
            print()
    
        def getPlay (self, inputState):
            self.states.append(inputState)
            self.maxDepth = 0
            state = self.states[-1]
            player = self.board.getCurrentPlayer(state)
            legal = self.board.getLegalPlays(state)
            # print(legal, state, self.states)
            if not legal:
                return
            elif len(legal) == 1:
                return legal[0]
    
            # print("Has legal")
    
            games = 0
    
            last = datetime.datetime.utcnow()
            begin = datetime.datetime.utcnow()
            
            # print("starting session")
            while datetime.datetime.utcnow() - begin < self.calculationTime:
    
                self.runSimulation()
                games += 1
    
            moveStates = [(p, self.board.nextState(state, p)) for p in legal]
           # print("Session complete")
            print (games, " games in ", datetime.datetime.utcnow() - begin, " seconds.")
    
            percentWins, move = max(
                                   (self.wins.get((player, s), 0) /
                                    self.plays.get((player, s), 1), p)
                                    for p, s in moveStates
                                   )
    
            print("Max depth reached: ", self.maxDepth)
    
            # self.saveDicts(".dp")
            return move
        
            
    
        def runSimulation(self):
            plays, wins = self.plays, self.wins
    
            visitedStates = set()
            statesCopy = self.states[:]
            state = statesCopy[-1]
            player = self.board.getCurrentPlayer(state)
    
            expand = True
    
            self.maxMoves = len(self.board.getLegalPlays(state))
            for t in range(self.maxMoves  + 1):
    
                legal = self.board.getLegalPlays(statesCopy[-1])
    
    
                moveStates = [(p, self.board.nextState(state, p)) for p in legal] # o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o
    
                if all(plays.get((player, s)) for p, s in moveStates):
                    logTotal = np.log(sum(plays[player, s] for p, s in moveStates))
                    value, move, state = max(
                                            ((wins[(player, s)] / plays[(player, s)]) +
                                            self.c * np.sqrt(logTotal / plays[(player, s)]), p, s)
                                            for p, s in moveStates
                                            )
    
                else:
                    # Todo: get the moves from NN chooses
                    global network
                    if t < 0:
                        move = network.get_top_action(self.board.getTensorState(self.gymBoard))
                        state = self.board.nextState(state, move)
                    else:
                        move, state = choice(moveStates)
    
                statesCopy.append(state)
    
                if t > self.maxDepth:
                        self.maxDepth = t
    
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
                    
        def setGymBoard(self, state):
            self.gymBoard = state
    
    Brd = Board()
    Brd.setBoard(inBoard)
    
    
    
    
    
    
    monty = monteCarlo(Brd)
    # state = Brd.startState()
    # print(monty.getPlay(state))
    
    
    def evaluate(player, board):
        environment = gym.make('Hex9x9-v0')
        
        episode_rewards = []
        environment.seed(0)
        for episode in range(50):
            state = environment.reset()
            terminal = False
            ep_t = 0
            
            while not terminal:
                
                # print("got here atleast")
                # print(np.matrix(state)deep
                # print(state)
                
                # environment.render()
                    
                player.setGymBoard(state)
                action = player.getPlay(board.miaState(state))
                state, reward, terminal, info = environment.step(action)
    
                ep_t += 1
                if ep_t == 40:
                    terminal = True
    
                episode_rewards.append(reward)
            
            print('Finished episode {}/25'.format(episode+1))
    
        print('Games won (out of 25): {}'.format(len(episode_rewards) - np.count_nonzero(np.array(episode_rewards) - 1.0)))
        print('Games lost (out of 25): {}'.format(len(episode_rewards) - np.count_nonzero(np.array(episode_rewards) + 1.0)))
    
    
    evaluate(monty, Brd)
    
    monty.saveDicts(".done")
    
