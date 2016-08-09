"""http://www.colourlovers.com/palette/4286894/otthonszerelem."""
import tkinter as tk
import atexit
import math
import time
import random
import threading
from _ast import Nonlocal


# Vars


running = True

def close():
    global running
    running = False

atexit.register(close)

hexRadius = int(input('Hexagon Radius : '))
boardWidth = int(input('Board Width    : '))
boardHeight = int(input('Board Height   : '))

playerOneIsBot = True if "y" in input('Player 1 Bot (y/n): ').lower() else False
playerTwoIsBot = True if "y" in input('Player 2 Bot (y/n): ').lower() else False

print(playerOneIsBot, playerTwoIsBot)

hitBoxRadius = hexRadius - (hexRadius/6)
pawnRadius = hexRadius - (hexRadius/3)

windowWidth = (boardWidth) * (math.cos(math.radians(30))*hexRadius*2) + \
              ((math.cos(math.radians(30))*hexRadius)*boardHeight) + \
              (math.cos(math.radians(30))*hexRadius)
windowHeight = (boardHeight+1) * (math.sin(math.radians(30))*hexRadius*3)

turn = 1
board = [[0]*boardWidth for i in range(boardHeight)]

hexagonPoints = []
hitBoxPoints = []

# Constuct


master = tk.Tk()
canvas = tk.Canvas(master,
                   bg="#F7FFF5",
                   height=windowHeight,
                   width=windowWidth)
canvas.pack()


# Funcions


class BotRandy(threading.Thread, object):
    move = [-1, -1]
    team = -1
    def __init__(self, player):
        threading.Thread.__init__(self)
        self.team = player

    def run(self):
        while running:
            y = random.randint(0,boardHeight-1)
            x = random.randint(0,boardWidth-1)
            if board[y][x] == 0:
                self.move = [x, y]
            while self.move != [-1, -1] and running:
                time.sleep(0.0025)
                
    def getMove(self):
        copy = (self.move)
        self.move = [-1, -1]
        return copy
    
    def isReady(self):
        return False if self.move == [-1][-1] else True
    
    
def clickCallback(event):
    """hai."""
    global turn
    smallest = hitBoxRadius
    for y in range(len(board)):
        for x in range(len(board[0])):
            newX = (x+1) * (math.cos(math.radians(30))*hexRadius*2) + \
                   ((math.cos(math.radians(30))*hexRadius)*y)
            newY = (y+1) * (math.sin(math.radians(30))*hexRadius*3)
            if math.sqrt(((newX-event.x)**2) + ((newY-event.y)**2)) < smallest:
                if board[y][x] == 0:
                    placePawn([x,y])
                break
    testForWinner()


def gameEnd(seconds):
    canvas.update()
    time.sleep(seconds)
    clearBoard()
    
    
def hexGen(radius):
    """Generator for the shape of a hexagon."""
    x = 0
    y = 0
    for angle in range(0, 360, 60):
        x = math.cos(math.radians(angle+30)) * radius
        y = math.sin(math.radians(angle+30)) * radius
        yield x, y


def getHexagonPointsAt(x, y, basePoints):
    """Create the points of a hexagon."""
    thisHexagon = list(basePoints)
    for i in range(0, 12, 2):
        thisHexagon[i] += x
        thisHexagon[i+1] += y
    return thisHexagon


def createHexagonAt(x, y, basePoints, colour, actfill, outln):
    """Create a hexagon."""
    canvas.create_polygon(getHexagonPointsAt(x, y, list(basePoints)),
                          fill=colour,
                          width=hexRadius/4,
                          outline=outln,
                          activefill=actfill)


def createCircleAt(x, y, r, colour):
    """Create a circle."""
    canvas.create_oval(x-r, y-r, x+r, y+r, fill=colour, width=0, tags='pawn')


def drawBoard():
    """draw a game board array."""
    line = [[], [], [], []]

    for y in range(len(board)):
        for x in range(len(board[0])):
            newX = (x+1) * (math.cos(math.radians(30))*hexRadius*2) + \
                   ((math.cos(math.radians(30))*hexRadius)*y)
            newY = (y+1) * (math.sin(math.radians(30))*hexRadius*3)
            createHexagonAt(newX, newY,
                            hexagonPoints,
                            "#F7FFF5", "#DAECE8", "#BCD9DB")

            if x == 0:
                lines = list(getHexagonPointsAt(newX, newY,
                             list(hexagonPoints)))
                line[0].append(lines[6])
                line[0].append(lines[7])
                line[0].append(lines[4])
                line[0].append(lines[5])
                maxRight = (len(board[0])) * \
                           (math.cos(math.radians(30))*hexRadius*2)
                line[1].append(lines[6]+maxRight)
                line[1].append(lines[7])
                line[1].append(lines[4]+maxRight)
                line[1].append(lines[5])
            if y == 0:
                lines = list(getHexagonPointsAt(newX, newY,
                             list(hexagonPoints)))
                line[2].append(lines[6])
                line[2].append(lines[7])
                line[2].append(lines[8])
                line[2].append(lines[9])
                maxRight = ((math.cos(math.radians(30))*hexRadius)*len(board))
                maxBottom = (len(board)) * (math.sin(math.radians(30)) *
                                            hexRadius*3)
                line[3].append(lines[6]+maxRight)
                line[3].append(lines[7]+maxBottom)
                line[3].append(lines[8]+maxRight)
                line[3].append(lines[9]+maxBottom)

    canvas.create_line(line[0],
                       width=hexRadius/4,
                       fill="#FF3D63")
    canvas.create_line(line[1],
                       width=hexRadius/4,
                       fill="#FF3D63")
    canvas.create_line(line[2],
                       width=hexRadius/4,
                       fill="#2B626B")
    canvas.create_line(line[3],
                       width=hexRadius/4,
                       fill="#2B626B")


def clearBoard():
    """clear the board."""
    canvas.delete('pawn')
    global board, turn
    board = [[0]*boardWidth for i in range(boardHeight)]
    turn = 1


def shearchForWin():
    """Check if player has won."""
    path = [[-1]*boardWidth for i in range(boardHeight)]
    wasHere = [[False]*boardWidth for i in range(boardHeight)]

    def resetPath():
        nonlocal path, wasHere
        path = [[-1]*boardWidth for i in range(boardHeight)]
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


def testForWinner():
    outCome = shearchForWin()
    if outCome is not None:
        line = []
        for y in range(len(outCome)):
            for x in range(len(outCome[0])):
                newX = (x+1) * (math.cos(math.radians(30))*hexRadius*2) + \
                       ((math.cos(math.radians(30))*hexRadius)*y)
                newY = (y+1) * (math.sin(math.radians(30))*hexRadius*3)
                if outCome[y][x] >= 1:
                    line.append([outCome[y][x], newX, newY])
        sort(line)
        winLine = []
        for i in line:
            winLine.append(i[1])
            winLine.append(i[2])
        if turn == 1:
            canvas.create_line(winLine, width=hitBoxRadius,
                               fill="#FF3D63", tags='pawn')
            gameEnd(0.5)
        if turn == 2:
            canvas.create_line(winLine, width=hitBoxRadius,
                               fill="#2B626B", tags='pawn')
            gameEnd(0.5)
            

def sort(alist):
    """Start quick Sort."""
    def quickSort(alist, first, last):
        """Main quick sort."""
        if first < last:
    
            splitpoint = partition(alist, first, last)
    
            quickSort(alist, first, splitpoint-1)
            quickSort(alist, splitpoint + 1, last)
    
    
    def partition(alist, first, last):
        """Partition off."""
        pivotvalue = alist[first][0]
    
        left = first + 1
        right = last
    
        done = False
        while not done:
    
            while left <= right and alist[left][0] <= pivotvalue:
                left = left + 1
    
            while alist[right][0] >= pivotvalue and right >= left:
                right = right - 1
    
            if right < left:
                done = True
            else:
                temp = alist[left]
                alist[left] = alist[right]
                alist[right] = temp
    
        temp = alist[first]
        alist[first] = alist[right]
        alist[right] = temp
    
        return right
    quickSort(alist, 0, len(alist)-1)


def initHexagon():
    """Initialize the points of a hexagon."""
    for i in hexGen(hexRadius):
        hexagonPoints.append(i[0])
        hexagonPoints.append(i[1])
    for i in hexGen(hitBoxRadius):
        hitBoxPoints.append(i[0])
        hitBoxPoints.append(i[1])


def placePawn(input):
    """draws a pawn onto the board."""
    global turn, board
    board[input[1]][input[0]] = turn
    newX = (input[0]+1) * (math.cos(math.radians(30))*hexRadius*2) + (
                          (math.cos(math.radians(30))*hexRadius)*input[1])
    newY = (input[1]+1) * (math.sin(math.radians(30))*hexRadius*3)
    if turn == 1:
        createCircleAt(newX, newY, pawnRadius, "#2B626B")
        turn = 2
    else:
        createCircleAt(newX, newY, pawnRadius, "#FF3D63")
        turn = 1   


def startUp():
    BotterA = None
    BotterB = None
    if playerOneIsBot:
        BotterA = BotRandy(1)
        BotterA.start()
    if playerTwoIsBot:
        BotterB = BotRandy(2)
        BotterB.start()
        
    def run():
        """RunLoop."""
        
        if turn == 1:
            nonlocal BotterA
            if playerOneIsBot and BotterA.isReady():
                move = BotterA.getMove()
                placePawn(move)
                testForWinner()
        else:
            nonlocal BotterB
            if playerTwoIsBot and BotterB.isReady():
                move = BotterB.getMove()
                placePawn(move)
                testForWinner()
        canvas.update()  
        canvas.after(0, run)
        
    initHexagon()
    drawBoard()
    canvas.bind("<Button-1>", clickCallback)
    
    def on_closing():
        global running
        running = False
        print("rekt")
        master.destroy()

    master.protocol("WM_DELETE_WINDOW", on_closing)

    canvas.after(10, run)
    canvas.mainloop()
    
startUp()