# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        #  Calculate distance from each of the ghost. Find the closest ghost.
        distToGhost = []
        for ghosts in newGhostStates:
            distToGhost.append(util.manhattanDistance(ghosts.getPosition(), newPos))
        minDistToGhost = min(distToGhost)
        # If the action takes pacman too close to the ghost, then return high neg value to indicate it is not a good
        # action.
        if minDistToGhost < 3.0:
            return -9999

        # Find the closest food dot distance. If the action takes pacman closer, we have to return a higher value.
        # So take 1/dist to food.
        distToFood = 9999
        for food in newFood.asList():
            dist = util.manhattanDistance(food, newPos)
            if dist < distToFood:
                distToFood = dist

        return successorGameState.getScore() + 1.0 / distToFood


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        action, cost = self.minimax(gameState, 0, 0)
        return action

    """
    This function is used to dispatch control to appropriate function. If agent is pacman then call max else call min.
    It also updates depth every time all the agents are done calculating.
    On reaching the leaf it return output of evaluation function.
    """
    def minimax(self, gameState, depth, agentIndex):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth = depth + 1
        if gameState.isWin() or gameState.isLose() or (depth == self.depth):
            return "leaf", self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, depth, 0)
        else:
            return self.minVal(gameState, depth, agentIndex)

    """
    This function handles the max agents. It looks at all the legal actions calculates the utility for each of the 
    alternative. returns action with max utility.
    """
    def maxVal(self, gameState, depth, agentIndex):
        actionCost = ("init", -float("inf"))
        maxCost = -float("inf")
        for action in gameState.getLegalActions(agentIndex):
            cost = self.minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if cost[1] > maxCost and cost[0] != "init":
                maxCost = cost[1]
                actionCost = (action, cost[1])

        return actionCost

    """
    This function handles the min agents. It looks at all the legal actions calculates the utility for each of the 
    alternative. returns action with min utility.
    """
    def minVal(self, gameState, depth, agentIndex):
        actionCost = ("init", float("inf"))
        minCost = 99999
        for action in gameState.getLegalActions(agentIndex):
            cost = self.minimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if minCost > cost[1] and cost[0] != "init":
                minCost = cost[1]
                actionCost = (action, cost[1])

        return actionCost


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action, cost = self.alphaBeta(gameState, -float("inf"), float("inf"), 0, 0)
        return action

    """
    This function is used to dispatch control to appropriate function. If agent is pacman then call max else call min.
    It also updates depth every time all the agents are done calculating.
    On reaching the leaf it return output of evaluation function.
    """
    def alphaBeta(self, gameState, alpha, beta, depth, agentIndex):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth = depth + 1
        if gameState.isWin() or gameState.isLose() or (depth == self.depth):
            return "leaf", self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, alpha, beta, depth, 0)
        else:
            return self.minVal(gameState, alpha, beta, depth, agentIndex)
    """
    This function is similar to maxval of minimax agent. Only difference is it doesnt look at all the nodes to take the 
    decision. When value of the node becomes > beta the rest of the nodes in the route is not considered.
    """
    def maxVal(self, gameState, alpha, beta, depth, agentIndex):
        value = ("init", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            cost = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth, agentIndex + 1)
            if cost[1] > value[1]:
                value = action, cost[1]

            if value[1] > beta:
                return value
            alpha = max(alpha, value[1])
        return value

    """
    This function is similar to minval of minimax agent. Only difference is it doesnt look at all the nodes to take the 
    decision. When value of the node becomes < alpha the rest of the nodes in the route is not considered.
    """
    def minVal(self, gameState, alpha, beta, depth, agentIndex):
        value = ("init", float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            # print(action)
            cost = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth, agentIndex + 1)
            if cost[1] < value[1]:
                value = action, cost[1]

            if value[1] < alpha:
                return value
            beta = min(beta, value[1])
        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, cost = self.expectimax(gameState, 0, 0)
        return action

    """
    Function same as to minimax.
    """
    def expectimax(self, gameState, depth, agentIndex):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth = depth + 1
        if gameState.isWin() or gameState.isLose() or (depth == self.depth):
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxVal(gameState, depth, 0)
        else:
            return self.expectVal(gameState, depth, agentIndex)

    """
    Same as maxval of minimax.
    """
    def maxVal(self, gameState, depth, agentIndex):
        actionCost = ("init", -float("inf"))
        maxCost = -float("inf")
        # print(len(gameState.getLegalActions(agentIndex)))
        for action in gameState.getLegalActions(agentIndex):
            cost = self.expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            # print("cost: " + str(cost))
            if cost is type(list):
                cost = cost[1]
            if cost > maxCost:
                maxCost = cost
                actionCost = (action, cost)

        return actionCost

    """
    This function is similar to minval(or maxval) of minimax. Only difference is it does not take min of the possible 
    utilities instead it takes expectation of each action. Here each action is taken to be equally likely.
    """
    def expectVal(self, gameState, depth, agentIndex):
        actionCost = ("init", float("inf"))
        expectation = 0
        utility = 0
        for action in gameState.getLegalActions(agentIndex):
            cost = self.expectimax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            if type(cost) is tuple:
                utility = cost[1]
            else:
                utility = cost
            expectation = expectation + (utility / len(gameState.getLegalActions(agentIndex)))

        return expectation


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: If the ghost is too close in any state return high neg value to show such a state is not desirable.
    Else Find the closest food dot distance. If the action takes pacman closer, we have to return a higher value.
    So take 1/dist to food. Add this with actual score.
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    distToGhost = []
    for ghosts in ghosts:
        distToGhost.append(util.manhattanDistance(ghosts.getPosition(), pos))
    minDistToGhost = min(distToGhost)
    if minDistToGhost < 1.0:
        return -9999

    distToFood = 9999
    for food in food.asList():
        dist = util.manhattanDistance(food, pos)
        if dist < distToFood:
            distToFood = dist

    return currentGameState.getScore() + 1.0 / distToFood



# Abbreviation
better = betterEvaluationFunction
