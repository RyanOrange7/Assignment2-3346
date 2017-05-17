# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
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
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    print "*"
    print successorGameState
    print oldFood
    print newPos
    print newGhostStates
    print newScaredTimes
    print successorGameState.getScore()

    if successorGameState.isWin():
      return float("inf") - 20
    scared = sum(newScaredTimes)
    ghostDistance = util.manhattanDistance(currentGameState.getGhostPosition(1),newPos)
    score = max(ghostDistance,1) + successorGameState.getScore()
    foodlist = successorGameState.getFood().asList()
    close = 100
    for foodPosition in foodlist:
      dist = util.manhattanDistance(foodPosition,newPos)
      if(dist < close):
        close = dist
    if(currentGameState.getNumFood()> successorGameState.getNumFood()):
      score += 100
    if action == Directions.STOP:
      score -= 3
    score -= 3 * close
    capsules = currentGameState.getCapsules()
    if successorGameState.getPacmanPosition() in capsules:
      score += 120
    return score
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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '1'):
    self.index = 0 # Pacman is always agent index 0
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    """
    Max and min value functions are implemented  to be used in the main getAction function which represents the MINIMAX-DECISION function
    """
    def maxValue(state,depth):
      if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state)
      v = -(float("inf"))
      legalActions = state.getLegalActions()
      for action in legalActions:
          v = max(v, minValue(state.generateSuccessor(0,action), (depth - 1),1))
      return v

    def minValue(state,depth,agentIndex):
        if state.isWin() or state.isLose() or depth == 0:
          return self.evaluationFunction(state)
        v = float("inf")
        legalActions = state.getLegalActions()
        if agentIndex == (state.getNumAgents() - 1): #for number of ghosts
          for action in legalActions:
            v = min(v, maxValue(state.generateSuccessor(0,action), depth - 1))
        else:
          for action in legalActions:
            v = min(v, minValue(state.generateSuccessor(0,action),depth,++agentIndex),)
        return v

    
    score = -(float("inf"))
    legal = gameState.getLegalActions()
    for action in legal:
      successorState = gameState.generateSuccessor(0,action)
      previous = score
      score = max(score, minValue(successorState,self.depth,1))
      if score > previous:
        best = action
    return best



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    #MAX-VALUE function for alpha beta is the same as for minimax but also has the parameters alpha and beta
    #alpha represents the best option for max while beta is the same for min
    def maxValue(state,depth,alpha,beta):
      if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state)
      v = -(float("inf"))
      legalActions = state.getLegalActions()
      for action in legalActions:
          v = max(v, minValue(state.generateSuccessor(0,action), (depth - 1),1,alpha,beta))
          if (v >= beta):
            return v
          else:
            alpha = max(alpha,v)
      return v
    #MAX-VALUE function for alpha beta is the same as for minimax but also has the parameters alpha and beta
    #alpha represents the best option for max while beta is the same for min
    def minValue(state,depth,agentIndex,alpha,beta):
        if state.isWin() or state.isLose() or depth == 0:
          return self.evaluationFunction(state)
        v = float("inf")
        legalActions = state.getLegalActions()
        for action in legalActions:
           if agentIndex == (state.getNumAgents() - 1): #for number of ghosts
             v = min(v, maxValue(state.generateSuccessor(0,action), depth - 1,alpha,beta))
             if (v <= alpha):
               return v
             beta = min(beta,v)
           else:
               v = min(v, minValue(state.generateSuccessor(0,action),depth,++agentIndex,alpha,beta))
               if (v <= alpha):
                 return v
               beta = min(beta,v)
        return v

    
    score = -(float("inf"))
    alpha = -(float("inf"))
    beta = float("inf")
    legal = gameState.getLegalActions()
    for action in legal:
      successorState = gameState.generateSuccessor(0,action)
      previous = score
      score = max(score, minValue(successorState,self.depth,1,alpha,beta))
      if score > previous:
        best = action
      if score >= beta:
        return best
      alpha = max(alpha,score)
    return best


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
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

