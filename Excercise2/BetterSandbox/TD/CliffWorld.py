"""
Created/updated on Nov 19 2019

@author: Robert Lowe adapted code (to include maze plots and Sarsa)
"""

import itertools
import matplotlib
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
from operator import itemgetter 

matplotlib.style.use('ggplot')

env = CliffWalkingEnv() # gym environment found in CliffWorld.py


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    # can uncomment below comments for testing purposes
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        #print("action probs", A)
        #input("now go!")
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        #print("action probs", A) 
        #input("now go!")
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environmen (see CliffWorld.py)
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # The current policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)


            # for console outputs (see below)
            vals = list(itemgetter(state)(states)) 
            actionMap[vals[0]][vals[1]] = np.argmax(action_probs)
            
        
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            best_next_action = np.argmax(Q[next_state])
            
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            valueTakenMap[vals[0]][vals[1]] = round(Q[state][action],2)
            best_action = np.argmax(Q[state]) # used for finding max values
            valueMap[vals[0]][vals[1]] = round(Q[state][best_action],2)
            
            if done:
                break
                
            state = next_state
            
 ################################# code for step by step testing #############          
            #console prints: uncomment to see step by step:
            # 1. agent action and state transition
            # 2. valuation for the action taken in the state, i.e. Q(s,a)
            """
            print('\n')
            print("############## episode",i_episode,"step",t,"##############")
            print('\n')
            #print("",actionTakenMap)
            if action == 0:
                print("Agent went UP")
            elif action == 1:
                print("Agent went RIGHT")
            elif action == 2:
                print("Agent went DOWN")
            else:
                print("Agent went LEFT")
                
            print("Position after Action:")
            env.render()
            #print('\n')
            print("Q value for taking action from the state:")
            print("",valueTakenMap) 
            input("Press Enter (in Console) to continue")
            #print("",valueMap) 
            #wait = input("proceed immediately!")
            
        print('\n')
        print("Position at end of episode:")
        env.render()
        input("Enter for next episode")
        """
        """
        if (i_episode) % 10 == 0:
            print("After episode",i_episode)
            print("max Q values:")
            print("",valueMap) 
            print('\n')
            print("policy values:")
            print("",actionMap)
            input("Enter to continue")
        """
    return Q, stats

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        # for console outputs (see below)
        vals = list(itemgetter(state)(states)) 
        actionMap[vals[0]][vals[1]] = np.argmax(action_probs)
        
        # One step in the environment
        for t in itertools.count():
            # Take a step
            next_state, reward, done, _ = env.step(action)
            
            # Pick the next action
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            td_target = reward + discount_factor * Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            
            valueTakenMap[vals[0]][vals[1]] = round(Q[state][action],2)
            best_action = np.argmax(Q[state]) # used for finding max values
            valueMap[vals[0]][vals[1]] = round(Q[state][best_action],2)
            
            if done:
                break
                
            action = next_action
            state = next_state  
            
            # for console outputs (see below)
            vals = list(itemgetter(state)(states)) 
            actionMap[vals[0]][vals[1]] = np.argmax(next_action_probs)
            
            ################################# code for step by step testing #############          
            #console prints: uncomment to see step by step:
            # 1. agent action and state transition
            # 2. valuation for the action taken in the state, i.e. Q(s,a)
            """
            print('\n')
            print("############## episode",i_episode,"step",t,"##############")
            print('\n')
            #print("",actionTakenMap)
            if action == 0:
                print("Agent went UP")
            elif action == 1:
                print("Agent went RIGHT")
            elif action == 2:
                print("Agent went DOWN")
            else:
                print("Agent went LEFT")
                
            print("Position after Action:")
            env.render()
            #print('\n')
            print("Q value for taking action from the state:")
            print("",valueTakenMap) 
            input("Press Enter (in Console) to continue")
            #print("",valueMap) 
            #wait = input("proceed immediately!")
            
        print('\n')
        print("Position at end of episode:")
        env.render()
        input("Enter for next episode")
        """
        """
        if (i_episode) % 10 == 0:
            print("After episode",i_episode)
            print("max Q values:")
            print("",valueMap) 
            print('\n')
            print("policy values:")
            print("",actionMap)
            input("Enter to continue")
        """
    
    return Q, stats

gridSizeX = 4
gridSizeY = 12
valueMap = np.zeros((gridSizeX, gridSizeY))
valueTakenMap = np.zeros((gridSizeX, gridSizeY))
actionMap = np.empty((gridSizeX, gridSizeY))
actionMap[:] = [np.nan]

print("", valueMap)
wait = input("Press Enter please!") 
states = [[i, j] for i in range(gridSizeX) for j in range(gridSizeY)]
print("states", states)


print(env.reset())

#Q, stats = sarsa(env,500) #50000)
Q, stats = q_learning(env,500)#50000) 

plotting.plot_episode_stats(stats)

input("Final max Q values:")
print("",valueMap) 
input("Final position/action on grid")
env.render()
input("Final policy:")
print("",actionMap)

