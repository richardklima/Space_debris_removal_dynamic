import numpy as np
import random
from mdp import getFeatureVector
NEG_NUMBER = -1E10
finalStateYear = 2117

actionVector = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]


def update_Q(Q_table, alpha, gamma, state, new_state, action, reward):
    thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    indx = 0
    for thr in thresholds:
        if thr == action:
            break
        indx += 1
    possibleActions = []
    possibleActions.append(action)
    if action != 12000:
        possibleActions.append(thresholds[indx + 1])
    if action != 1000:
        possibleActions.append(thresholds[indx - 1])
    listValues = []
    for act in possibleActions:
        listValues.append(Q_table[new_state.totDebrisLevel, new_state.year, act])

    if state.year == finalStateYear:
        Q_table[state.totDebrisLevel, state.year, action] = (1 - alpha) * Q_table[state.totDebrisLevel, state.year, action] + alpha * (reward)
    else:
        maxActionIndex = np.argmax(listValues)
        value = listValues[maxActionIndex]
        Q_table[state.totDebrisLevel, state.year, action] = (1 - alpha) * Q_table[state.totDebrisLevel, state.year, action] + alpha * (reward + gamma * value)

    return Q_table


def initQtable(debrisLevels, actionVector, years):
    Q_table = dict()
    for dbLvl in debrisLevels:
        for yr in years:
            for act in actionVector:
                # Q_table[dbLvl, yr, act] = NEG_NUMBER
                Q_table[dbLvl, yr, act] = 0

    return Q_table


def epsilon_greedy_strat(Q_table, state, targetThreshold, epsilon, atBeginning):
    thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    indx = 0
    for thr in thresholds:
        if thr == targetThreshold:
            break
        indx += 1

    possibleActions = []
    if atBeginning:
        possibleActions = thresholds
    else:
        possibleActions.append(targetThreshold)
        if targetThreshold != 12000:
            possibleActions.append(thresholds[indx + 1])
        if targetThreshold != 1000:
            possibleActions.append(thresholds[indx - 1])

    if random.random() < epsilon:
        action = np.random.choice(possibleActions)

    else:
        listValues = []
        for act in possibleActions:
            listValues.append(Q_table[state.totDebrisLevel, state.year, act])
        maxActionIndex = np.argmax(listValues)
        action = possibleActions[maxActionIndex]

    return action


def update_weights(weights, alpha, gamma, state, action, state_next, reward):
    thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    indx = 0
    for thr in thresholds:
        if thr == action:
            break
        indx += 1
    possibleActions = []
    # if we are at beginning of the game we let the player to choose any action
    possibleActions.append(action)
    if action != 12000:
        possibleActions.append(thresholds[indx + 1])
    if action != 1000:
        possibleActions.append(thresholds[indx - 1])

    size_features = len(weights)

    phi = getFeatureVector(state, action, size_features)
    value = sum([i * j for i, j in zip(phi, weights)])

    if state_next.year == finalStateYear:
        td_error = reward - value

    else:
        # phi = getFeatureVectFull(state_def, state_att, action_def, size_of_map, size_features)
        value_next_list = []
        for act in possibleActions:
            phi_next = getFeatureVector(state_next, act, size_features)
            value_next_list.append(sum([i * j for i, j in zip(phi_next, weights)]))
        value_next = max(value_next_list)

        td_error = reward + gamma * value_next - value
    for i in range(len(weights)):
        weights[i] += alpha * td_error * phi[i]

    return weights, td_error


def epsilon_greedy_fnApprox(weights, state, targetThreshold, epsilon, atBeginning):
    thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    indx = 0
    for thr in thresholds:
        if thr == targetThreshold:
            break
        indx += 1
    possibleActions = []
    # if we are at beginning of the game we let the player to choose any action
    if atBeginning:
        possibleActions = thresholds
    else:
        possibleActions.append(targetThreshold)
        if targetThreshold != 12000:
            possibleActions.append(thresholds[indx + 1])
        if targetThreshold != 1000:
            possibleActions.append(thresholds[indx - 1])

    value_list = []
    size_features = len(weights)
    if random.random() < epsilon:
        action = np.random.choice(possibleActions)
    else:
        for act in possibleActions:
            phi = getFeatureVector(state, act, size_features)
            value_list.append(sum([i * j for i, j in zip(phi, weights)]))
        action = possibleActions[np.argmax(value_list)]
    return action
