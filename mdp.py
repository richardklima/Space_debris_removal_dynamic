import re

file_inputParameters = "params.txt"
fParam = open(file_inputParameters)
START_YEAR = int(re.sub("[^0-9]", "", fParam.next()))
timeHorizon = int(re.sub("[^0-9]", "", fParam.next()))
TIME_STEP = int(re.sub("[^0-9]", "", fParam.next()))
ratioC_L_C_R = float(re.findall("\d+\.\d+", fParam.next())[0])
collAvoidanceSuccess = float(re.findall("\d+\.\d+", fParam.next())[0])
rewardDiscountGamma = float(re.findall("\d+\.\d+", fParam.next())[0])
alpha = float(re.findall("\d+\.\d+", fParam.next())[0])
share_IA = float(re.findall("\d+\.\d+", fParam.next())[0])
totIter = int(re.sub("[^0-9]", "", fParam.next()))
debrisLevelStep = int(re.sub("[^0-9]", "", fParam.next()))

actionVector = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
debrisLevels = [i for i in range(13000, 350000, debrisLevelStep)]
years = [i for i in range(START_YEAR, START_YEAR + timeHorizon, TIME_STEP)]


class MDP:

    def __init__(self, thresholdCurves):
        self.thresholdCurves = thresholdCurves

    def getReward(self, expLostAssets, expRemoved):

        reward = -(expLostAssets + ratioC_L_C_R * expRemoved)

        return reward

    def getReward_multiAgent(self, expLostAssets, expRemoved):

        reward = -(share_IA * expLostAssets + ratioC_L_C_R * expRemoved)

        return reward

    def getReward_multiAgent_att(self, expLostAssets, expRemoved):

        reward = -((1 - share_IA) * expLostAssets + ratioC_L_C_R * expRemoved)

        return reward

    # not functional at the moment, need to reimplement with the current state definition
    def getDiscountedReward(self, state, action):
        indxAction = 0
        for curve in self.thresholdCurves:
            if curve.threshold == action:
                break
            indxAction += 1

        ix = 0
        sumDiscReward = 0
        for i in range(state.year - START_YEAR, 100):
            reward = -(state.targetCurve.lostAssets[i] + ratioC_L_C_R * state.targetCurve.removed[i])
            sumDiscReward += pow(rewardDiscountGamma, ix) * reward
            ix += 1
        averagedEndReward = 0
        size = 0
        for j in range(95, 100):
            averagedEndReward += -(state.targetCurve.lostAssets[j] + ratioC_L_C_R * state.targetCurve.removed[j])
            size += 1

        averagedEndReward = averagedEndReward / size
        infSum2 = pow(rewardDiscountGamma, ix) * averagedEndReward * (1 / (1 - rewardDiscountGamma))

        sumDiscReward += infSum2

        return sumDiscReward

    def getDiscountedRewardsFromSequence(self, states, actions, lostAssets, removed):
        averagedEndReward = 0
        size = 0
        for j in range(95, 100):
            averagedEndReward += -(lostAssets[j] + ratioC_L_C_R * removed[j])
            size += 1
        averagedEndReward = averagedEndReward / size
        infSum = pow(rewardDiscountGamma, 100) * averagedEndReward * (1 / (1 - rewardDiscountGamma))

        discRewardList = []
        ix = 0
        for st in states:
            reward = 0
            ixPower = 0
            print ix, (st.year - START_YEAR)
            for i in range(st.year - START_YEAR, 100):
                # reward += -(lostAssets[i] * C_L + removed[i] * C_R)
                reward += -(lostAssets[i] + ratioC_L_C_R * removed[i])
                sumDiscReward += pow(rewardDiscountGamma, ixPower) * reward
                ixPower += 1
            ix += 1
            sumDiscReward += infSum
            discRewardList.append(sumDiscReward)
        return discRewardList

    class state:

        def __init__(self, totDebrisLevel, year):
            self.totDebrisLevel = totDebrisLevel
            self.year = year

    # threshold is defined by action the player took
    def transitionOfStates(self, state, action, approach, timeStep):
        targetCurve = 0
        movingUp = 0
        movingDown = 0
        stayingOnSameLevel = 0
        # print action
        for curve in self.thresholdCurves:
            if curve.threshold == action:
                targetCurve = curve
                break
        if targetCurve.totDebris[state.year - START_YEAR] == state.totDebrisLevel:
            stayingOnSameLevel = 1
        elif targetCurve.totDebris[state.year - START_YEAR] > state.totDebrisLevel:
            movingUp = 1
        else:
            movingDown = 1

        if approach == "optimistic":
            if stayingOnSameLevel:
                index = state.year - START_YEAR + timeStep

            elif movingUp:
                for ixDebris in range(len(targetCurve.totDebris[:-1])):
                    if state.totDebrisLevel >= targetCurve.totDebris[ixDebris] and state.totDebrisLevel < targetCurve.totDebris[ixDebris + 1]:
                        index = ixDebris + timeStep
                        break
                    elif state.totDebrisLevel < targetCurve.totDebris[0]:
                        index = timeStep
                        break

            elif movingDown:
                index = state.year - START_YEAR + timeStep

        elif approach == "pessimistic":
            if stayingOnSameLevel:
                index = state.year - START_YEAR + timeStep

            elif movingUp:

                if len(targetCurve.totDebris) > (state.year - START_YEAR + timeStep):
                    index = state.year - START_YEAR + timeStep
                else:
                    index = state.year - START_YEAR

            elif movingDown:
                # moving right - we need to find index for given level of total debris
                for ixDebris in range(len(targetCurve.totDebris[:-1])):
                    if state.totDebrisLevel >= targetCurve.totDebris[ixDebris] and state.totDebrisLevel < targetCurve.totDebris[ixDebris + 1]:
                        index = ixDebris + timeStep
                        break
                    elif state.totDebrisLevel >= targetCurve.totDebris[-1]:
                        index = len(targetCurve.totDebris) - 1
                        break

        expCollisionsCumul = []
        if index < len(targetCurve.removed):
            expRemoved = round(sum(targetCurve.removed[(index - timeStep):index]), 7)
            expLostAssets = sum(targetCurve.lostAssets[(index - timeStep):index])
            totDebrisLevel = state.totDebrisLevel + (targetCurve.totDebris[index] - targetCurve.totDebris[index - timeStep])
        else:
            expRemoved = round(sum(targetCurve.removed[-timeStep:]), 7)
            expLostAssets = sum(targetCurve.lostAssets[-timeStep:])
            inx = index - timeStep
            totDebrisLevel = state.totDebrisLevel + (targetCurve.totDebris[inx] - targetCurve.totDebris[inx - timeStep])

        return expLostAssets, totDebrisLevel, expRemoved, targetCurve.threshold

    def findCurve(self, expRemoved, state, approach, timeStep):
        targetCurve = 0
        movingUp = 0
        movingDown = 0
        stayingOnSameLevel = 0

        listOfExpRemoved = []

        for curve in self.thresholdCurves:
            expLostAssets, totDebrisLevel, expRemovedPom, targetThreshold = self.transitionOfStates(state, curve.threshold, approach, timeStep)
            listOfExpRemoved.append(expRemovedPom)

        targetCurve = 0
        expRemovedForCurve = listOfExpRemoved[0]
        if expRemovedForCurve <= expRemoved:
            targetCurve = self.thresholdCurves[0]
        else:
            for curveIndex in range(1, len(listOfExpRemoved)):
                if listOfExpRemoved[curveIndex - 1] > expRemoved and listOfExpRemoved[curveIndex] <= expRemoved:
                    targetCurve = self.thresholdCurves[curveIndex]
                    break
        return targetCurve


def getFeatureVector(state, action, size):

    indx = 0
    actIndex = 0
    for act in actionVector:
        if action == act:
            actIndex = indx
            break
        indx += 1

    indx = 0
    yearIndex = 0
    for yr in years:
        if state.year == yr:
            yearIndex = indx
            break
        indx += 1

    debrIndex = 0
    for indx in range(len(debrisLevels[:-1])):
        if state.totDebrisLevel > debrisLevels[indx] and state.totDebrisLevel < debrisLevels[indx + 1]:
            debrIndex = indx
            break

    phi = [0 for i in range(size)]
    sizeWoActions = size / len(actionVector)
    phi[actIndex * sizeWoActions + yearIndex] = 1
    phi[actIndex * sizeWoActions + len(years) + debrIndex] = 1

    # bias term to scale function independently to features
    phi[-1] = 1
    return phi


def approximateState(state):
    debrIndex = 0
    for indx in range(len(debrisLevels[:-1])):
        if state.totDebrisLevel > debrisLevels[indx] and state.totDebrisLevel < debrisLevels[indx + 1]:
            debrIndex = indx
            break

    newDebrisLevel = debrisLevels[debrIndex]
    return newDebrisLevel
