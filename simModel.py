# import matplotlib
# matplotlib.use('Agg')
import re
import mdp
import matplotlib.pyplot as plt
import pylab
import qLearning
import numpy as np
import math
import random

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

finalYear = START_YEAR + timeHorizon
C_L = 1.0
C_R = ratioC_L_C_R * C_L
infiniteRewSum = 0
actionVector = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
debrisLevels = [i for i in range(13000, 350000, debrisLevelStep)]
years = range(2017, 2017 + 101, TIME_STEP)

approach = "pessimistic"
# approach = "optimistic"

epsilon = 0.1
fileN = "infSum" + str(ratioC_L_C_R) + "_" + str(alpha) + "_" + str(rewardDiscountGamma) + "_" + str(totIter) + ".txt"
figN = "infSum" + str(ratioC_L_C_R) + "_" + str(alpha) + "_" + str(rewardDiscountGamma) + "_" + str(totIter) + ".pdf"

print "+++++" * 30


class ThresholdCurve:
    def __init__(self, threshold, totDebris, collidedOrAvoidedAssets, collAll, removed):
        self.threshold = threshold
        self.collidedOrAvoidedAssets = collidedOrAvoidedAssets
        self.lostAssets = [(1 - collAvoidanceSuccess) * coll for coll in collidedOrAvoidedAssets]
        self.totDebris = totDebris
        self.collAll = collAll
        collAllCumul = []
        for i in range(len(collAll)):
            collList = []
            for yr in range(len(collAll[0])):
                sumOfColl = 0
                for j in range(i, len(collAll)):
                    sumOfColl += collAll[j][yr]
                collList.append(sumOfColl)
            collAllCumul.append(collList)

        self.collAllCumul = collAllCumul
        self.removed = removed


def plotDebEvol(scenariosO, scenariosP, case):

    colorOfExper = ['r', 'g', 'b', 'k', 'y', 'm', 'c', 'b', 'k', 'y']
    colorPessOpt = ['g', 'r']
    lnstyles = ['-.', '--', '-', '-', '--']
    if case == 1:
        # actions = ["no removal", "above 1000", "above 2000", "above 3000", "above 4000", "above 5000", "above 6000", "above 8000", "above 9000", "above 10000"]
        # actions = ["approx - no removal", "approx - above 1000", "approx - above 3000", "approx - above 5000", "approx - above 8000"]
        # actions = ["approx 20y - 8k, 20y - 4k, 20y - 6k, 40y - 3k"]
        # actions = ["approx 40y - 5k, 30y - NR, 30y - 1k"]
        # actions = ["approx 20y - 9k, 20y - 8k, 20y - NR, 40y - 6k"]
        # actions = ["approx 9k, 8k, NR, 6k - optimistic", "approx 9k, 8k, NR, 6k - pessimistic"]
        # actions = ["approx 2k, 3k, 1k, 2k - optimistic", "approx 2k, 3k, 1k, 2k - pessimistic"]
        actions = ["approx - optimistic", "approx - pessimistic"]
        # actions = ["approx 8k, 4k, 6k, 3k - optimistic", "approx 8k, 4k, 6k, 3k - pessimistic"]
        # actions = ["approx - 5k, NR, 1k - optimistic", "approx - 5k, NR, 1k - pessimistic"]
        # actions = ["approx - NR, 1k, 5k - optimistic", "approx - NR, 1k, 5k - pessimistic"]
    if case == 2:
        actions = ["remove 0", "remove 0.5/2y", "remove 1/2y", "remove 2/2y"]

    years = range(2017, 2017 + 101, 2)
    print len(scenariosO[0])

    labelsExp = ["approx [NR, NR]", "approx [NR, 1k]", "approx [NR, 3k]", "approx [NR, 5k]", "approx [NR, 8k]"]
    # plt.figure()

    for exp in range(len(scenariosO)):
        plt.plot(years, scenariosO[exp], linestyle=lnstyles[0], color=colorPessOpt[0], linewidth=4.0, label=actions[0])
        plt.plot(years, scenariosP[exp], linestyle=lnstyles[1], color=colorPessOpt[1], linewidth=4.0, label=actions[1])
    changes = [40, 70]
    experiment = 0

    for ch in range(len(changes)):
        # plt.axvline(x=(2017 + changes[ch]), ymin=0, ymax=1, color='k', label="change [20y, 40y, 60y]" if ch == 0 else "")
        # plt.axvline(x=(2018 + changes[ch]), ymin=0, ymax=1, color='k', label="change [50y]" if ch == 0 else "")
        plt.axvline(x=(2018 + changes[ch]), ymin=0, ymax=1, color='k', label="change [40y, 70y]" if ch == 0 else "")
        # plt.axvline(x=(2018 + changes[ch]), ymin=0, ymax=1, color='k', label="change [40y, 30y]" if ch == 0 else "")

    pylab.xlim(2015, 2120)
    plt.ylabel('total number of objects', fontsize=20)
    plt.xlabel('year', fontsize=20)
    plt.title('Objects number evolution', fontsize=20)
    # plt.grid()
    # plt.legend(loc='upper left')
    # plt.legend(loc='upper left', fancybox=True, framealpha=0.5)
    plt.legend(loc='lower right', fancybox=True, framealpha=0.5)
    plt.show()


def getThresholdCurves():
    fileNameThresholds = "Data/thresholdCurves.txt"
    fileNameThresholdsEnds = "Data/thresholdCurvesEnds.txt"
    f = open(fileNameThresholds)
    f2 = open(fileNameThresholdsEnds)
    thresholdNo = 10
    thresholdCurves = []
    for i in range(thresholdNo):

        threshold = int(f.readline())
        averageDebStr = f.readline()
        averageDebStr = averageDebStr[1:-2]
        averageDebArray = averageDebStr.split(",")
        averageDeb = []
        for debNo in averageDebArray:
            averageDeb.append(float(debNo))

        if i > 0:
            threshold2 = int(f2.readline())
            averageDebStr2 = f2.readline()
            averageDebStr2 = averageDebStr2[1:-2]
            averageDebArray2 = averageDebStr2.split(",")
            averageDeb2 = []
            for debNo in averageDebArray2:
                averageDeb2.append(float(debNo))

            averageDeb.extend(averageDeb2)
        else:
            for jj in range(14):
                a = f2.readline()

        averageCollOrAvoidStr = f.readline()
        averageCollOrAvoidStr = averageCollOrAvoidStr[1:-2]
        averageCollOrAvoidArray = averageCollOrAvoidStr.split(",")
        averageCollOrAvoid = []
        for debNo in averageCollOrAvoidArray:
            averageCollOrAvoid.append(float(debNo))

        if i > 0:
            averageCollOrAvoidStr2 = f2.readline()
            averageCollOrAvoidStr2 = averageCollOrAvoidStr2[1:-2]
            averageCollOrAvoidArray2 = averageCollOrAvoidStr2.split(",")
            averageCollOrAvoid2 = []
            for debNo in averageCollOrAvoidArray2:
                averageCollOrAvoid2.append(float(debNo))

            averageCollOrAvoid.extend(averageCollOrAvoid2)

        collThresholdsAll = []
        for j in range(thresholdNo):
            collThresholdsStr = f.readline()
            collThresholdsStr = collThresholdsStr[1:-2]
            collThresholdsArray = collThresholdsStr.split(",")
            collThresholds = []
            for debNo in collThresholdsArray:
                collThresholds.append(float(debNo))
            collThresholdsAll.append(collThresholds)

        if i > 0:
            collThresholdsAll2 = []
            for j in range(thresholdNo):
                collThresholdsStr2 = f2.readline()
                collThresholdsStr2 = collThresholdsStr2[1:-2]
                collThresholdsArray2 = collThresholdsStr2.split(",")
                collThresholds2 = []
                for debNo in collThresholdsArray2:
                    collThresholds2.append(float(debNo))
                # collThresholdsAll2.append(collThresholds2)
                collThresholdsAll[j].extend(collThresholds2)

        removedStr = f.readline()
        removedStr = removedStr[1:-2]
        removedArray = removedStr.split(",")
        averageRemoved = []
        for debNo in removedArray:
            averageRemoved.append(float(debNo))

        if i > 0:
            removedStr2 = f2.readline()
            removedStr2 = removedStr2[1:-2]
            removedArray2 = removedStr2.split(",")
            averageRemoved2 = []
            for debNo in removedArray2:
                averageRemoved2.append(float(debNo))

            averageRemoved.extend(averageRemoved2)

        removedDistributed = []
        for indx in range(1, len(averageRemoved), 2):
            remIn2Years = averageRemoved[indx]
            removedDistributed.append(remIn2Years / 2.)
            removedDistributed.append(remIn2Years / 2.)

        line = f.readline()
        line2 = f2.readline()

        if i == 0:
            pom = ThresholdCurve(threshold, averageDeb, averageCollOrAvoid, collThresholdsAll, removedDistributed)
        else:
            thresholdCurves.append(ThresholdCurve(threshold, averageDeb, averageCollOrAvoid, collThresholdsAll, removedDistributed))

    thresholdCurves.append(pom)
    f.close()
    f2.close()
    return thresholdCurves


# Test method to plot fixed scenarios
def plotCurve():
    thresholdCurves = getThresholdCurves()

    currentCurve = thresholdCurves[0]
    currentYear = 2080
    action = 8000

    approachOptim = "optimistic"
    approachPess = "pessimistic"
    statesOptim = []
    statesPess = []
    scenariosOptim = []
    scenariosPess = []
    actions = []

    mdp1 = mdp.MDP(thresholdCurves)
    # actions2 = [20000, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000]
    # actions2 = [12000, 1000, 3000, 5000, 8000]
    # seqOfAction = [12000, 1000, 5000]
    seqOfAction = [5000, 12000, 1000]
    # seqOfAction = [1000, 20000, 5000]
    # seqOfAction = [2000, 3000, 1000, 2000]
    # seqOfAction = [8000, 4000, 6000, 3000]
    # seqOfAction = [9000, 8000, 12000, 6000]
    # for i in range(len(actions2)):
    #     actions = []
    #     for j in range(25):
    #         actions.append(actions2[0])
    #     for j in range(25):
    #         actions.append(actions2[i])
    #     # actions = [4000] * 50
    #     # debris0 = 13000
    #     debris0 = thresholdCurves[0].totDebris[0]
    #     states.append(debris0)
    #     for i in range(50):
    #         yr = (2 * i) + 2017
    #         # print yr
    #         states.append(mdp(thresholdCurves, yr, states[i], actions[i], approach))
    #     scenarios.append(states[:-1])
    #     states = []
    #     actions = []
    for j in range(20):
        actions.append(seqOfAction[0])
        # actions.append(20000)
    for j in range(15):
        actions.append(seqOfAction[1])
        # actions.append(1000)
    for j in range(15):
        actions.append(seqOfAction[2])
    # for j in range(20):
    #     actions.append(seqOfAction[3])
    # print actions
    # for j in range(20):
    #     actions.append(seqOfAction[3])
    # actions = [4000] * 50
    statesOptim = []
    statesPess = []
    timeStep = 2
    debris0 = thresholdCurves[4].totDebris[0]
    totDebrisLevel = debris0
    state1O = mdp1.state(totDebrisLevel, START_YEAR)
    state1P = mdp1.state(totDebrisLevel, START_YEAR)
    targetThreshold = 0

    statesOptim.append(debris0)
    statesPess.append(debris0)
    yr = 2017
    i = 0
    while (yr - START_YEAR) < 100:
        expLostAssets, totDebrisLevelO, expRemoved, targetThreshold = mdp1.transitionOfStates(state1O, actions[i], approachOptim, timeStep)
        expLostAssets, totDebrisLevelP, expRemoved, targetThreshold = mdp1.transitionOfStates(state1P, actions[i], approachPess, timeStep)
        state1O = mdp1.state(totDebrisLevelO, yr)
        state1P = mdp1.state(totDebrisLevelP, yr)

        statesOptim.append(state1O.totDebrisLevel)
        statesPess.append(state1P.totDebrisLevel)
        i += 1
        yr = (timeStep * i) + 2017
    scenariosOptim.append(statesOptim)
    scenariosPess.append(statesPess)
    plotDebEvol(scenariosOptim, scenariosPess, 1)


def plotRewards():

    thresholdCurves = getThresholdCurves()
    mdp1 = MDP(thresholdCurves)

    # thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    # thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000]
    thresholds = [5000]

    debris0 = mdp1.thresholdCurves[0].totDebris[0]
    approach = "pessimistic"
    years = range(2017, 2017 + 100, 2)
    actionsLabels = ["above 1000", "above 2000", "above 3000", "above 4000", "above 5000", "above 6000", "above 8000", "above 9000", "above 10000", "no removal"]
    colorOfExper = ['g', 'b', 'k', 'y', 'm', 'c', 'k--', 'b--', 'g--', 'r', 'y']
    ix = 0
    for j in thresholds:
        immReward = []
        discRewardList = []
        expLostAList = []
        expCollList = []
        expRemList = []
        totDebrisLevel = debris0
        state0 = mdp1.state(totDebrisLevel, 2017)
        state1 = state0
        action = j
        for i in range(50):
            yr = (2 * i) + 2017
            expLostAssets, totDebrisLevel, expRemoved, targetThreshold = mdp1.transitionOfStates(state1, action, approach, TIME_STEP)

            state1 = mdp1.state(totDebrisLevel, yr)

            reward = mdp1.getReward(expLostAssets, expRemoved)
            immReward.append(reward)
        print sum(immReward), j

        plt.plot(years, immReward, colorOfExper[ix], label=actionsLabels[ix])
        ix += 1
    pylab.xlim(2015, 2120)
    plt.ylabel('immediate reward', fontsize=20)
    # plt.ylabel('removed', fontsize=20)
    # plt.ylabel('lost assets', fontsize=20)
    plt.xlabel('year', fontsize=20)
    # plt.title('Objects number evolution', fontsize=20)
    # plt.title('Immediate reward evolution - C_R/C_L = 0.1', fontsize=20)
    # plt.title('Immediate reward evolution - threshold = 3000', fontsize=20)
    plt.title('Expected # of removed', fontsize=20)
    # plt.title('Expected # of lost assets', fontsize=20)
    plt.grid()
    # plt.legend(loc='upper left')
    plt.legend(loc='lower left')
    plt.show()


def learnStrategy():
    thresholdCurves = getThresholdCurves()
    mdp1 = mdp.MDP(thresholdCurves)
    debris0 = mdp1.thresholdCurves[0].totDebris[0]
    action0 = 3000
    attackerStrategy = [2000, 2000, 2000, 2000, 2000, 12000, 12000, 12000, 12000, 12000, 12000]
    attackerStrategy1 = [5000, 12000]
    defenderFixedStrats = []
    for ii in range(len(thresholdCurves)):
        defenderFixedStrats.append([(ii + 1) * 1000 for jj in range(timeHorizon / TIME_STEP + 1)])
    fixedToPlay = 4

    finalStrategy = []
    finalDebrisLevel = []
    Q_table = qLearning.initQtable(debrisLevels, actionVector, years)
    print len(Q_table)

    # # fn approx
    # lenOfPhi = (timeHorizon / TIME_STEP) + len(range(13000, 350000, debrisLevelStep))
    # # length is the total size + 1 to account for a bias term
    # weights = [random.random() for j in range(lenOfPhi * len(actionVector) + 1)]

    discRewardEvol = []
    immRewardEvol = []
    listOfQvalues = []
    no_states = 0
    for iteration in range(totIter):

        totDebrisLevel = debris0
        state0 = mdp1.state(totDebrisLevel, 2017)
        state = state0
        rewardList = []
        discReward = []
        targetThreshold = action0
        atBeginning = 1
        if iteration == totIter - 1:
            finalDebrisLevel.append(totDebrisLevel)

        i = 0
        yr = 2017
        while (yr - START_YEAR) < 100:
            stateApprox = mdp1.state(mdp.approximateState(state), state.year)

            if iteration == totIter - 1:
                action = qLearning.epsilon_greedy_strat(Q_table, stateApprox, targetThreshold, 0, atBeginning)
            else:
                action = qLearning.epsilon_greedy_strat(Q_table, stateApprox, targetThreshold, epsilon, atBeginning)

            atBeginning = 0

            expLostAssets, totDebrisLevel, expRemoved, targetThreshold = mdp1.transitionOfStates(state, action, approach, TIME_STEP)

            state_next = mdp1.state(totDebrisLevel, yr + TIME_STEP)

            state_nextApprox = mdp1.state(mdp.approximateState(state_next), state_next.year)

            if infiniteRewSum:
                # ======= last reward is infinite sum of future discounted rewards with constant values ==============
                if (yr + timeHorizon % TIME_STEP) == finalYear:
                    averagedEndReward = 0
                    indx = 0
                    for thr in thresholdCurves:
                        if thr.threshold == action:
                            break
                        indx += 1

                    size = 0
                    for j in range(95, 100):
                        averagedEndReward += -(thresholdCurves[indx].lostAssets[j] * C_L + thresholdCurves[indx].removed[j] * C_R)
                        size += 1

                    averagedEndReward = averagedEndReward / size
                    rewardInf = averagedEndReward * (1 / (1 - rewardDiscountGamma))
                    reward = mdp1.getReward(expLostAssets, expRemoved_def) + rewardInf

                else:
                    reward = mdp1.getReward(expLostAssets, expRemoved_def)
                # ================================================================================
            else:
                reward = mdp1.getReward(expLostAssets, expRemoved)

            discReward.append(pow(rewardDiscountGamma, i - 1) * reward)

            qLearning.update_Q(Q_table, alpha, rewardDiscountGamma, stateApprox, state_nextApprox, action, reward)
            state = state_next
            rewardList.append(reward)
            if iteration == totIter - 1:
                finalStrategy.append(action)
                finalDebrisLevel.append(totDebrisLevel)
            i += 1
            yr = (TIME_STEP * i) + 2017

        discRewardEvol.append(sum(discReward))
    print "step in debris size: ", debrisLevelStep
    print "Learned sequence of actions: ", finalStrategy
    print "Total number of explored states ", no_states
    # print weights
    print "discounted reward is :", sum(discReward)
    print "imm reward is :", sum(rewardList)
    # print "total error is: ", sum(td_errorList)

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    wins_MA = []
    return finalStrategy


def learnStrategyMultiAgent():
    thresholdCurves = getThresholdCurves()
    mdp1 = mdp.MDP(thresholdCurves)
    debris0 = mdp1.thresholdCurves[0].totDebris[0]
    action0 = 5000
    defenderFixedStrats = []
    for ii in range(len(thresholdCurves)):
        defenderFixedStrats.append([(ii + 1) * 1000 for jj in range(timeHorizon / TIME_STEP + 1)])
    fixedToPlay = 4

    finalStrategy = []
    finalStrategy_att = []
    finalDebrisLevel = []
    Q_table = qLearning.initQtable(debrisLevels, actionVector, years)
    Q_table_att = qLearning.initQtable(debrisLevels, actionVector, years)

    discRewardEvol = []
    immReward = []
    immReward_att = []
    listOfQvalues = []
    no_states = 0
    td_errorList = []
    td_errorList_att = []
    for iteration in range(totIter):

        totDebrisLevel = debris0
        state0 = mdp1.state(totDebrisLevel, 2017)

        state = state0
        discReward = []
        discReward_att = []
        targetThreshold_def = action0
        targetThreshold_att = action0

        atBeginning = 1

        if iteration == totIter - 1:
            finalDebrisLevel.append(totDebrisLevel)

        # for i in range(1, 50):
        i = 0
        yr = 2017
        while (yr - START_YEAR) < 100:

            # if iteration == 350:
            #     action = 3000
            stateApprox = mdp1.state(mdp.approximateState(state), state.year)

            if iteration == totIter - 1:
                # epsilon = 0
                # action = qLearning.epsilon_greedy_strat(Q_table, state, targetThreshold, 0)
                # print "learned ", Q_table[state.totDebrisLevel, state.year, action]
                # listOfQvalues.append(Q_table[state.totDebrisLevel, state.year, action])
                action = qLearning.epsilon_greedy_strat(Q_table, stateApprox, targetThreshold_def, 0, atBeginning)
                # action = 4000

                action_att = qLearning.epsilon_greedy_strat(Q_table_att, stateApprox, targetThreshold_att, 0, atBeginning)
                # action_att = 6000
                # fn approx
                # action = qLearning.epsilon_greedy_fnApprox(weights, state, targetThreshold, 0, atBeginning)
                # action = 4000
                # action_att = qLearning.epsilon_greedy_fnApprox(weights_att, state, targetThreshold_att, 0, atBeginning)
                # action_att = 12000
                # action = 12000
                # action = defenderFixedStrats[fixedToPlay][i]
            else:
                # action = qLearning.epsilon_greedy_strat(Q_table, state, targetThreshold, epsilon)
                # action = actionsLearned[i]
                # print atBeginning
                action = qLearning.epsilon_greedy_strat(Q_table, stateApprox, targetThreshold_def, epsilon, atBeginning)
                # action = 4000

                action_att = qLearning.epsilon_greedy_strat(Q_table_att, stateApprox, targetThreshold_att, epsilon, atBeginning)
                # action_att = 6000

                # fn approx
                # action = qLearning.epsilon_greedy_fnApprox(weights, state, targetThreshold, epsilon, atBeginning)
                # action = 4000
                # action_att = qLearning.epsilon_greedy_fnApprox(weights_att, state, targetThreshold_att, epsilon, atBeginning)
                # action_att = 12000
                # action = defenderFixedStrats[fixedToPlay][i]
            atBeginning = 0

            # pastAgentAction = action

            expLostAssets_def, totDebrisLevel_def, expRemoved_def, targetThreshold_def = mdp1.transitionOfStates(state, action, approach, TIME_STEP)

            # OPPONENT ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # print opponentAction
            # expLostAssets_att, totDebrisLevel_att, expRemoved_att, targetThreshold_att = mdp1.transitionOfStates(state, opponentAction, approach, TIME_STEP)
            expLostAssets_att, totDebrisLevel_att, expRemoved_att, targetThreshold_att = mdp1.transitionOfStates(state, action_att, approach, TIME_STEP)

            # print expRemoved_def
            expRemoved_total = expRemoved_def + expRemoved_att
            # print expRemoved_def
            if expRemoved_total != 0:
                remProportional_def = expRemoved_def / expRemoved_total
                remProportional_att = expRemoved_att / expRemoved_total
            else:
                remProportional_def = 0
                remProportional_att = 0

            curve_total = mdp1.findCurve(expRemoved_total, state, approach, TIME_STEP)
            action_joint = curve_total.threshold

            expLostAssets, totDebrisLevel, expRemoved, targetThreshold = mdp1.transitionOfStates(state, action_joint, approach, TIME_STEP)
            # print expRemoved_total, expRemoved
            expRemoved_def = expRemoved * remProportional_def
            expRemoved_att = expRemoved * remProportional_att

            state_next = mdp1.state(totDebrisLevel, yr + TIME_STEP)
            state_nextApprox = mdp1.state(mdp.approximateState(state_next), state_next.year)

            if infiniteRewSum:
                # ======= last reward is infinite sum of future discounted rewards with constant values ==============
                # !!!!!! this type of reward not defined for the attacker yet **********************************************
                if (yr + timeHorizon % TIME_STEP) == finalYear:
                    averagedEndReward = 0
                    indx = 0
                    for thr in thresholdCurves:
                        if thr.threshold == action:
                            break
                        indx += 1

                    size = 0
                    for j in range(95, 100):
                        averagedEndReward += -(share_IA * thresholdCurves[indx].lostAssets[j] + ratioC_L_C_R * thresholdCurves[indx].removed[j])
                        size += 1

                    averagedEndReward = averagedEndReward / size
                    rewardInf = averagedEndReward * (1 / (1 - rewardDiscountGamma))
                    reward = mdp1.getReward_multiAgent(expLostAssets, expRemoved_def) + rewardInf

                else:
                    reward = mdp1.getReward_multiAgent(expLostAssets, expRemoved_def)
                # ================================================================================
            else:
                reward = mdp1.getReward_multiAgent(expLostAssets, expRemoved_def)
                reward_att = mdp1.getReward_multiAgent_att(expLostAssets, expRemoved_att)


            discReward.append(pow(rewardDiscountGamma, i - 1) * reward)
            discReward_att.append(pow(rewardDiscountGamma, i - 1) * reward_att)

            # Q learning ++++++++++++++++++++++++++++++++++++++++++
            qLearning.update_Q(Q_table, alpha, rewardDiscountGamma, stateApprox, state_nextApprox, action, reward)

            qLearning.update_Q(Q_table_att, alpha, rewardDiscountGamma, stateApprox, state_nextApprox, action_att, reward_att)
            # fn approx
            # weights, td_error = qLearning.update_weights(weights, alpha, rewardDiscountGamma, state, action, state_next, reward)

            # weights_att, td_error_att = qLearning.update_weights(weights_att, alpha, rewardDiscountGamma, state, action_att, state_next, reward_att)

            # td_errorList.append(math.sqrt(td_error ** 2))
            # td_errorList_att.append(math.sqrt(td_error_att ** 2))

            state = state_next
            if iteration == totIter - 1:
                finalStrategy.append(action)
                finalStrategy_att.append(action_att)
                finalDebrisLevel.append(totDebrisLevel)
                immReward.append(reward)
                immReward_att.append(reward_att)
            i += 1
            yr = (TIME_STEP * i) + 2017

        discRewardEvol.append(sum(discReward))
    print
    print "++++" * 20
    print "share: ", share_IA, "ratio: ", ratioC_L_C_R
    print "Q-learning params, alpha :", alpha, " epsilon: ", epsilon, " gamma: ", rewardDiscountGamma
    print "step in debris size: ", debrisLevelStep
    print "DEF: Learned sequence of actions: ", finalStrategy
    print "ATT: Learned sequence of actions: ", finalStrategy_att
    # print "Total number of explored states ", no_states
    # print weights
    print "DEF: discounted reward is :", sum(discReward)
    print "ATT: discounted reward is :", sum(discReward_att)
    # print np.cumsum(discReward)[::1]
    print "DEF: imm reward is :", sum(immReward)
    print "ATT: imm att reward is :", sum(immReward_att)
    print
    print "Both tot reward is: ", sum(immReward) + sum(immReward_att)
    print
    print "DEF: total error is: ", sum(td_errorList)
    print "ATT: total error is: ", sum(td_errorList_att)
    print "++++" * 20
    print

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    return finalStrategy


def showFinalStrategy(finalStrategy):
    years = range(2017, 2017 + 100, TIME_STEP)
    thresholdCurves = getThresholdCurves()
    # thresholds = [1000, 2000, 3000, 4000, 5000, 6000]
    thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    # thresholds = [1000, 4000, 8000, 12000]
    thresholdsNames = ["learned", 1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    strategies = []
    strategies.append(finalStrategy)
    for thr in thresholds:
        strat = [thr] * 52
        strategies.append(strat)
    mdp1 = mdp.MDP(thresholdCurves)

    # actionsLabels = ["above 1000", "above 2000", "above 3000", "above 4000", "above 5000", "above 6000", "above 8000", "above 9000", "above 10000", "no removal"]
    actionsLabels = ["Q-learned strat", "above 1000", "above 2000", "above 3000", "above 4000", "above 5000", "above 6000", "above 8000", "above 9000", "above 10000", "no removal"]
    # actionsLabels = ["above 1000", "above 4000", "above 8000", "no removal"]
    # actionsLabels = ["$\gamma$ = 0.99", "$\gamma$ = 0.975", "$\gamma$ = 0.95", "$\gamma$ = 0.9", "$\gamma$ = 0.8"]
    # actionsLabels = ["C_L/C_R = 0.05", "C_L/C_R = 0.1", "C_L/C_R = 0.2", "C_L/C_R = 0.3", "C_L/C_R = 0.4", "C_L/C_R = 0.5"]
    # colorOfExper = ['g--', 'g', 'b', 'k', 'y', 'm', 'c', 'k', 'b', 'g', 'r', 'y']
    colorOfExper = ['g', 'g--', 'b--', 'k--', 'y--', 'm--', 'c--', 'k--', 'b--', 'g--', 'r--', 'y']
    file1 = fileN
    ff = open(file1, 'w')


    ix = 0
    for thr in range(0, len(strategies)):

        finalStrategy = strategies[thr]
        if thr == 0:
            debris0 = mdp1.thresholdCurves[2].totDebris[0]
        else:
            debris0 = mdp1.thresholdCurves[thr - 1].totDebris[0]

        totDebrisLevel = debris0
        state0 = mdp1.state(totDebrisLevel, 2017)
        state1 = state0
        immReward = []
        discReward = []
        finalDebrisLevel = []
        finalDebrisLevel.append(totDebrisLevel)
        totRemoved = []
        totLost = []
        i = 0
        yr = 2017
        while (yr - START_YEAR) < 100:
            action = finalStrategy[i]

            expLostAssets, totDebrisLevel, expRemoved, targetThreshold = mdp1.transitionOfStates(state1, action, approach, TIME_STEP)
            totRemoved.append(expRemoved)
            totLost.append(expLostAssets)

            if infiniteRewSum:
                # ======= last reward is infinite sum of future discounted rewards with constant values ==============
                if (yr + timeHorizon % TIME_STEP) == finalYear:
                    averagedEndReward = 0
                    indx = 0
                    for thre in thresholdCurves:
                        if thre.threshold == action:
                            break
                        indx += 1

                    size = 0
                    for j in range(95, 100):
                        averagedEndReward += -(thresholdCurves[indx].lostAssets[j] * C_L + thresholdCurves[indx].removed[j] * C_R)
                        size += 1

                    averagedEndReward = averagedEndReward / size
                    rewardInf = averagedEndReward * (1 / (1 - rewardDiscountGamma))
                    reward = mdp1.getReward(expLostAssets, expRemoved) + rewardInf

                else:
                    reward = mdp1.getReward(expLostAssets, expRemoved)
                    # reward = mdp1.getReward_multiAgent(expLostAssets, expRemoved)
                # ================================================================================
            else:
                reward = mdp1.getReward(expLostAssets, expRemoved)
                # reward = mdp1.getReward_multiAgent(expLostAssets, expRemoved)
            immReward.append(reward)
            # discReward.append(round((pow(g, i - 1) * reward), 4))
            discReward.append(round((pow(rewardDiscountGamma, i - 1) * reward), 4))
            finalDebrisLevel.append(totDebrisLevel)
            i += 1
            # print i, yr
            yr = (TIME_STEP * i) + 2017
            # print yr
            state1 = mdp1.state(totDebrisLevel, yr)
        print
        print "Sum of discounted reward for threshold ", thresholdsNames[ix], " is ", sum(discReward)
        print "Sum of immediate reward for threshold is ", sum(immReward)
        print "Discounted reward for threshold ", thresholdsNames[ix], " is ", np.cumsum(discReward)[::1]
        print "total removed: ", sum(totRemoved)
        print "total lost: ", sum(totLost)
        print

        ff.write("%s\n" % finalStrategy)
        ff.write("%s\n" % sum(discReward))
        ff.write("%s\n" % np.cumsum(discReward))
        ff.write("\n")

        totRew = round(sum(immReward), 3)

        if ix == 0:
            plt.plot(years, finalDebrisLevel[0:-1], colorOfExper[ix], label=actionsLabels[ix], linewidth=3)
            # plt.plot(years, finalDebrisLevel, colorOfExper[ix], label=actionsLabels[ix], linewidth=3)
        else:
            plt.plot(years, finalDebrisLevel[0:-1], colorOfExper[ix], label=actionsLabels[ix], linewidth=1.5)
            # plt.plot(years, finalDebrisLevel, colorOfExper[ix], label=actionsLabels[ix], linewidth=1.5)
        ix += 1
        # inxG += 1
    ff.close()
    pylab.xlim(2015, 2120)
    # pylab.ylim(-1.2, 0.2)
    # plt.ylabel('discounted reward', fontsize=20)
    plt.ylabel('number of objects', fontsize=20)
    # plt.ylabel('removed', fontsize=20)
    # plt.ylabel('lost assets', fontsize=20)
    plt.xlabel('year', fontsize=20)
    # plt.title('Objects number evolution', fontsize=20)
    # plt.title('Objects number evolution - $C_R/C_L = 0.5$', fontsize=20)
    plt.title('Discounted reward - C_R/C_L = 0.3, $\gamma$ = 0.95', fontsize=20)
    # plt.title('Discounted reward - above 1000, time step 2 years', fontsize=17)
    # plt.title('Immediate reward evolution', fontsize=20)
    # plt.title('Immediate reward evolution - threshold = 3000', fontsize=20)
    # plt.title('Expected # of removed', fontsize=20)
    # plt.title('Expected # of removed', fontsize=20)
    # plt.title('Expected # of lost assets', fontsize=20)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()


def findOptimalSequence(attackerFixed):
    years = range(2017, 2017 + 101, TIME_STEP)
    thresholdCurves = getThresholdCurves()
    # thresholds = [1000, 2000, 3000, 4000, 5000, 6000]
    thresholds = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 9000, 10000, 12000]
    mdp1 = mdp.MDP(thresholdCurves)

    startAction = [3000]
    yr = 0
    # seq = []
    seqAll = []

    def getSeq(seq, action, yr):
        if yr < 100:
            seq.append(action)
            indx = 0
            for thr in thresholds:
                if thr == action:
                    break
                indx += 1
            yr += TIME_STEP
            getSeq(seq[:], action, yr)
            if action != 12000:
                getSeq(seq[:], thresholds[indx + 1], yr)
            if action != 1000:
                getSeq(seq[:], thresholds[indx - 1], yr)
        else:
            seqAll.append(seq)

        return seqAll

    seq = []
    stratsAll = []
    for thrs in thresholds:
        stratsAll.extend(getSeq([], thrs, yr))

    strats = stratsAll
    allDiscRews = []
    allRews = []
    allRews_att = []
    allRews_both = []
    action_att = attackerFixed
    # showFinalStrategy(strats[100])
    for strat in strats:
        finalStrategy = strat
        debris0 = mdp1.thresholdCurves[2].totDebris[0]

        totDebrisLevel = debris0
        state0 = mdp1.state(totDebrisLevel, 2017)
        state1 = state0
        immReward = []
        immReward_att = []
        discReward = []
        finalDebrisLevel = []
        finalDebrisLevel.append(totDebrisLevel)
        i = 0
        yr = 2017
        while (yr - START_YEAR) < 100:

            action = finalStrategy[i]

            # adding attacker +++++++++++++++++++++++++++++++++++++++++++++++++++++++
            expLostAssets_def, totDebrisLevel_def, expRemoved_def, targetThreshold = mdp1.transitionOfStates(state1, action, approach, TIME_STEP)
            expLostAssets_att, totDebrisLevel_att, expRemoved_att, targetThreshold_att = mdp1.transitionOfStates(state1, action_att, approach, TIME_STEP)

            expRemoved_total = expRemoved_def + expRemoved_att

            if expRemoved_total != 0:
                remProportional_def = expRemoved_def / expRemoved_total
                remProportional_att = expRemoved_att / expRemoved_total
            else:
                remProportional_def = 0
                remProportional_att = 0

            curve_total = mdp1.findCurve(expRemoved_total, state1, approach, TIME_STEP)
            action_joint = curve_total.threshold

            expLostAssets, totDebrisLevel, expRemoved, targetThreshold = mdp1.transitionOfStates(state1, action_joint, approach, TIME_STEP)
            # print expRemoved_total, expRemoved
            expRemoved_def = expRemoved * remProportional_def
            expRemoved_att = expRemoved * remProportional_att

            # adding attacker +++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if infiniteRewSum:
                # ======= last reward is infinite sum of future discounted rewards with constant values ==============
                if (yr + timeHorizon % TIME_STEP) == finalYear:
                    averagedEndReward = 0
                    indx = 0
                    for thre in thresholdCurves:
                        if thre.threshold == action:
                            break
                        indx += 1

                    size = 0
                    for j in range(95, 100):
                        averagedEndReward += -(thresholdCurves[indx].lostAssets[j] * C_L + thresholdCurves[indx].removed[j] * C_R)
                        size += 1

                    averagedEndReward = averagedEndReward / size
                    rewardInf = averagedEndReward * (1 / (1 - rewardDiscountGamma))
                    reward = mdp1.getReward(expLostAssets, expRemoved) + rewardInf

                else:
                    reward = mdp1.getReward(expLostAssets, expRemoved)
                # ================================================================================
            else:
                # reward = mdp1.getReward(expLostAssets, expRemoved)

                # adding attacker +++++++++++++++++++++++++++++++++++++++++++++++++++++++
                reward = mdp1.getReward_multiAgent(expLostAssets, expRemoved_def)
                reward_att = mdp1.getReward_multiAgent_att(expLostAssets, expRemoved_att)
                # adding attacker +++++++++++++++++++++++++++++++++++++++++++++++++++++++

            immReward.append(reward)
            immReward_att.append(reward_att)
            # discReward.append(round((pow(g, i - 1) * reward), 4))
            discReward.append(round((pow(rewardDiscountGamma, i - 1) * reward), 4))
            finalDebrisLevel.append(totDebrisLevel)

            i += 1
            # print i, yr
            yr = (TIME_STEP * i) + 2017
            # print yr
            state1 = mdp1.state(totDebrisLevel, yr)
        allDiscRews.append(sum(discReward))
        allRews.append(sum(immReward))
        allRews_att.append(sum(immReward_att))
        allRews_both.append(sum(immReward) + sum(immReward_att))

    # print max(allDiscRews)
    # maxRewDef = max(allRews)
    maxRewEnv = max(allRews_both)
    BR_defender = np.argmax(allRews)
    BR_env = np.argmax(allRews_both)

    # maxRewsDef = [strats[i] for i, j in enumerate(allRews) if j == maxRewDef]
    # maxRewsEnv = [strats[i] for i, j in enumerate(allRews_both) if j == maxRewEnv]
    maxRewsEnv = [i for i, j in enumerate(allRews_both) if j == maxRewEnv]
    maxRewsEnv_maxRewDefPom = [allRews[i] for i in maxRewsEnv]
    maxRewsEnv_maxRewDef = maxRewsEnv[np.argmax(maxRewsEnv_maxRewDefPom)]
    # BR_env = np.argmax([i for i in )
    # print maxRewsDef
    # print "====" *20
    print "ratio: ", ratioC_L_C_R, "opponent fixed: ", attackerFixed
    # print "def reward is: ", max(allRews)
    print "def reward is: ", allRews[BR_defender]
    # print "att reward is: ", (allRews_att[maxIndex])
    print "att reward is: ", allRews_att[BR_defender]
    # print "tot reward both is: ", (max(allRews) + allRews_att[maxIndex])
    print "tot reward both is: ", (allRews[BR_defender] + allRews_att[BR_defender])
    print strats[BR_defender]
    print
    # print "def reward is: ", allRews[BR_env]
    # print "att reward is: ", allRews_att[BR_env]
    # # print "env reward both is: ", max(allRews_both)
    # print "env reward both is: ", (allRews[BR_env] + allRews_att[BR_env])
    # print strats[BR_env]
    print "def reward is: ", allRews[maxRewsEnv_maxRewDef]
    print "att reward is: ", allRews_att[maxRewsEnv_maxRewDef]
    # print "env reward both is: ", max(allRews_both)
    print "env reward both is: ", (allRews[maxRewsEnv_maxRewDef] + allRews_att[maxRewsEnv_maxRewDef])
    print strats[maxRewsEnv_maxRewDef]
    # print "====" *20
