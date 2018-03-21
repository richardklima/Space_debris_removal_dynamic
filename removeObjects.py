import copy
import logging
from operator import itemgetter
import math
import time

START_YEAR = 2017


def updateCollisionTuples(tupleOfCollisions, removedObjects, satellites, expectedDebrisThresholdForRem, debris, year):
    newListOfCommonRiskPlanets = []
    remStats = []
    for pln in tupleOfCollisions[:]:
        if pln[0] in removedObjects:
            tupleOfCollisions.remove(pln)
        elif pln[2] in removedObjects:
            tupleOfCollisions.remove(pln)

    tupleOfCollisionasFiltered = [tupl for tupl in tupleOfCollisions if (tupl[4] * tupl[5]) > expectedDebrisThresholdForRem]
    for couple in tupleOfCollisionasFiltered:
        risk = couple[4]
        numOfDebris = couple[5]
        expNoDebris = risk * numOfDebris

        # we do not remove important assets
        if any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[0])) and any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[2])):
            continue

        elif any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[0])) and not any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[2])):
            newListOfCommonRiskPlanets.append([couple[2], couple[3], expNoDebris])

            remStats.append([year - START_YEAR, couple[0], couple[1], couple[2], couple[3], couple[4], couple[5], 1, 0, 0, 0])

        elif not any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[0])) and any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[2])):
            newListOfCommonRiskPlanets.append([couple[0], couple[1], expNoDebris])

            remStats.append([year - START_YEAR, couple[0], couple[1], couple[2], couple[3], couple[4], couple[5], 1, 0, 0, 0])

        # if none is important asset we remove the one with lower mass
        elif not any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[0])) and not any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[2])):

            pln1Mass = couple[1]
            pln2Mass = couple[3]

            if pln1Mass < pln2Mass:
                newListOfCommonRiskPlanets.append([couple[0], couple[1], expNoDebris])
            else:
                newListOfCommonRiskPlanets.append([couple[2], couple[3], expNoDebris])

            remStats.append([year - START_YEAR, couple[0], couple[1], couple[2], couple[3], couple[4], couple[5], 0, 0, 0, 0])

    newListOfCommonRiskPlanets.sort(key=lambda k: (k[2]), reverse=True)

    return newListOfCommonRiskPlanets, tupleOfCollisionasFiltered, remStats


def updateCollisionTuples_copy(tupleOfCollisions, removedObjects, satellites, debris):
    newListOfCommonRiskPlanets = []
    for pln in tupleOfCollisions:
        if pln[0] in removedObjects:
            tupleOfCollisions.remove(pln)
        elif pln[1] in removedObjects:
            tupleOfCollisions.remove(pln)

    for couple in tupleOfCollisions:
        risk = couple[2]
        numOfDebris = couple[3]
        expNoDebris = risk * numOfDebris
        # we list only objects which are not important assets to be removed
        if any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[0])):
            pass
        else:
            if any(deb for deb in debris if (deb.name.strip() == couple[0])):
                pln1Mass = next(deb.mass for deb in debris if (deb.name.strip() == couple[0]))

                for planet in newListOfCommonRiskPlanets:
                    if couple[0] == planet[0]:
                        planet[1] += expNoDebris
                        break
                else:
                    newListOfCommonRiskPlanets.append([couple[0], expNoDebris, pln1Mass])

        if any(sat for sat in satellites if (sat.importantAsset and sat.name == couple[1])):
            pass
        else:
            if any(deb for deb in debris if (deb.name.strip() == couple[1])):
                pln2Mass = next(deb.mass for deb in debris if (deb.name.strip() == couple[1]))
                for planet in newListOfCommonRiskPlanets:
                    if couple[1] == planet[0]:
                        planet[1] += expNoDebris
                        break
                else:
                    newListOfCommonRiskPlanets.append([couple[1], expNoDebris, pln2Mass])

    newListOfCommonRiskPlanets.sort(key=lambda k: (k[1], -k[2]), reverse=True)

    return newListOfCommonRiskPlanets, tupleOfCollisions


def getTotalCostOfImportantAssets(satellites, agentList):

    costOfImpAssets = []
    for agent in agentList:
        costOfImpAssets.append(sum(sat.cost for sat in satellites if (sat.importantAsset and sat.owner in agent)))

    return costOfImpAssets


def removeRiskyObjects_dynamic_withOwnRisks(satellites, collisionRiskCouplesVirtual, agentList, debris, objectRemovalOwn, objectRemovalCommon):

    thresholdForCostOfLosingAsset = 5e7
    costOfImpAssets = getTotalCostOfImportantAssets(satellites, agentList)

    objectsThreatingImportantAssets = map(list, [[]] * len(agentList))
    listOfRemovedObjects = []
    expectedNumberOfDebris = []

    # We save objects which threat important assets ######################################
    for couple in collisionRiskCouplesVirtual:
        pln1 = couple[0]
        pln2 = couple[1]
        risk = couple[2]
        numOfDebris = couple[3]
        expectedNumberOfDebris.append(risk * numOfDebris)
        for agentIndex in range(0, len(agentList)):
            if any(sat for sat in satellites if (sat.importantAsset and sat.name == pln1 and sat.owner in agentList[agentIndex])):
                impAssetThreatened = next(sat for sat in satellites if (sat.importantAsset and sat.name == pln1 and sat.owner in agentList[agentIndex]))
                if any(obj for obj in objectsThreatingImportantAssets[agentIndex] if obj[0] == pln2):

                    objPom = next(obj for obj in objectsThreatingImportantAssets[agentIndex] if obj[0] == pln2)
                    objPom[1] += risk
                else:
                    objectsThreatingImportantAssets[agentIndex].append([pln2, risk, pln1, impAssetThreatened.cost])

            if any(sat for sat in satellites if (sat.importantAsset and sat.name == pln2 and sat.owner in agentList[agentIndex])):
                impAssetThreatened = next(sat for sat in satellites if (sat.importantAsset and sat.name == pln2 and sat.owner in agentList[agentIndex]))
                if any(obj for obj in objectsThreatingImportantAssets[agentIndex] if obj[0] == pln1):
                    objPom = next(obj for obj in objectsThreatingImportantAssets[agentIndex] if obj[0] == pln1)
                    objPom[1] += risk
                else:
                    objectsThreatingImportantAssets[agentIndex].append([pln1, risk, pln2, impAssetThreatened.cost])

    levelOfRisk = [0] * 4
    for agentIndex in range(0, len(agentList)):
        levelOfRisk[agentIndex] = sum([pair[1] for pair in objectsThreatingImportantAssets[agentIndex]])

    levelOfCommonRisk = sum([pair[2] for pair in collisionRiskCouplesVirtual])

    expectedNumberOfDebrisRatio = sum(expectedNumberOfDebris) / len(debris)
    threatCostForPlayers = [cost * expectedNumberOfDebrisRatio for cost in costOfImpAssets]

    logger = logging.getLogger(__name__)

    # We do not remove any important assets ##################################
    for threatsToAgent in objectsThreatingImportantAssets:
        for asset in threatsToAgent:
            if any(sat for sat in satellites if (sat.importantAsset and sat.name == asset[0])):
                threatsToAgent.remove(asset)

    # need to use deep copy because of cloning list of lists of objects
    pomThreats = []
    pomThreats = copy.deepcopy(objectsThreatingImportantAssets)

    # removing objecting threatening important assets #################################################

    # remove objects until n objects are removed
    listOfRemoved = []
    iterator = [0] * len(agentList)
    if objectRemovalOwn == 1:
        for agentIndex in range(0, len(agentList)):

            if len(objectsThreatingImportantAssets[agentIndex]) == 0:
                continue
            highestThreat = max(objectsThreatingImportantAssets[agentIndex], key=itemgetter(1))
            riskOfCollision = highestThreat[1]
            costOfAsset = highestThreat[3]

            while len(objectsThreatingImportantAssets[agentIndex]) != 0 and (riskOfCollision * costOfAsset) > thresholdForCostOfLosingAsset:
                highestThreat = max(objectsThreatingImportantAssets[agentIndex], key=itemgetter(1))
                riskOfCollision = highestThreat[1]
                threatenedAsset = next(sat for sat in satellites if sat.name == highestThreat[2])
                costOfAsset = highestThreat[3]
                objectsToRemove = [threat for threat in objectsThreatingImportantAssets[agentIndex] if threat[0] == highestThreat[0]]
                for obj in objectsToRemove:
                    objectsThreatingImportantAssets[agentIndex].remove(obj)

                if any(deb for deb in debris if (deb.name.strip() == highestThreat[0])):
                    debToRemove = next(deb for deb in debris if (deb.name.strip() == highestThreat[0]))
                    logger.info(agentList[agentIndex] + " removing {} {}".format(debToRemove.name, highestThreat[1]) + " threatened to {}".format(highestThreat[2]))
                    debris.remove(debToRemove)
                    listOfRemovedObjects.append([agentList[agentIndex], "o", debToRemove.name, 0, 0, 0, 0, debToRemove.mass])
                    listOfRemoved.append(debToRemove.name)
                    iterator[agentIndex] += 1

                if any(sat for sat in satellites if sat.name == highestThreat[0]):
                    objPom = next(sat for sat in satellites if sat.name == highestThreat[0])
                    satellites.remove(objPom)

    listOfCommonRiskPlanets = []
    listOfCommonRiskPlanets, collisionRiskCouplesVirtual = updateCollisionTuples(collisionRiskCouplesVirtual, listOfRemoved)

# REMOVING COMMON OBJECTS ###############################################
    if objectRemovalCommon == 1:
        maxThreatIndex = threatCostForPlayers.index(max(threatCostForPlayers))

        for agentIndex in range(0, len(agentList)):
            if agentIndex == maxThreatIndex:

                # We remove object which has the highest risk*expected_number_of_debris value
                highestThreat = max(listOfCommonRiskPlanets, key=itemgetter(1))
                expNoDebris = highestThreat[1]
                expectedNoOfDebrisRatio = expNoDebris / len(debris)
                threatReduced = costOfImpAssets[agentIndex] * expectedNoOfDebrisRatio

                listOfCommonRiskPlanets.remove(highestThreat)
                # we do not remove important assets
                if any(sat for sat in satellites if (sat.importantAsset and sat.name == highestThreat[0])):
                    continue

                if any(deb for deb in debris if (deb.name.strip() == highestThreat[0])):
                    debToRemove = next(deb for deb in debris if (deb.name.strip() == highestThreat[0]))

                    logger.info(agentList[agentIndex] + " removing common {} {}".format(debToRemove.name, highestThreat[1]))
                    debris.remove(debToRemove)
                    listOfRemovedObjects.append([agentList[agentIndex], "c", debToRemove.name, 0, 0, 0, 0, debToRemove.mass])
                    listOfCommonRiskPlanets, collisionRiskCouplesVirtual = updateCollisionTuples(collisionRiskCouplesVirtual, debToRemove.name)
                    iterator[agentIndex] += 1
#
    for agentIndex in range(0, len(agentList)):
        print "agent ", agentList[agentIndex], " remove 0: ", threatCostForPlayers[agentIndex] / 1e9, ", remove 1: ", (threatCostForPlayers[agentIndex] - threatReduced) / 1e9

    for remObj in listOfRemovedObjects:
        for agentIndex in range(0, len(agentList)):
            for threat in pomThreats[agentIndex]:
                if remObj[2] == threat[0]:
                    remObj[agentIndex + 3] += threat[1]

    return listOfRemovedObjects, collisionRiskCouplesVirtual


def removeRiskyObjects_dynamic(satellites, collisionRiskCouplesVirtual, agentList, debris, expectedDebrisThresholdForRem, year):
    thresholdForCostOfLosingAsset = 5e7
    costOfImpAssets = getTotalCostOfImportantAssets(satellites, agentList)

    expectedNumberOfDebris = []

    for couple in collisionRiskCouplesVirtual:
        pln1 = couple[0]
        pln2 = couple[2]
        risk = couple[4]
        numOfDebris = couple[5]
        expectedNumberOfDebris.append(risk * numOfDebris)

    # orders objects in collison tuples that the first one is the lighter one so we can remove it

    toBeRemovedIndx = [i for i in range(len(expectedNumberOfDebris)) if expectedNumberOfDebris[i] > 100]

    expectedNumberOfDebrisRatio = sum(expectedNumberOfDebris) / len(debris)
    threatCostForPlayers = [cost * expectedNumberOfDebrisRatio for cost in costOfImpAssets]

    logger = logging.getLogger(__name__)

    listOfRemoved = []
    listOfCommonRiskPlanets = []
    remStats = []
    time1 = time.time()
    listOfCommonRiskPlanets, collisionRiskCouplesVirtual, remStats = updateCollisionTuples(collisionRiskCouplesVirtual, listOfRemoved, satellites, expectedDebrisThresholdForRem, debris, year)
    time2 = time.time()
    logger.info('Time in removal: ' + str(time2 - time1))

# REMOVING COMMON OBJECTS ###############################################
    for objToRem in listOfCommonRiskPlanets:
        maxThreatIndex = threatCostForPlayers.index(max(threatCostForPlayers))

        for agentIndex in range(0, len(agentList)):
            if agentIndex == maxThreatIndex:

                # We remove object which has the highest risk*expected_number_of_debris value
                if any(deb for deb in debris if (deb.name.strip() == objToRem[0] and deb.mass == objToRem[1])):
                    debToRemove = next(deb for deb in debris if (deb.name.strip() == objToRem[0] and deb.mass == objToRem[1]))
                else:
                    break

                logger.info(agentList[agentIndex] + " removing common {} {}".format(debToRemove.name, objToRem[1], objToRem[2]))
                debris.remove(debToRemove)
    return remStats
