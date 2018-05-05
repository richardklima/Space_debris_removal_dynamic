import time
import math
import logging
import PyKEP as kep
import numpy as np
import re
import copy
import launch
from itertools import combinations
import random
from breakup import breakup
from mpi4py import MPI
from cubeAlone import cube, collision_prob, setradii, setmass, update
from removeObjects import removeRiskyObjects_dynamic

START_YEAR = 2017
MASS_YEAR_2000 = 2 * 1E5
spacecraft_classes = []


scenario = 1
if scenario == 1:
    # CONSERVATIVE
    TOTAL_MASS_PER_YEAR_ALPHA = 0
    spacecraft_classes.append(launch.SpacecraftClass("large", 2020, 60, 40E6, 700E6, 500, 5000, 10, 20, 10, 25))
    spacecraft_classes.append(launch.SpacecraftClass("medium", 2060, 40, 15E6, 40E6, 100, 500, 1, 5, 7, 20))
    spacecraft_classes.append(launch.SpacecraftClass("small", 2150, 50, 1E6, 15E6, 10, 100, 0.5, 2, 1, 7))
    spacecraft_classes.append(launch.SpacecraftClass("ultra_small", 2200, 50, 2E3, 1E6, 0.1, 10, 0.5, 1, 0.5, 2))
elif scenario == 2:
    # MODERATE
    TOTAL_MASS_PER_YEAR_ALPHA = 1E-4
    spacecraft_classes.append(launch.SpacecraftClass("large", 1970, 60, 40E6, 700E6, 500, 5000, 10, 20, 10, 25))
    spacecraft_classes.append(launch.SpacecraftClass("medium", 2060, 50, 15E6, 40E6, 100, 500, 1, 5, 7, 20))
    spacecraft_classes.append(launch.SpacecraftClass("small", 2090, 30, 1E6, 15E6, 10, 100, 0.5, 2, 1, 7))
    spacecraft_classes.append(launch.SpacecraftClass("ultra_small", 2150, 35, 2E3, 1E6, 0.1, 10, 0.5, 1, 0.5, 2))
elif scenario == 3:
    # AGGRESSIVE
    TOTAL_MASS_PER_YEAR_ALPHA = -1E-5
    spacecraft_classes.append(launch.SpacecraftClass("large", 1975, 60, 40E6, 700E6, 500, 5000, 10, 20, 10, 25))
    spacecraft_classes.append(launch.SpacecraftClass("medium", 2040, 30, 15E6, 40E6, 100, 500, 1, 5, 7, 20))
    spacecraft_classes.append(launch.SpacecraftClass("small", 2100, 40, 1E6, 15E6, 10, 100, 0.5, 2, 1, 7))
    spacecraft_classes.append(launch.SpacecraftClass("ultra_small", 2150, 50, 2E3, 1E6, 0.1, 10, 0.5, 1, 0.5, 2))

# MPI identification
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
proc_name = MPI.Get_processor_name()
mpi_instance_id = "proc_" + '{0:03d}'.format(rank) + "_of_" + '{0:03d}'.format(comm.size) + "_on_" + proc_name

# Create logger.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create file handler to log debug messages.
fh = logging.FileHandler(mpi_instance_id + '.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))

# Create console handler to display info messages.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('[{}] %(message)s'.format(mpi_instance_id)))

# Add new handlers to the logger.
logger.addHandler(fh)
logger.addHandler(ch)


obeyMitigation = [0] * 4

file_inputParameters = "parameters_simulator.txt"
fParam = open(file_inputParameters)
numOfExperiments = int(re.sub("[^0-9]", "", fParam.next()))

# length of experiment in years, standard setting 100 years
timeHorizon = int(re.sub("[^0-9]", "", fParam.next()))

objectRemovalOwn = int(re.sub("[^0-9]", "", fParam.next()))
objectRemovalCommon = int(re.sub("[^0-9]", "", fParam.next()))
# debrisInjected = int(re.sub("[^0-9]", "", fParam.next()))
expectedDebrisThresholdForRem = int(re.sub("[^0-9]", "", fParam.next()))

fParam.close()

for qq in range(1, numOfExperiments + 1):
    satcat = launch.loadData_satcat()
    debris = launch.loadData_tle()

    # list of all EU states and ESA and its bodies
    EU_list = ["EU", "ASRA", "BEL", "CZCH", "DEN", "ESA", "ESRO", "EST", "EUME", "EUTE", "FGER", "FR", "FRIT", "GER", "GREC", "HUN", "IT", "LTU", "LUXE", "NETH", "NOR", "POL", "POR", "SPN", "SWED", "SWTZ", "UK"]
    agentList = ['CIS', 'US', 'PRC', 'EU']
    agentListWithEU = ['CIS', 'US', 'PRC', EU_list]
#  filtering only objects in LEO
    debris_LEO = []
    ep = kep.epoch((START_YEAR - 2000.) * 365.25)
    for p in debris:
        try:
            oe = p.osculating_elements(ep)
            if oe[0] * (1 - oe[1]) < 6378000.0 + 2000000:
                debris_LEO.append(p)
        except:
            pass
    debris = debris_LEO
    debris_pom = copy.deepcopy(debris)
    satcat_pom = copy.deepcopy(satcat)
    # contains all satellites i.e. not debris, not R/B, and not decayed at the beginning of run
    satellites = launch.getSatellites(satcat_pom, debris_pom, spacecraft_classes, START_YEAR, 1)
    agentImportantAssetsAtStart = []
    for agentIndex in range(0, len(agentList)):
        impAssets = [asset for asset in satellites if (asset.importantAsset and asset.owner in agentListWithEU[agentIndex])]
        agentImportantAssetsAtStart.append(len(impAssets))
        impAssets = []

    satellitesDecayedByHand2 = 0
    satellitesDecayedByHand3 = 0
    decaysImmediatelly = 0
    numberOfLaunchesPerMonth = []
    launchedPerYear = []
    activeSatelitesNo = []
    activeLarge = []
    activeMedium = []
    activeSmall = []
    activeUltraSmall = []
    launchedLarge = []
    launchedMedium = []
    launchedSmall = []
    launchedUltraSmall = []
    launchedLargePerMonth = []
    launchedMediumPerMonth = []
    launchedSmallPerMonth = []
    launchedUltraSmallPerMonth = []
    debrisObjectsNo = []
    removedBySizeTotal = []
    removedLaunchedBySGP4 = []
    removedLaunchedBySGP4Yearly = []
    collisionsCatasYearly = []
    collisionsCatasPom = 0
    maneuverCollAvoid = [0] * (timeHorizon + 1)

    # osculating elements distributions
    oe_histograms = []
#    oe_histograms = launch.findHistograms(agentListWithEU)
    oe_histograms = launch.findHistograms()
    yearLaunchesDistributions = oe_histograms[10]

    beginningOfSequence = 2006
    expectedLifespan = 10

    removeHorizon = 2  # how often we remove objects - 1 -> every year, 10 -> every 10 years

    # list of important assets store all active satellites, it is a dicitonary of satcats
    # it us used to check whether there was a collision with important asset or whether we are about to remove important asset
    # it is updated once a year before storing the info data on numbers of IA of each player.
    # The "always-up-to-date" list is "debris" which is propagated every 5 days and contains all objects currently on orbits
    listOfCollided = []

    # setting radius based on cross section area and then deriving its mass
    setradii(debris, satcat)
    setmass(debris)

    decadeYearMultiplier = 1
    numberOfCatastrophicCollisions = 0
    numberOfNonCatastrophicCollisions = 0
    totalNumberOfNewDebris = 0
    totalDebrisEvol = []
    spaceDensities = []

    collisionRiskCouplesVirtual = []
    collisionRiskCouplesReal = []
    agentThreats = map(list, [[]] * len(agentList))
    objectsThreatingImportantAssets = map(list, [[]] * len(agentList))
    agentRisksProb = map(list, [[]] * len(agentList))
    agentYearRiskProb = [0] * len(agentList)
    agentTotalImportantAssets = map(list, [[]] * len(agentList))
    totalImportantAssetsL = []
    totalImportantAssetsM = []
    totalImportantAssetsS = []
    totalImportantAssetsU_S = []
    agentCollisions = []
    for agentIndex in range(0, len(agentList)):
        agentCollisions.append([0] * (timeHorizon + 1))

    collidedOrAvoidedLarge = [0] * (timeHorizon + 1)
    collidedOrAvoidedMedium = [0] * (timeHorizon + 1)
    collidedOrAvoidedSmall = [0] * (timeHorizon + 1)
    collidedOrAvoidedUltraSmall = [0] * (timeHorizon + 1)

    collidedLarge = [0] * (timeHorizon + 1)
    collidedMedium = [0] * (timeHorizon + 1)
    collidedSmall = [0] * (timeHorizon + 1)
    collidedUltraSmall = [0] * (timeHorizon + 1)
    # list of removed objects and who removed it
    listOfRemovedObjects = []
    listOfRemovedObjectsFinal = []
    removedPerYear = [0] * (timeHorizon + 1)

    objectsWithPossitiveProbOfCollisionAll = []
    objectsProbsOfCollisionsAll = []
    commonRisksProb = []
    commonRisksProbYear = 0

    stringOfRemovals = "rem" + str(objectRemovalOwn) + str(objectRemovalCommon)

    if objectRemovalOwn == 1 or objectRemovalCommon == 1:
        virtualRunEnabled = 1
    else:
        virtualRunEnabled = 0

    yearToPlot = 0
    ep = int(math.floor((START_YEAR - 2000) * 365.25))
    month = 0
    virtualRun = 0  # initial setting, if 1 the run is virtual for computing probs of collisions
    # start of removing objects
    yearPom = 2018

    virtualDebris = []
    virtualDebris = debris[:]
    dataSaving = yearPom
    timeHorizonInDays = int(math.ceil((START_YEAR - 2000) * 365.25 + timeHorizon * 365.25))

    # main loop, one step is 5 days
    while ep < timeHorizonInDays or virtualRun == 1:
        t0 = time.time()
        ep += 5
        year = int(math.floor((ep - (START_YEAR - 2000) * 365.25) / 365.25) + START_YEAR)

        # ###############################################################################################################
        # Injecting debris pieces to obtain threat function
        # if year == 2070 and not debrisInjectedAlready:
        #     # we create two large satellites
        #     sampleDebrisPieces = []
        #     while debrisInjected > len(sampleDebrisPieces):
        #         # print 'a'
        #         type_of_satellite_to_launchPom = spacecraft_classes[0]
        #         p1Pom = launch.Satellite.new_sat(type_of_satellite_to_launchPom, agentIndex, oe_histograms, 2017, 1, 0, 1)
        #         p2Pom = launch.Satellite.new_sat(type_of_satellite_to_launchPom, agentIndex, oe_histograms, 2017, 1, 0, 1)
        #         epPom = int(math.floor((START_YEAR - 2000) * 365.25))
        #         twoSats = []
        #         twoSats.append(p1Pom.tle)
        #         twoSats.append(p2Pom.tle)
        #         update(twoSats, epPom)
        #         debris1, debris2 = breakup(epPom, twoSats[0], twoSats[1])
        #         sampleDebrisPieces = debris1 + debris2
        #         # print len(debris1), len(debris2)
        #         # print len(sampleDebrisPieces)
        #     indices = random.sample(range(len(sampleDebrisPieces)), debrisInjected)
        #     # print indices
        #     # print sampleDebrisPieces[indices]
        #     # sampledDebris = [sampleDebrisPieces[i] for i in range(len(sampleDebrisPieces)) if i in indices]
        #     sampledDebris = [sampleDebrisPieces[i] for i in indices]
        #     # print sampledDebris
        #     debris.extend(sampledDebris)
        #     debrisInjectedAlready = True
        # ###############################################################################################################

# *************************************************** END OF VIRTUAL RUN ***************************************************
        if virtualRun == 1 and year % removeHorizon == 0 and year != yearPom and virtualRunEnabled:
            ep = ep - int(removeHorizon * 365.25)
            virtualRun = 0
            yearPom = year
            year = int(math.floor((ep - (START_YEAR - 2000) * 365.25) / 365.25) + START_YEAR)


# *************************************************** Dynamic game - removing risky objects based on risks to important assets ***********************************************************
            listOfRemovedObjects = removeRiskyObjects_dynamic(satellites, collisionRiskCouplesVirtual, agentList, debris, expectedDebrisThresholdForRem, year)

            removedPerYear[year - START_YEAR] += len(listOfRemovedObjects)
            listOfRemovedObjectsFinal.extend(listOfRemovedObjects)
            listOfRemovedObjects = []

            # restarting vector of threats to each agents at the end of virtual run
            agentThreats = map(list, [[]] * len(agentList))
            objectsThreatingImportantAssets = map(list, [[]] * len(agentList))

            collisionRiskCouplesVirtual = []

            logger.info('Switching to real run ' + str(year) + ', total objects: ' + str(len(debris)))
#


# ***************************************************  END OF REAL RUN ***************************************************
        if virtualRun == 0 and year % removeHorizon == 0 and year == yearPom and year != (START_YEAR + timeHorizon) and virtualRunEnabled:
            virtualRun = 1
            yearPom = year
            virtualDebris = debris[:]
            logger.info('Switching to virtual run ' + str(year))
#

# ***************************************************DATA SAVING DURING REAL RUN EVERY YEAR - NO VIRTUAL RUN ***************************************************
        if virtualRun == 0 and dataSaving == year:
            dataSaving += 1
            t1 = time.time()

            len1 = len(satellites)

            len1 = len(satellites)
            satellites, decayedSats = launch.decayOldObjects(satellites, listOfCollided, year, month)
            listOfCollided = []

            numberOfActiveSats = 0
            numberOfActiveLarge = 0
            numberOfActiveMedium = 0
            numberOfActiveSmall = 0
            numberOfActiveUltraSmall = 0

            removedBySize = []
            numberOfRemSats = 0
            numberOfRemLarge = 0
            numberOfRemMedium = 0
            numberOfRemSmall = 0
            numberOfRemUltraSmall = 0

            debrisList = debris[:]
            for deb in debrisList:
                if deb.name.strip() in decayedSats:
                    debris.remove(deb)

                    if deb.name.strip()[-2:] == 'xL':
                        numberOfRemLarge = numberOfRemLarge + 1
                    elif deb.name.strip()[-2:] == 'xM':
                        numberOfRemMedium = numberOfRemMedium + 1
                    elif deb.name.strip()[-2:] == 'xS':
                        numberOfRemSmall = numberOfRemSmall + 1
                    elif deb.name.strip()[-2:] == 'xU':
                        numberOfRemUltraSmall = numberOfRemUltraSmall + 1

                else:
                    activeSat = 0
                    if deb.name.strip()[-2:] == 'xL':
                        numberOfActiveLarge = numberOfActiveLarge + 1
                        activeSat = 1
                    elif deb.name.strip()[-2:] == 'xM':
                        numberOfActiveMedium = numberOfActiveMedium + 1
                        activeSat = 1
                    elif deb.name.strip()[-2:] == 'xS':
                        numberOfActiveSmall = numberOfActiveSmall + 1
                        activeSat = 1
                    elif deb.name.strip()[-2:] == 'xU':
                        numberOfActiveUltraSmall = numberOfActiveUltraSmall + 1
                        activeSat = 1
                    if activeSat == 1:
                        numberOfActiveSats = numberOfActiveSats + 1

            decayedSats = []

            removedBySize.append(numberOfRemLarge)
            removedBySize.append(numberOfRemMedium)
            removedBySize.append(numberOfRemSmall)
            removedBySize.append(numberOfRemUltraSmall)

            removedBySizeTotal.append(removedBySize)

            activeSatelitesNo.append(numberOfActiveSats)

            # impAssets = [asset for asset in satellites if (asset.importantAsset)]

            activeLarge.append(numberOfActiveLarge)
            activeMedium.append(numberOfActiveMedium)
            activeSmall.append(numberOfActiveSmall)
            activeUltraSmall.append(numberOfActiveUltraSmall)
            debrisObjectsNo.append(len(debris) - numberOfActiveSats)
#            print "after: ",len(satellites)

            for agentIndex in range(0, len(agentList)):
                agentRisksProb[agentIndex].append(agentYearRiskProb[agentIndex])

            commonRisksProb.append(commonRisksProbYear)
            agentYearRiskProb = [0] * len(agentList)
            commonRisksProbYear = 0
            noAssetsAgent = [0] * len(agentList)

            impAssets = []
            impAssetsL = []
            impAssetsM = []
            impAssetsS = []
            impAssetsU_S = []
            for agentIndex in range(0, len(agentList)):
                impAssets = [asset for asset in satellites if (asset.importantAsset and asset.owner in agentListWithEU[agentIndex])]
                agentTotalImportantAssets[agentIndex].append(len(impAssets))
                impAssets = []

            impAssetsL = [asset for asset in satellites if (asset.importantAsset and asset.spacecraftClass.spacecraftClass == 'large')]
            impAssetsM = [asset for asset in satellites if (asset.importantAsset and asset.spacecraftClass.spacecraftClass == 'medium')]
            impAssetsS = [asset for asset in satellites if (asset.importantAsset and asset.spacecraftClass.spacecraftClass == 'small')]
            impAssetsU_S = [asset for asset in satellites if (asset.importantAsset and asset.spacecraftClass.spacecraftClass == 'ultra_small')]

            totalImportantAssetsL.append(len(impAssetsL))
            totalImportantAssetsM.append(len(impAssetsM))
            totalImportantAssetsS.append(len(impAssetsS))
            totalImportantAssetsU_S.append(len(impAssetsU_S))
            impAssetsL = []
            impAssetsM = []
            impAssetsS = []
            impAssetsU_S = []
            launch.setImportantAssets(satellites, year, month)

            launchedLarge.append(sum(launchedLargePerMonth))
            launchedMedium.append(sum(launchedMediumPerMonth))
            launchedSmall.append(sum(launchedSmallPerMonth))
            launchedUltraSmall.append(sum(launchedUltraSmallPerMonth))

            launchedLargePerMonth = []
            launchedMediumPerMonth = []
            launchedSmallPerMonth = []
            launchedUltraSmallPerMonth = []

            launchedPerYear.append(sum(numberOfLaunchesPerMonth))
            numberOfLaunchesPerMonth = []

            collisionsCatasYearly.append(collisionsCatasPom)
            collisionsCatasPom = 0

            removedLaunchedBySGP4Yearly.append(sum(removedLaunchedBySGP4))
            removedLaunchedBySGP4 = []

            yearToPlot = int(math.ceil((ep - (START_YEAR - 2000) * 365.25) / 365.25))
            # statistics - output file
            totalDebrisEvol.append(len(debris))
            t4 = time.time()
            # print 'every year', (t4 - t1)
            logger.debug('every year ' + str(t4 - t1))
        # getting month for relaunching sequences
        monthPom = month
        month = math.ceil((((ep - (START_YEAR - 2000) * 365.25) / 365.25 + 0.000001) - math.floor((ep - (START_YEAR - 2000) * 365.25) / 365.25)) * 12)

# ***************************************************Launching new satellites ***************************************************
        if virtualRun == 0 and monthPom != month:

            total_mass_current_year = MASS_YEAR_2000 * (1 + TOTAL_MASS_PER_YEAR_ALPHA * (year - 2000)**2)
            total_mass_per_month = total_mass_current_year / 12

            mass_launched_this_month = 0

            probOfLaunch = [0] * len(spacecraft_classes)
            indx = 0
            # we fix the launch scenario after 100 years, because it's only designed for 100 years
            if year < START_YEAR + 100:
                for sc in spacecraft_classes:
                    probOfLaunch[indx] = sc.getProbability(year, spacecraft_classes)
                    indx = indx + 1
            else:
                for sc in spacecraft_classes:
                    probOfLaunch[indx] = sc.getProbability(START_YEAR + 100, spacecraft_classes)
                    indx = indx + 1

            launchIterator = 1
            launchIteratorLarge = 0
            launchIteratorMedium = 0
            launchIteratorSmall = 0
            launchIteratorUltraSmall = 0
            addingNewPlanet = 0

            # TODO modify and decide who is launching
            while mass_launched_this_month < total_mass_per_month:
                ownershipProportion = []
                ownershipProportion = [float(impAssetNo) / sum(agentImportantAssetsAtStart) for impAssetNo in agentImportantAssetsAtStart]
                agentIndex = np.random.choice(np.arange(0, len(agentList)), p=ownershipProportion)

                type_of_satellite_to_launch = spacecraft_classes[np.random.choice(range(0, len(spacecraft_classes)), p=probOfLaunch)]
                new_satellite = launch.Satellite.new_sat(type_of_satellite_to_launch, agentIndex, oe_histograms, year, month, launchIterator, 1)
                try:
                    a, b = map(np.array, new_satellite.tle.eph(ep))
                except:
                    decaysImmediatelly = decaysImmediatelly + 1

                mass_launched_this_month = mass_launched_this_month + new_satellite.mass
                launchIterator += 1

                satellites.append(new_satellite)
                if new_satellite.decayTime + year < (timeHorizon + START_YEAR):
                    satellitesDecayedByHand2 = satellitesDecayedByHand2 + 1

                if new_satellite.spacecraftClass.spacecraftClass == "large":
                    launchIteratorLarge += 1
                elif new_satellite.spacecraftClass.spacecraftClass == "medium":
                    launchIteratorMedium += 1
                elif new_satellite.spacecraftClass.spacecraftClass == "small":
                    launchIteratorSmall += 1
                elif new_satellite.spacecraftClass.spacecraftClass == "ultra_small":
                    launchIteratorUltraSmall += 1

                debris.append(new_satellite.tle)
                addingNewPlanet = 1

            numberOfLaunchesPerMonth.append(launchIterator - 1)
            launchedLargePerMonth.append(launchIteratorLarge)
            launchedMediumPerMonth.append(launchIteratorMedium)
            launchedSmallPerMonth.append(launchIteratorSmall)
            launchedUltraSmallPerMonth.append(launchIteratorUltraSmall)

        if virtualRun == 1:
            update(virtualDebris, ep)
            volumes = cube(virtualDebris)
        else:
            lengthDebris = len(debris)

            removedLaunched = update(debris, ep)
            listOfCollided.extend(removedLaunched)

            removedLaunchedBySGP4.append(len(removedLaunched))
            satellitesDecayedByHand3 = satellitesDecayedByHand3 + (lengthDebris - len(debris))

            volumes = cube(debris)
        maxp = 0
        for volume in volumes:
            for p1, p2 in combinations(volume, 2):
                if p1.name == p2.name and p1.mass == p2.mass:
                    continue
                p1.name = p1.name.strip()
                p2.name = p2.name.strip()
                if tuple(p1._v) == tuple(p2._v):
                    pass
                else:
                    # probability of collision
                    Pij = collision_prob(p1, p2)
                    # probability of collision over 5 days
                    P = Pij * 5. * kep.DAY2SEC
                    maxp = max(maxp, P)
                    if P > 0:
                        dv = np.linalg.norm(p1._v - p2._v)
                        catastrophRatio = (p2.mass * dv**2) / (2 * p1.mass * 1000)
                        LB = 0.05
                        # if the ratio is smaller than 40 J/g then it is non-catastrophic collision
                        if catastrophRatio > 40:
                            M = p1.mass + p2.mass
                            Lc = np.linspace(LB, 1.)
                            num = int(0.1 * M ** 0.75 * LB ** -1.71)
                        else:
                            num = 0

                        expectedDebris = P * num

                        if virtualRun == 1:
                            collisionRiskCouplesVirtual.append([p1.name.strip(), p1.mass, p2.name.strip(), p2.mass, P, num])
                        elif virtualRun == 0:
                            commonRisksProbYear += P
                            isImpAsset = 0
                            for agentIndex in range(0, len(agentList)):
                                if any(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip() and sat.owner in agentList[agentIndex])):
                                    agentYearRiskProb[agentIndex] += P
                                    isImpAsset += 1

                                if any(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip() and sat.owner in agentList[agentIndex])):
                                    agentYearRiskProb[agentIndex] += P
                                    isImpAsset += 1

                            bigEnough = 0
                            if expectedDebris > 1000 or isImpAsset > 0:
                                collisionRiskCouplesReal.append([year - START_YEAR, p1.name.strip(), p1.mass, p2.name.strip(), p2.mass, P, num, isImpAsset, 0, 0, 0])
                                bigEnough = 1

                            # Monte-carlo simulation - there is a collision if random number is lower than prob of collision
                            if random.random() < P:
                                # collision is triggered
                                if bigEnough:
                                    collisionRiskCouplesReal[-1][-3] = 1

                                # We assume that newly launched satellites can perform collision avoidance (90% succes) during their active years
                                if any(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip())) or any(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip())):

                                    satel1 = 0
                                    satel2 = 0
                                    if any(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip())) and any(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip())):
                                        satel1 = next(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip()))
                                        satel2 = next(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip()))
                                    elif any(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip())):
                                        satel1 = next(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip()))
                                    elif any(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip())):
                                        satel1 = next(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip()))

                                    if satel1 != 0:
                                        if bigEnough:
                                            collisionRiskCouplesReal[-1][-2] = 1

                                        if satel1.spacecraftClass.spacecraftClass == "large":
                                            collidedOrAvoidedLarge[year - START_YEAR] += 1
                                        elif satel1.spacecraftClass.spacecraftClass == "medium":
                                            collidedOrAvoidedMedium[year - START_YEAR] += 1
                                        elif satel1.spacecraftClass.spacecraftClass == "small":
                                            collidedOrAvoidedSmall[year - START_YEAR] += 1
                                        elif satel1.spacecraftClass.spacecraftClass == "ultra_small":
                                            collidedOrAvoidedUltraSmall[year - START_YEAR] += 1

                                    if satel2 != 0:
                                        if bigEnough:
                                            collisionRiskCouplesReal[-1][-2] = 2

                                        if satel2.spacecraftClass.spacecraftClass == "large":
                                            collidedOrAvoidedLarge[year - START_YEAR] += 1
                                        elif satel2.spacecraftClass.spacecraftClass == "medium":
                                            collidedOrAvoidedMedium[year - START_YEAR] += 1
                                        elif satel2.spacecraftClass.spacecraftClass == "small":
                                            collidedOrAvoidedSmall[year - START_YEAR] += 1
                                        elif satel2.spacecraftClass.spacecraftClass == "ultra_small":
                                            collidedOrAvoidedUltraSmall[year - START_YEAR] += 1

                                    if random.random() < 0.9:
                                        maneuverCollAvoid[year - START_YEAR] += 1
                                        continue

                                # collision avoidance was not successful and collision happens
                                if bigEnough:
                                    collisionRiskCouplesReal[-1][-1] = 1

                                for agentIndex in range(0, len(agentList)):
                                    if any(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip() and sat.owner in agentList[agentIndex])):
                                        impSat = next(sat for sat in satellites if (sat.importantAsset and sat.name == p1.name.strip() and sat.owner in agentList[agentIndex]))

                                        agentCollisions[agentIndex][year - START_YEAR] += 1

                                        if impSat.spacecraftClass.spacecraftClass == "large":
                                            collidedLarge[year - START_YEAR] += 1
                                        elif impSat.spacecraftClass.spacecraftClass == "medium":
                                            collidedMedium[year - START_YEAR] += 1
                                        elif impSat.spacecraftClass.spacecraftClass == "small":
                                            collidedSmall[year - START_YEAR] += 1
                                        elif impSat.spacecraftClass.spacecraftClass == "ultra_small":
                                            collidedUltraSmall[year - START_YEAR] += 1

                                    if any(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip() and sat.owner in agentList[agentIndex])):

                                        impSat = next(sat for sat in satellites if (sat.importantAsset and sat.name == p2.name.strip() and sat.owner in agentList[agentIndex]))

                                        agentCollisions[agentIndex][year - START_YEAR] += 1

                                        if impSat.spacecraftClass.spacecraftClass == "large":
                                            collidedLarge[year - START_YEAR] += 1
                                        elif impSat.spacecraftClass.spacecraftClass == "medium":
                                            collidedMedium[year - START_YEAR] += 1
                                        elif impSat.spacecraftClass.spacecraftClass == "small":
                                            collidedSmall[year - START_YEAR] += 1
                                        elif impSat.spacecraftClass.spacecraftClass == "ultra_small":
                                            collidedUltraSmall[year - START_YEAR] += 1

                                # if there is a collision the object is not operational anymore and it becomes a debris
                                if any(sat for sat in satellites if sat.name == p1.name.strip()):
                                    sat1 = next(sat for sat in satellites if (sat.name == p1.name.strip()))
                                    sat1.importantAsset = False
                                if any(sat for sat in satellites if sat.name == p2.name.strip()):
                                    sat2 = next(sat for sat in satellites if (sat.name == p2.name.strip()))
                                    sat2.importantAsset = False

                                dv = np.linalg.norm(p1._v - p2._v)

                                if p2.mass < p1.mass:
                                    catastrophRatio = (p2.mass * dv**2) / (2 * p1.mass * 1000)
                                else:
                                    catastrophRatio = (p1.mass * dv**2) / (2 * p2.mass * 1000)
                                if catastrophRatio < 40:
                                    numberOfNonCatastrophicCollisions += 1
                                else:

                                    try:
                                        logger.info("!" * 25 + " BOOOOOM " + "!" * 25)
                                        logger.info('planet {} with mass {}'.format(p1.name, p1.mass))
                                        logger.info('planet {} with mass {}'.format(p2.name, p2.mass))
                                        logger.info('collision velocity ' + str(dv))

                                        debris1, debris2 = breakup(ep, p1, p2)
                                        logger.info(str(len(debris1) + len(debris2)) + ' new pieces from collision')
                                        totalNumberOfNewDebris += (len(debris1) + len(debris2))
                                        collisionsCatasPom += 1
                                        debris.remove(p1)
                                        debris.remove(p2)

                                        debris.extend(debris1)
                                        debris.extend(debris2)
                                        numberOfCatastrophicCollisions += 1
                                    except:

                                        pass

        logger.debug('%.2f %d %d %d %10.8f' % (float(ep) / 365.25 - (START_YEAR - 2000), len(debris), len(volumes), max(map(len, volumes)) if len(volumes) else 0, maxp))
    logger.info('There were ' + str(numberOfCatastrophicCollisions) + ' catastrophic collisions')
    logger.info('There were ' + str(numberOfNonCatastrophicCollisions) + ' NON-catastrophic collisions')
    logger.info('From collisions there were ' + str(totalNumberOfNewDebris) + ' of new debris')
    years = range(0, yearToPlot + 1, 1)

    totalDebrisEvol.append(len(debris))

    fileName = stringOfRemovals + "proc_" + '{0:03d}'.format(rank) + "_of_" + '{0:03d}'.format(comm.size) + "_sim" + str(qq) + "_scenario" + str(scenario) + "_" + str(expectedDebrisThresholdForRem) + "_long.txt"
    f = open(fileName, 'w')
    f.write("%s\n" % years)
    f.write("%s\n" % totalDebrisEvol)
    f.write("%s\n" % numberOfCatastrophicCollisions)
    f.write("%s\n" % numberOfNonCatastrophicCollisions)
    f.write("%s\n" % totalNumberOfNewDebris)
    for agentIndex in range(0, len(agentList)):
        f.write("%s\n" % agentCollisions[agentIndex])
    for agentIndex in range(0, len(agentList)):
        f.write("%s\n" % agentRisksProb[agentIndex])
    f.write("%s\n" % commonRisksProb)
    for agentIndex in range(0, len(agentList)):
        f.write("%s\n" % agentTotalImportantAssets[agentIndex])

    f.write("%s\n" % totalImportantAssetsL)
    f.write("%s\n" % totalImportantAssetsM)
    f.write("%s\n" % totalImportantAssetsS)
    f.write("%s\n" % totalImportantAssetsU_S)

    for remObject in listOfRemovedObjectsFinal:
        f.write("%s\n" % remObject)

    f.write("%s\n" % removedPerYear)

    f.write("%s\n" % collidedLarge)
    f.write("%s\n" % collidedMedium)
    f.write("%s\n" % collidedSmall)
    f.write("%s\n" % collidedUltraSmall)

    f.write("%s\n" % collidedOrAvoidedLarge)
    f.write("%s\n" % collidedOrAvoidedMedium)
    f.write("%s\n" % collidedOrAvoidedSmall)
    f.write("%s\n" % collidedOrAvoidedUltraSmall)

    for coll in collisionRiskCouplesReal:
        f.write("%s\n" % coll)

    f.write("%s\n" % launchedPerYear)
    # active in sense of not decayed i.e. operational and non-operational but not decayed yet
    f.write("%s\n" % activeSatelitesNo)
    f.write("%s\n" % activeLarge)
    f.write("%s\n" % activeMedium)
    f.write("%s\n" % activeSmall)
    f.write("%s\n" % activeUltraSmall)

    f.write("%s\n" % launchedLarge)
    f.write("%s\n" % launchedMedium)
    f.write("%s\n" % launchedSmall)
    f.write("%s\n" % launchedUltraSmall)

    f.write("%s\n" % debrisObjectsNo)
#    f.write("%s\n" % len(satellites))
    f.write("%s\n" % sum(launchedPerYear))
    f.write("%s\n" % decaysImmediatelly)
    f.write("%s\n" % removedLaunchedBySGP4Yearly)
    f.write("%s\n" % collisionsCatasYearly)
    f.write("%s\n" % maneuverCollAvoid)
    for remByS in removedBySizeTotal:
        f.write("%s\n" % remByS)

    f.close()
