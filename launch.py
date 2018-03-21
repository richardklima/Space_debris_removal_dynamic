import PyKEP as kep
import numpy as np
import random
import math
import datetime
import time
from scipy.stats import norm
import matplotlib.mlab as mlab
from collections import namedtuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

START_YEAR = 2017
# How many of past years we take distributions of orbital element from
DISTRIBUTION_FROM_PAST_YEARS = 20

importantAssets = []
importantAssetsTLE = []
inclination_distributions = []
eccentricity_distributions = []
W_distributions = []
w_distributions = []
monthOfLaunchesDistributions = []
yearLaunchesDistributions = []
meanMotion_distributions = []
bstar_distributions = []

# sample:
# a (semi major axis = altitude + earth radius)
# i (inclination)
# W (progression of the orbital plane)
# w (where is the closest point of the elipse to the center?)
# always sample uniform between 0, 360 deg
# M (where is the debris on the orbit at the reference epoch?)


def loadData_satcat():
    satcat = kep.util.read_satcat('satcat2017.txt')
    return satcat


def loadData_tle():
    debris = kep.util.read_tle('tle2017.tle', with_name=False)
    return debris


def findImportantAssets(planets, currentYear, expectedLifespan):
    importantAssets = dict()
    for i in planets.items():
        beginning = currentYear - expectedLifespan
        decay = i[1].decay.strip()
        # some satcat entries are missing information - e.g. orbital status code NEA - no elements available
        try:
            apogee = int(i[1].apogee)
            perigee = int(i[1].perigee)
            yearOfLaunchNo = int(i[1].launchdate.split('-')[0])
        except:
            apogee = 0
            perigee = 0
            yearOfLaunchNo = 0
        if yearOfLaunchNo >= beginning and i[1].decay.strip() == "" and 'DEB' not in i[1].name and 'R/B' not in i[1].name and perigee > 0 and perigee < 2000:
            importantAssets[i[0]] = i[1]

    return importantAssets


def findTLEs(satcatList, TLEList):
    TLEs = []
    for planetTLE in TLEList:
        for planet in satcatList.items():
            if planet[0].strip() == planetTLE.name.strip():
                TLEs.append(planetTLE)
                break
    return TLEs


def setImportantAssets(satellites, currentYear, month):
    for sat in satellites:

        endOperationalYear = int(sat.endOperationalDate.split('-')[0])
        endOperationalMonth = int(sat.endOperationalDate.split('-')[1])

        if endOperationalYear < currentYear or (endOperationalYear == currentYear and endOperationalMonth <= month):
            sat.importantAsset = False


def findDistributions(importantAssets, importantAssetsTLE):
    global inclination_distributions
    global eccentricity_distributions
    global W_distributions
    global w_distributions
    global monthOfLaunchesDistributions
    global yearLaunchesDistributions
    global meanMotion_distributions
    global bstar_distributions

    # finding distributions for all agents combined, only some satellites - filtered by suitbale ranges of orbital elements
    inclinPom = []
    eccentrPom = []
    WPom = []
    wPom = []
    meanMotionPom = []
    monthOfLaunches = []
    yearLaunches = []
    bstarPom = []
    for plan in importantAssets.items():
        for tleObj in importantAssetsTLE:
            if plan[0].strip() == tleObj.name.strip():

                inclin = float(tleObj.line2[8:16])
                eccent = float("0." + tleObj.line2[26:33])
#                print eccent
                # mean motion = n
                n = float(tleObj.line2[52:63])
                # Standard gravitational parameter of Earth
                mu = float(398600)
                # semimajor axis
                a = (mu / (n * 2. * math.pi / (float(24 * 3600))) ** 2.) ** (1. / 3.)

                if tleObj.line1[53:54] == " ":
                    bstarMantissa = float("0." + tleObj.line1[54:59])
                elif tleObj.line1[53:54] == "-":
                    bstarMantissa = float("-0." + tleObj.line1[54:59])

#                bstarMantissa = float("0."+tleObj.line1[53:59])
                bstarExponent = float(tleObj.line1[59:61])
                bstar = bstarMantissa * math.pow(10, bstarExponent)
                # reference air density
                rhoZero = 2.461e-5
                # draf coeffficient for a sphere
                Cd = 2.2
                if "N/A" in plan[1].radarA:
                    pass
                else:
                    if bstar == 0.0:
                        pass
                    else:
                        massPom = rhoZero * Cd * float(plan[1].radarA) / (2. * bstar)

                if inclin > 0 and inclin < 360 and eccent > 0.0 and eccent < 0.5 and (a - 6378) > 300 and (a - 6378) < 1200:
                    inclinPom.append(inclin)
                    eccentrPom.append(eccent)
                    meanMotionPom.append(float(tleObj.line2[52:63]))

                    # we won't use these ones - we sample uniformly at random
                    WPom.append(float(tleObj.line2[17:25]))
                    wPom.append(float(tleObj.line2[34:42]))
                    bstarPom.append(bstar)

                    yearLaunches.append(int(plan[1][6][:4]))
                    monthOfLaunches.append(int(plan[1][6][5:7]))

                break

    inclination_distributions.append(inclinPom)
    eccentricity_distributions.append(eccentrPom)
    W_distributions.append(WPom)
    w_distributions.append(wPom)
    meanMotion_distributions.append(meanMotionPom)
    monthOfLaunchesDistributions.append(monthOfLaunches)
    yearLaunchesDistributions.append(yearLaunches)
    bstar_distributions.append(bstarPom)


def findHistograms():
    # lifespan for finding oe distributions
    expectedLifespan = DISTRIBUTION_FROM_PAST_YEARS
    planetsTLE = loadData_tle()
    satcat = loadData_satcat()
    importantAssets = findImportantAssets(satcat, START_YEAR, expectedLifespan)
    importantAssetsTLE = findTLEs(importantAssets, planetsTLE)
    findDistributions(importantAssets, importantAssetsTLE)
    histValues_inclination = []
    histValues_eccentricity = []
    histValues_W = []
    histValues_w = []
    binsEdges_inclination = []
    binsEdges_eccentricity = []
    binsEdges_W = []
    binsEdges_w = []
    histValues_meanMotion = []
    binsEdges_meanMotion = []
    histValues_bstar = []
    binsEdges_bstar = []
    startOfLaunches = int(START_YEAR - expectedLifespan)
    years = range(startOfLaunches, START_YEAR, 1)
    yearLaunches = []
    # iterate over players plus 1 for other players and plus 1 for all together
    i = 0
    yearLaunchesPom = dict()
    for y1 in years:
        yearLaunchesPom[y1] = 0
    h_i, b_i = np.histogram(inclination_distributions[i], bins=50, normed=True)
    h_i = h_i.astype(np.float32)
    histValues_inclination.append(h_i)
    binsEdges_inclination.append(b_i)

    h_e, b_e = np.histogram(eccentricity_distributions[i], bins=50, normed=True)
    h_e = h_e.astype(np.float32)
    histValues_eccentricity.append(h_e)
    binsEdges_eccentricity.append(b_e)

    h_W, b_W = np.histogram(W_distributions[i], bins=50, normed=True)
    h_W = h_W.astype(np.float32)
    histValues_W.append(h_W)
    binsEdges_W.append(b_W)

    h_w, b_w = np.histogram(w_distributions[i], bins=50, normed=True)
    h_w = h_w.astype(np.float32)
    histValues_w.append(h_w)
    binsEdges_w.append(b_w)

    h_meanMotion, b_meanMotion = np.histogram(meanMotion_distributions[i], bins=50, normed=True)
    h_meanMotion = h_meanMotion.astype(np.float32)
    histValues_meanMotion.append(h_meanMotion)
    binsEdges_meanMotion.append(b_meanMotion)

    h_bstar, b_bstar = np.histogram(bstar_distributions[i], bins=50, normed=True)
    h_bstar = h_bstar.astype(np.float32)
    histValues_bstar.append(h_bstar)
    binsEdges_bstar.append(b_bstar)
    for y in yearLaunchesDistributions[i]:
        if y in years:
            yearLaunchesPom[y] += 1
    yearLaunches.append(yearLaunchesPom)
#    print yearLaunchesPom
    oe_histograms = [histValues_inclination, binsEdges_inclination, histValues_eccentricity, binsEdges_eccentricity, histValues_W, binsEdges_W, histValues_w, binsEdges_w, histValues_meanMotion, binsEdges_meanMotion, histValues_bstar, binsEdges_bstar, yearLaunches]
    return oe_histograms  # print len(histValues_inclination)



def sample_oe(oe_histograms):
    player = 0
    histValues_inclination = oe_histograms[0]
    binsEdges_inclination = oe_histograms[1]
    histValues_eccentricity = oe_histograms[2]
    binsEdges_eccentricity = oe_histograms[3]
    histValues_W = oe_histograms[4]
    binsEdges_W = oe_histograms[5]
    histValues_w = oe_histograms[6]
    binsEdges_w = oe_histograms[7]
    histValues_meanMotion = oe_histograms[8]
    binsEdges_meanMotion = oe_histograms[9]
    histValues_bstar = oe_histograms[10]
    binsEdges_bstar = oe_histograms[11]
    # change player number to index of player
#    player = player - 1
    probs_i = histValues_inclination[player] / sum(histValues_inclination[player])
    sampled_i_pom = int(np.random.choice(range(1, len(binsEdges_inclination[player])), 1, p=probs_i))
    sampled_i = random.uniform(binsEdges_inclination[player][sampled_i_pom - 1], binsEdges_inclination[player][sampled_i_pom])

    probs_e = histValues_eccentricity[player] / sum(histValues_eccentricity[player])
    sampled_e_pom = int(np.random.choice(range(1, len(binsEdges_eccentricity[player])), 1, p=probs_e))
    sampled_e = random.uniform(binsEdges_eccentricity[player][sampled_e_pom - 1], binsEdges_eccentricity[player][sampled_e_pom])

    sampled_W = random.uniform(0, 360)

    sampled_w = random.uniform(0, 360)

    probs_meanMotion = histValues_meanMotion[player] / sum(histValues_meanMotion[player])
    sampled_meanMotion_pom = int(np.random.choice(range(1, len(binsEdges_meanMotion[player])), 1, p=probs_meanMotion))
    sampled_meanMotion = random.uniform(binsEdges_meanMotion[player][sampled_meanMotion_pom - 1], binsEdges_meanMotion[player][sampled_meanMotion_pom])

    probs_bstar = histValues_bstar[player] / sum(histValues_bstar[player])
    sampled_bstar_pom = int(np.random.choice(range(1, len(binsEdges_bstar[player])), 1, p=probs_bstar))
    sampled_bstar = random.uniform(binsEdges_bstar[player][sampled_bstar_pom - 1], binsEdges_bstar[player][sampled_bstar_pom])

    sample_M = random.uniform(0, 360)

    return sampled_i, sampled_e, sampled_W, sampled_w, sample_M, sampled_meanMotion, sampled_bstar


def _checksum(line):
    res = 0
    for c in line:
        if 48 < ord(c) <= 58:
            res += ord(c) - 48
        if c == '-':
            res += 1
    return res % 10


def create_tle(oscul_elements, agentNo, year, month, launchIterator, typeOfSat):
    month = int(month)
    if len(str(month)) == 1:
        month = '0' + str(month)

    if len(str(launchIterator)) == 1:
        launchIterator = '00' + str(launchIterator)
    if len(str(launchIterator)) == 2:
        launchIterator = '0' + str(launchIterator)

    # Standard gravitational parameter of Earth
    mu = float(398600)
    n = float(oscul_elements[5])
    a = (mu / (n * 2. * math.pi / (float(24 * 3600))) ** 2.) ** (1. / 3.)
    e = float(oscul_elements[1])
    apogee = (1 + e) * (a - 6378)
    perigee = (1 - e) * (a - 6378)

    # satellite name with code of agent
    satellite_number = str(year)[2:] + str(agentNo) + '99'
    launchNo = str(month) + str(launchIterator)
    if oscul_elements[6] < 0:
        s = "%.*e" % (4, oscul_elements[6])
    else:
        s = "%.*e" % (4, oscul_elements[6])

    mantissa, exp = s.split('e')
    mantissa = float(mantissa) / 10.0
    mantissa = "{:0.5f}".format(float(mantissa))
    exp = int(exp) + 1
    bstarFormat = "%s%+0*d" % (mantissa, 2, int(exp))
    if bstarFormat[0] == "-":
        bstarFormat = bstarFormat[0:1] + bstarFormat[3:]
    else:
        bstarFormat = " " + bstarFormat[2:]

    line1 = "1 " + satellite_number + "  " + str(year)[2:] + launchNo + "  11111.11111111 +.11111111 +00000-0  00000-0 0  1110"
#    line1 = "1 "+satellite_number+"  "+str(year)[2:]+launchNo+"  11111.11111111 +.11111111 +00000-0 "+bstarFormat+" 0  1110"
    ep_date = str(year)
    ep_day = 99.9999999
    ep_str = str(ep_date[2:] + '{:12.8f}'.format(ep_day)[:14])
    line1 = line1[:18] + ep_str + line1[32:-1]
    line1 += str(_checksum(line1))
    line2 = "2 " + satellite_number + " "
    line2 += '{:8.4f} '.format(oscul_elements[0])  # inclination (i)
    line2 += '{:8.4f} '.format(oscul_elements[2])  # RA (W)
    line2 += '{:.7f} '.format(oscul_elements[1])[2:]            # eccentrictiy (e)
    line2 += '{:8.4f} '.format(oscul_elements[3])  # argument of perigee (w)
    line2 += '{:8.4f} '.format(oscul_elements[4])  # mean anomaly (M)
    line2 += '{:11.8f}'.format(oscul_elements[5])  # mean motion (n)
    line2 += '{:5d}'.format(0)  # revolutions
    line2 += str(_checksum(line2))
    tlePlanet = kep.planet.tle(line1, line2)
    dayOfYearFractional = float(int(month) * 30)
    tlePlanet.set_epoch(year, dayOfYearFractional)
    tlePlanet.name = str(year) + '-' + launchNo + typeOfSat
    return tlePlanet


def plotOrbits(planets_tle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ep = kep.epoch(17)
    for pln in planets_tle:
        e = float("0." + pln.line2[26:32])
        i = float(pln.line2[8:15])
        W = float(pln.line2[17:24])
        w = float(pln.line2[34:41])
        M = 20
        # print e,i,W,w
        oe = [2000000 + 6378000, e, i * kep.DEG2RAD, W * kep.DEG2RAD, w * kep.DEG2RAD, M * kep.DEG2RAD]
        pl = kep.planet.keplerian(ep, oe, earth.mu_self, 0, 0, 0, '')
        kep.orbit_plots.plot_planet(pl, ax=ax, alpha=0.2, s=0, color='red')
    plt.show()


def create_satcat(tleObject, agentNo, year, month, obeyMitigation, decayDate):
    if len(str(month)) == 1:
        month = '0' + str(month)
    # Standard gravitational parameter for the earth
    mu = float(398600)
    n = float(tleObject.line2[52:63])
    a = (mu / (n * 2. * math.pi / (float(24 * 3600))) ** 2.) ** (1. / 3.)
    e = float('0.' + tleObject.line2[26:33])

    agents = ['CIS', 'US', 'PRC', 'EU', 'OTH', 'ALL']
    nameOfNew = tleObject.name.strip()
    satcatentry = namedtuple('satcatentry', 'noradn multnameflag payloadflag operationstatus name ownership launchdate launchsite decay period incl apogee perigee radarA orbitstatus')
    intdsgn = tleObject.name.strip()
    noradNo = '9999'
    nameFlag = ' '
    payloadFlag = ' '
    statusCode = ' '
    sateliteName = 'NEW_LAUNCH'
    owner = agents[agentNo]
    launchDate = str(year) + '-' + str(month) + '-' + '99'
    launchSite = 'none'
    decayDate_str = decayDate
    orbitalPeriod = '9999'
    inclination = float(tleObject.line2[8:15])
    apogee = (1 + e) * (a - 6378)
    perigee = (1 - e) * (a - 6378)
    crossSectionA = '5'
    orbStatusCode = ' '

    return [intdsgn, satcatentry(noradNo, nameFlag, payloadFlag, statusCode, sateliteName, owner, launchDate, launchSite, decayDate_str, orbitalPeriod, inclination, apogee, perigee, crossSectionA, orbStatusCode)]


def decayOldObjects(satellites, listOfCollided, currentYear, month):
    decayedSats = []
    time1 = time.time()
    satellitesList = satellites[:]
    for sat in satellitesList:

        decayYear = int(sat.decayDate.split('-')[0])
        decayMonth = int(sat.decayDate.split('-')[1])

        if decayYear > START_YEAR and (decayYear < currentYear or (decayYear == currentYear and decayMonth < month)):

            decayedSats.append(sat.name)
            satellites.remove(sat)

    time4 = time.time()

    while any(sat for sat in satellites if sat.name in listOfCollided):
        satToRem = next(sat for sat in satellites if sat.name in listOfCollided)
        satellites.remove(satToRem)
        listOfCollided.remove(satToRem.name)

    return satellites, decayedSats


def getSatellites(satcat, debris, spacecraft_classes, currentYear, month):
    satellites = []
    for debTLE in debris:
        if int(debTLE.name.strip()[0:4]) < 1990:
            debris.remove(debTLE)

    for debSATCAT in satcat.items():
        try:
            yearOfLaunch = int(debSATCAT[1].launchdate.split('-')[0])
            if 'DEB' in debSATCAT[1].name or 'R/B' in debSATCAT[1].name or yearOfLaunch < 1990:
                del satcat[debSATCAT[0]]
        except:
            del satcat[debSATCAT[0]]

    for debTLE in debris:
        for debSATCAT in satcat.items():
            if debSATCAT[0].strip() == debTLE.name.strip():

                yearOfLaunch = int(debSATCAT[1].launchdate.split('-')[0])
                monthOfLaunch = int(debSATCAT[1].launchdate.split('-')[1])

                try:
                    a = float(debSATCAT[1].radarA)
                    radius = math.sqrt(a / math.pi)
                except Exception as e:
                    radius = 0.5  # TODO: sample from radii distribution / use mean

                mass = 4. / 3. * math.pi * radius ** 3 * 92.937 * (radius * 2) ** -0.74  # EQ 1
                if mass > 500:
                    type_of_satellite_to_launch = spacecraft_classes[0]
                if mass < 500 and mass > 100:
                    type_of_satellite_to_launch = spacecraft_classes[1]
                if mass < 100 and mass > 10:
                    type_of_satellite_to_launch = spacecraft_classes[2]
                if mass < 10:
                    type_of_satellite_to_launch = spacecraft_classes[3]

                new_satellite = Satellite.old_sat(type_of_satellite_to_launch, debSATCAT, debTLE, mass, radius, yearOfLaunch, monthOfLaunch)
                decayYear = int(new_satellite.decayDate.split('-')[0])
                decayMonth = int(new_satellite.decayDate.split('-')[1])

                if decayYear < currentYear or (decayYear == currentYear and decayMonth < month):
                    pass
                else:
                    satellites.append(new_satellite)

                break

    return satellites


class Satellite:

    @classmethod
    def new_sat(cls, spacecraftClass, agentNo, oe_histograms, year, month, launchIterator, obeyMitigation):
        agents = ['CIS', 'US', 'PRC', 'EU', 'OTH', 'ALL']
        obj = cls()
        obj.spacecraftClass = spacecraftClass
        obj.cost = obj.getCost()
        obj.mass = obj.getMass()
        obj.operationalTime = obj.getOperationalTime()
        obj.decayTime = obj.getDecayTime(obj.operationalTime)

        endOperationalYear = int(math.floor(obj.operationalTime))
        endOperationalMonth = int(math.floor((obj.operationalTime - endOperationalYear) / (1. / 12.)) + 1)
        endOperationalMonth = int(endOperationalMonth + month)
        if endOperationalMonth < 10:
            endOperationalMonth = '0' + str(endOperationalMonth)
        elif endOperationalMonth > 12:
            endOperationalMonth = endOperationalMonth % 12
            endOperationalYear += 1
        endOperationalDate = str(year + endOperationalYear) + '-' + str(endOperationalMonth) + '-99'
        obj.endOperationalDate = endOperationalDate

        decayYear = int(math.floor(obj.decayTime))
        decayMonth = int(math.floor((obj.decayTime - decayYear) / (1. / 12.)) + 1)
        decayMonth = int(decayMonth + month)
        if decayMonth < 10:
            decayMonth = '0' + str(decayMonth)
        elif decayMonth > 12:
            decayMonth = decayMonth % 12
            decayYear += 1
        decayDate = str(year + decayYear) + '-' + str(decayMonth) + '-99'
        obj.decayDate = decayDate

        obj.tle = obj.getTLE(agentNo, oe_histograms, year, month, launchIterator)
        obj.tle.mass = obj.mass
        obj.tle.radius = (obj.mass / ((4. / 3.) * math.pi * 92.937 * 2**(-0.74)))**(1 / 2.26)
        obj.satcat = obj.getSATCAT(obj.tle, agentNo, year, month, obeyMitigation, decayDate)
        obj.name = obj.tle.name.strip()
        # obj.owner = obj.satcat[1][5].strip()
        obj.owner = agents[agentNo]
        obj.importantAsset = True
        return obj

    @classmethod
    def old_sat(cls, spacecraftClass, satcat, tle, mass, radius, yearOfLaunch, monthOfLaunch):
        obj = cls()
        obj.spacecraftClass = spacecraftClass
        obj.cost = obj.getCost()
        obj.operationalTime = obj.getOperationalTime()
        obj.decayTime = obj.getDecayTime(obj.operationalTime)

        endOperationalYear = int(math.floor(obj.operationalTime))
        endOperationalMonth = int(math.floor((obj.operationalTime - endOperationalYear) / (1. / 12.)) + 1)
        endOperationalMonth = int(endOperationalMonth + monthOfLaunch)
        if endOperationalMonth < 10:
            endOperationalMonth = '0' + str(endOperationalMonth)
        elif endOperationalMonth > 12:
            endOperationalMonth = endOperationalMonth % 12
            if endOperationalMonth < 10:
                endOperationalMonth = '0' + str(endOperationalMonth)
            endOperationalYear += 1
        endOperationalDate = str(yearOfLaunch + endOperationalYear) + '-' + str(endOperationalMonth) + '-99'
        obj.endOperationalDate = endOperationalDate
#        print obj.endOperationalDate

        decayYear = int(math.floor(obj.decayTime))
        decayMonth = int(math.floor((obj.decayTime - decayYear) / (1. / 12.)) + 1)
        decayMonth = int(decayMonth + monthOfLaunch)
        if decayMonth < 10:
            decayMonth = '0' + str(decayMonth)
        elif decayMonth > 12:
            decayMonth = decayMonth % 12
            decayYear += 1
        decayDate = str(yearOfLaunch + decayYear) + '-' + str(decayMonth) + '-99'
        obj.decayDate = decayDate

        obj.tle = tle
        obj.tle.mass = mass
        obj.tle.radius = radius
        obj.satcat = satcat
        obj.name = obj.tle.name.strip()
        obj.owner = obj.satcat[1][5].strip()
        obj.importantAsset = True

        return obj
# ENABLE for onlyLaunchingSim.py
#    def __init__(self, spacecraftClass, year, month, launchIterator):
#        self.spacecraftClass = spacecraftClass
#        self.typeOfSpaceCraft = self.spacecraftClass.spacecraftClass
#        self.mass = self.getMass()
#        self.operationalTime = self.getOperationalTime()
#        self.decayTime = self.getDecayTime(self.operationalTime)
#
#        decayYear = int(math.floor(self.decayTime))
#        decayMonth = int(math.floor((self.decayTime - decayYear) / (1./12.)) + 1)
#        if decayMonth < 10:
#            decayMonth = '0'+str(decayMonth)
#        decayDate = str(year+decayYear) + '-'+ str(decayMonth) + '-99'
#        self.decayDate = decayDate

    def getCost(self):
        cost = random.uniform(self.spacecraftClass.costMin, self.spacecraftClass.costMax)
        return cost

    def getMass(self):
        mass = random.uniform(self.spacecraftClass.massMin, self.spacecraftClass.massMax)
        return mass

    def getOperationalTime(self):
        opTime = random.uniform(self.spacecraftClass.opTimeMin, self.spacecraftClass.opTimeMax)
        return opTime

    def getDecayTime(self, operationalTime):
        decayTime = operationalTime + random.uniform(self.spacecraftClass.decayTimeMin, self.spacecraftClass.decayTimeMax)

        return decayTime

    def getTLE(self, agentNo, oe_histograms, year, month, launchIterator):
        if self.spacecraftClass.spacecraftClass == "large":
            typeOfSat = 'xL'
        elif self.spacecraftClass.spacecraftClass == "medium":
            typeOfSat = 'xM'
        elif self.spacecraftClass.spacecraftClass == "small":
            typeOfSat = 'xS'
        elif self.spacecraftClass.spacecraftClass == "ultra_small":
            typeOfSat = 'xU'
#        tle = create_tle(sample_oe(agentNo, oe_histograms),agentNo, year, month, launchIterator, typeOfSat)
        tle = create_tle(sample_oe(oe_histograms), agentNo, year, month, launchIterator, typeOfSat)
        return tle

    def getSATCAT(self, tle, agentNo, year, month, obeyMitigation, decayDate):

        satcat = create_satcat(tle, agentNo, year, month, obeyMitigation, decayDate)
        return satcat


class SpacecraftClass:

    def __init__(self, spacecraftClass, mean_Normal, sd_Normal, costMin, costMax, massMin, massMax, opTimeMin, opTimeMax, decayTimeMin, decayTimeMax):
        self.spacecraftClass = spacecraftClass
        self.mean_Normal = mean_Normal
        self.sd_Normal = sd_Normal
        self.costMin = costMin
        self.costMax = costMax
        self.massMin = massMin
        self.massMax = massMax
        self.opTimeMin = opTimeMin
        self.opTimeMax = opTimeMax
        self.decayTimeMin = decayTimeMin
        self.decayTimeMax = decayTimeMax

    def getProbability(self, year, spacecraft_classes):

        g = np.exp(-np.power(year - self.mean_Normal, 2.) / (2 * np.power(self.sd_Normal, 2.)))
        sum_g = 0
        for sc in spacecraft_classes:
            sum_g = sum_g + np.exp(-np.power(year - sc.mean_Normal, 2.) / (2 * np.power(sc.sd_Normal, 2.)))

        probOfLaunch = g / sum_g

        return probOfLaunch


fig = plt.figure()

oe_histograms = findHistograms()
weights0 = np.ones_like(inclination_distributions[0])/len(inclination_distributions[0])

ax1 = fig.add_subplot(132)
n2, binsEdges2, patches2 = ax1.hist(inclination_distributions[0], 50, weights=weights0, facecolor='blue', alpha=0.5, label = 'RU')

ax1.set_title('Inclination distributions', fontsize=22)
ax1.set_xlabel('degrees [deg]', fontsize=20)

weights0 = np.ones_like(eccentricity_distributions[0])/len(eccentricity_distributions[0])



ax2 = fig.add_subplot(133)
n2, binsEdges2, patches2 = ax2.hist(eccentricity_distributions[0], 50, weights=weights0, facecolor='blue', alpha=0.5)

ax2.set_title('Eccentricity distributions', fontsize=22)
ax2.set_xlabel('eccentricity', fontsize=20)

# Standard gravitational parameter of Earth
mu = float(398600)
semi_major_axis_distribution = []
for n in meanMotion_distributions[0]:
# semimajor axis
    a = (mu/(n * 2. * math.pi/(float(24 * 3600))) ** 2.) ** (1./3.) - 6378
    semi_major_axis_distribution.append(a)

weights5 = np.ones_like(semi_major_axis_distribution)/len(semi_major_axis_distribution)

ax3 = fig.add_subplot(131)
n2, binsEdges2, patches2 = ax3.hist(semi_major_axis_distribution, 50, weights=weights5,  facecolor='blue', alpha=0.5, label = 'ALL')

ax3.set_title('Semi-major axis distributions', fontsize=22)
ax3.set_xlabel('semi-major axis [km] - R_E', fontsize=20)
ax3.set_ylabel('number of objects (normalized)', fontsize=20)

# plt.show()

