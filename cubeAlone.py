import math
import PyKEP as kep
import numpy as np
import sys
from collections import defaultdict

CUBE_RES = 10e3

#"""Returns a list of list of debris which are at time t in the same volume of resolution res."""
def cube(planets, res=CUBE_RES):
    cubemap = defaultdict(list)
    for p in planets:
        try:
            key = tuple(map(lambda foo: int(math.floor(foo/CUBE_RES)), p._r))
            cubemap[key].append(p)
        except:
            pass
    res = [foo for foo in cubemap.values() if len(foo) > 1]
    return res


def collision_prob(p1, p2):
    sigma = (p1.radius + p2.radius) ** 2 * math.pi # crossectional collision area
    dU = CUBE_RES ** 3 # volume of the cube
    Vimp = np.linalg.norm(p1._v - p2._v) # rel. velocity
    return  Vimp/dU * sigma

def setradii(planets, satcat):
    for p in planets:
        try:
            a = float(satcat[p.name.strip()].radarA)
            p.radius = math.sqrt(a/math.pi)
        except Exception as e:
            p.radius = 0.1 # TODO: sample from radii distribution / use mean

def setmass(planets):    
    for p in planets:
        p.mass = 4./3. * math.pi * p.radius ** 3 * 92.937 * (p.radius * 2) ** -0.74 # EQ 1

def setradiiFromMass(p):    
#    for p in planets:
    p.radius = (p.mass/((4./3.)*math.pi*92.937*2**(-0.74)))**(1/2.26)


def update(planets, ep):
#    removedLaunched = 0
    removedLaunched = []
    for p in planets[:]:
#        ec = float('0.'+p.line2[26:33])
        try:
            p._r, p._v = map(np.array, p.eph(ep))
        except Exception as e:

            planets.remove(p)

            if p.name.strip()[-2:] == 'xL' or p.name.strip()[-2:] == 'xM' or p.name.strip()[-2:] == 'xS' or p.name.strip()[-2:] == 'xU':
#                removedLaunched += 1
                removedLaunched.append(p.name.strip())

    return removedLaunched
