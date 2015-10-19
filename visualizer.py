# from utils import *
import sys
import matplotlib.pyplot as plt
import matplotlib.path as path
from matplotlib.patches import Ellipse
import numpy
import math

# 0 - off
# 1 - key/mouse input
# 2 - verbose 
debug = 1

def init_vis(lidar_map, landmarks, states, true_state, vis):
    """
    initializes visual display with a set of initial points
    :param landmarks: the actual landmarks, as a grayscale map
    :param states: a class with robot_position, which is (x, y, phi) and a list of predicted landmarks with a mean and covaraince
    :param true_state: the current true state as given by the ground.csv file
    """
    h,w = lidar_map.shape
    
    plt.ion()
    dpi = 100
    plt.figure(figsize=(w*2/dpi, h/dpi), dpi=dpi)
    update_vis(lidar_map, landmarks, states, true_state, vis)

def generateSprite(state, size=20):
    """
    This creates the points of a triangle, to draw a sprite on the screen. It should be passed into plt.plot() to render
    :param
    """
    x, y, phi = state
    phi1 = phi + math.pi/12
    phi2 = phi - math.pi/12
    spriteX = [x, x - math.cos(phi1)*size, x - math.cos(phi2)*size, x]
    spriteY = [y, y - math.sin(phi1)*size, y - math.sin(phi2)*size, y]
    return spriteX, spriteY


def update_vis(lidar_map, landmarks, states, true_state, vis):
    """
    updates the already initialized visual display with new points
    :param landmarks: a grayscale map, showing the actual landmarks
    :param states: an array of particles
    :param true_state: the current true state as given by the ground.csv file, decribed with an x, y, and phi
    :param vis: a kind of visualizer (1 : default, 2 : particle)
    """


    plt.clf()
    plt.subplot(1, 2, 1)
    h,w = lidar_map.shape
    plt.title('Real World')
    plt.imshow(lidar_map, cmap='Greys', aspect='auto')
    plt.axis((0, w, 0, h))

    sprite_size = min(w, h) / 20

    trueX, trueY = generateSprite(true_state, size=sprite_size)
    plt.scatter([l[0] for l in landmarks], [l[1] for l in landmarks], c='cyan',  marker="o", alpha=1)
    plt.plot(trueX, trueY, '-o', c='red', ms=0, lw=2)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Robot World')
    plt.axis((0, w, 0, h))

    max_particle = max(states, key=lambda x: x.weight)
    if debug > 1:
        print "# of landmarks : ", len(max_particle.landmarks)

    if vis == 1:
        for l in max_particle.landmarks:
            if debug > 1:
                print "landmark : ", l[0][0], l[0][1]
                print "cov : ", l[1]

            eigvalues, eigvectors = numpy.linalg.eig(l[1])
            theta = numpy.degrees(math.atan2(*eigvectors[:,0][::-1]))
            width, height = 4 * numpy.sqrt(eigvalues)

            #plt.plot(l[0][0], l[0][1], c="cyan", marker="o", alpha=1)        
            if debug > 1:
                print "eig values : ", eigvalues[0], eigvalues[1]
                print "eig vectors : ", eigvectors[0], eigvectors[1]
                print "wid, height : ", width, height

            ellip = Ellipse(xy=(l[0][0], l[0][1]), width=width, height=height, angle=theta)
            plt.gca().add_artist(ellip)
    else:   
        divide_weight = math.exp(max_particle.weight);     
        if max_particle.weight == 0:
            divide_weight = 1

        for particle in states:
            col = plt.cm.jet(math.exp(particle.weight) / divide_weight)
            for l in particle.landmarks:
                plt.plot(l[0][0], l[0][1], c=col, marker=".", alpha=1)

    for s in states:
        sX, sY = generateSprite(s.robot_position, size=sprite_size)
        plt.plot(sX, sY, '-o', c='green', ms=0, lw=2)

    if debug > 0:
        print "click any point in the window..."
        sys.stdout.flush()
        plt.waitforbuttonpress()

    #plt.plot(trueX, trueY, '-o', c='red', ms=0, lw=2)
    #plt.axis('off')

    
    plt.draw()
