from __future__ import division
import re
import numpy as np
from numpy import dot
from numpy import identity 
from scipy.stats import norm as gaussian
import math
import sys

max_float = sys.float_info.max
debug = False

def calculate_likelihood_probability(measurement, predicted_measurement, covariance): 
    """ Calculates the likelihood probability of a new measurement 
    given the probability distribution of the predicted measurement 
    from the previous timestep.
    The distribution of the predicted measurement is a Gaussian 
    distribution and given as its mean (distance, bearing) and its 
    covariance matrix.
    Compared to multivariate_gauss_prob, this function takes care of
    the singularity issue with the range of bearing measurement.

    args:
        measurement: the measurement at which to calculate the probability
        predicted_measurement: the mean of the Gaussian distribution of 
            the predicted measurement 
        covariance: the covariance of the Gaussian distribution of the 
            predicted measurement

    returns:
        the likelihood probability of a current measurement

    """
    
    return None
 
def multivariate_gauss_prob(observed, mean, covariance): 

    """ Calculates the probability density at the observed point
    of a gaussian distribution with the specified mean and covariance

    The probability density at an observed point is, given a distribution, 
    what is the likelihood of observing the provided point. The mean 
    and covariance describe a gaussian distribution. 

    args: 
        observed: The point at which to calculate the probability density
        mean: the mean of the gaussian distribution
        covariance: the covariance of the gaussian distribution

    returns: 
        the probability of selecting the observed point

    """

    return None

def distance(p1, p2):

    """Calculates the distance between points p1 and p2

    args:
        p1: (x, y)
        p2: (x, y)

    returns: 
        the distance

    """
    return None

def calculate_jacobian(robot_position, landmark_pos): 

    """ Calculates the Jacobian matrix of a particular observatoin

    The Jacobian matrix is a linearization of the map from (x, y) to 
    (distance, bearing) coordinates

    args: 
        robot_position: the (x, y) coordinates of the robot's position
            Note: will work fine if given (x, y, phi)
        landmark_pos: the (x, y) position of a particular landmark

    returns: 
        the Jacobian matrix of the robot_position, with respect to 
        the landmark_pos
    """

    return None

def compute_measurement_covariance(jacobian, oldCovariance, sigmaObservation):

    """ Compute the measurement covariance which is used in calculating the probability
        of a new measurement given the existing measurements (the likelihood of correspondence).

        args:
            jacobian: The Jacobian matrix of the newly sampled robot position with
            respect to the existing landmark position

            oldCovariance: The covariance matrix of the existing landmark at the previous
            time step

            sigmaObservation: The sigmaObservation represents the noise added to
            the the observation and is a constant value provided in the stencil code

        returns:
            The measurement covariance matrix according to the newly sampled robot pose and
            the previous landmark position
    """    

    return None

def compute_initial_covariance(jacobian, sigmaObservation):
    """ Compute the initial covariance matrix for a landmark

    args:
        jacobian: The Jacobian matrix of the robot's position, with respect to 
        the landmark position (taken from a measurement). 

        sigmaObservation: The sigmaObservation represents the noise added to
        the the observation and is a constant value provided in the stencil code        
    """

    return None


def compute_kalman_gain(jacobian, oldCovariance, measurementCovariance):
    """ Compute the Kalman gain 

    The Kalman gain represents how much a new landmark measurement affects an old measurement
    
    args:
        jacobian: The Jacobian matrix of the robot's position, with respect to 
        the landmark position (taken from a measurement). Here the Jacobian 
        represents the new landmark measurement
        
        oldCovariance: The covariance associated with a landmark measurement
        prior to an update. Here the old covariance matrix is representative
        of the old measurement

        measurementCovariance: The measurementCovariance represents the covariance of the previous
        measurement and the noise added to the new measurement
    """

    return None

def compute_new_landmark(z, z_hat, kalmanGain, old_landmark):
    """ Compute the new landmark's position estimate according to the Kalman Gain
    
    args:
        z : the current measurement of a new landmark

        z_hat : the predicted measurement from the previous landmark position estimate and
        the new robot control measurement
        
        kalmanGain: The measure of how much the new measurement affects the 
        old/believed measurment of the landmark's location
        
        old_landmark : the position esitimate of the landmark in the previous timestep

    returns:
        The updated landmark mean based on the new measurement, believed measurement,
        and Kalman Gain    
    """

    return None
    
def compute_new_covariance(kalmanGain, jacobian, oldCovariance):
    """ Compute the new covariance of the landmark

    The new covariance matrix of the landmark being updated
    is based on the Kalman gain, the Jacobian matrix, and the
    old covariance of this landmark

    args:
        kalmanGain: The measure of how much the new measurement affects the 
        old/believed measurment of the landmark's location

        jacobian: The Jacobian matrix of the robot's position, with respect to 
        the landmark position (taken from a measurement)
        
        oldCovariance: The covariance associated with a landmark measurement
        prior to an update
        
    returns:
        The new covariance of the landmark being updated    
    """

    return None

def gauss_sample(mean, covariance):
    """ Samples randomly from a gaussian (normal) distribution

    args: 
        mean: the mean of the distribution (distance, bearing)
        param covariance: the covariance of the distribution, a 2x2 numpy array

    returns: 
        a random sample from a gaussian with the specified parameters of the form (distance, bearing)
    """

    return None

