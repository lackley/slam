import numpy as np
from numpy import dot
import math

def compute_initial_covariance(jacobian, sigmaObservation):
    """ Compute the initial covariance matrix for a landmark

    args:
        jacobian: The Jacobian matrix of the robot's position, with respect to 
        the landmark position (taken from a measurement). 

        sigmaObservation: The sigmaObservation represents the noise added to
        the the observation and is a constant value provided in the stencil code        
    """
    H = np.matrix(jacobian)
    S = np.matrix(sigmaObservation)
    simga_init = H.I * S * (H.I).T

    return simga_init

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

    H = np.matrix(jacobian)
    S = np.matrix(sigmaObservation)
    O = np.matrix(oldCovariance)
    Q = H * O * H.T + S

    return Q

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
    H = np.matrix(jacobian)
    O = np.matrix(oldCovariance)
    Q = np.matrix(measurementCovariance)
    K = O * H.T * Q.I

    return K

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
    #vector of old lanmark posistion
    Lo = np.matrix([[old_landmark[0]], [old_landmark[1]]])
    #vector of current measurement
    Mt = np.matrix([[z[0]], [z[1]]])
    #vector of perdicted measurement
    Mp = np.matrix([[z_hat[0]], [z_hat[1]]])
    K = np.matrix(kalmanGain)
    L = Lo + K * (Mt-Mp)

    return L

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
    I = np.matrix([[1, 0], [0, 1]])
    K = np.matrix(kalmanGain)
    H = np.matrix(jacobian)
    O = np.matrix(oldCovariance)

    new_cov = (I - K*H) * O

    return new_cov
