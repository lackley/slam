#! /usr/bin/python

from __future__ import division
import numpy as np
from numpy import dot
import math
import argparse
import utilsObfuscated as utils
import visualizer
import random
from time import sleep
import copy

np.set_printoptions(suppress=True)
debug = False

class Particle(object): 
    """ container for particle information

    each particle holds its estimation of the robot's location
    as well as its estimate of each landmark's location

    attributes: 
        robot_position: a tuple of the robot's position (x, y) and 
            heading (phi)
                i.e., (x, y, phi)
        landmarks: a list of tuples of landmark locations described by 
            mean (x, y) and a covariance matrix sigma
                    i.e., [((x, y), sigma)]
        weight: the current weight for the particle; stored inside
            the particle for convenience. remember to reset after
            resampling

    """

    def __str__(self): 
       """ 
       Return a string reprsenting each particle; good for debugging purposes. 
       """
       return ""

    def __init__(self, x, y, phi):
        super(Particle, self).__init__()
        self.robot_position = (x, y, phi)
        self.landmarks = []
        # log of the weight
        self.weight = 0.0; 


# ----- PARAMETERS ----- Feel free to experiment, but please use these values for your final handin
# velocity noise
sigma_v = 0.1
# radial noise
sigma_r = 0.01
# added control noise; for update_robot sample
sigma_control = np.array([[sigma_v**2, 0], [0, sigma_r**2]])

# distance (range) noise 
sigma_d = 3
# bearing noise
sigma_p = .30
# observation noise
sigma_observation = np.array([[sigma_d**2, 0], [0, sigma_p**2]])

# probability threshold  (for update_map)
prob_threshold = 0.005

def generate_initial_particles(lmap, num_states, init_state):
    """ generates the list of initial particles

    args: 
        num_states: the number of particles to generate
        init_state: the position in which to generate the particles

    returns: 
        the list of generated particles

    """
    return [Particle(init_state[0], init_state[1], init_state[2]) for i in range(num_states)]


def update_robot(particle, control): 
    """ updates the robot's position according to control

    the robot's new position is determined based on the control, with 
    added noise to account for control error. 

    args: 
        particle: the particle containing the robot position information
        control: the control inputted on the robot
    
    returns: 
        the inputed particle 
    """
    curr_x, curr_y, curr_theta = particle.robot_position
    control_theta, control_velocity = control
    

    #value of this noise is sampled from a multivariate gaussian with 
    #mean of the control's velocity and bearing
    # covariance (represented as a 2x2 matrix) of the parameter sigma_control (the control noise)

    mean = (control_velocity, control_theta)
    noise_distance, noise_bearing = utils.gauss_sample(mean, sigma_control)
    
    #new absolute angle is current angle plus the bearing calculated
    new_theta = curr_theta + noise_bearing
    #use new angle to calculate new posistions
    new_x = curr_x + math.cos(new_theta)*noise_distance
    new_y = curr_y + math.sin(new_theta)*noise_distance
    #update the robot's posistion
    particle.robot_position = (new_x, new_y, new_theta)

    return particle

def add_landmark(particle, measurement): 
    """ adds newly observed landmark to particle

    if a landmark has not been matched to an existing landmark, 
    add it to the particle's list with the appropriate 
    mean (x, y) and covariance (sigma)

    args: 
        particle: the particle to add the new landmark to
        measurement: the measurement to the new landmark (distance, bearing) to add to the particle

    returns: 
        None
    """
    robot_x, robot_y, robot_theta = particle.robot_position
    distance, bearing = measurement


    #use trig to find the landmark's possistion
    landmark_x = distance*math.cos(bearing + robot_theta) + robot_x
    landmark_y = distance*math.sin(bearing + robot_theta) + robot_y

    jacobian = utils.calculate_jacobian((robot_x, robot_y), (landmark_x, landmark_y))
    init_cov = utils.compute_initial_covariance(jacobian, sigma_observation)

    #initialize particle.landmarks if necessary 
    #add the posistion and covariacne
    if len(particle.landmarks) == 0:
        particle.landmarks = [((landmark_x, landmark_y), init_cov)]
    else:
        particle.landmarks.append(((landmark_x, landmark_y), init_cov))

    return None

def update_landmark(particle, landmark, measurement):
    """ update the mean and covariance of a landmark

    uses the Extended Kalman Filter (EKF) to update the existing
    landmark's mean (x, y) and covariance according to the new measurement

    args: 
        particle: the particle to update
        landmark: the old_landmark to update, ((x, y), covariance)
        measurement: the new measurement to `landmark`, 
            in the form of (distance, bearing)
        
    returns: 

        None
    """
    robot_x, robot_y, robot_theta = particle.robot_position
    distance, bearing = measurement

    landmark_pos = landmark[0]
    oldCovariance = landmark[1]
    landmark_x = landmark_pos[0]
    landmark_y = landmark_pos[1]
    x_dif = landmark_x-robot_x
    y_dif = landmark_y-robot_y
    #calculate the predicted measurement using landmark and robot posistions
    m_r = math.sqrt(x_dif*x_dif + y_dif*y_dif)
    m_phi = math.atan2(y_dif, x_dif)
    predicted_measurement = (m_r, m_phi)

    jacobian = utils.calculate_jacobian((robot_x, robot_y), (landmark_x, landmark_y))
    Q = utils.compute_measurement_covariance(jacobian, oldCovariance, sigma_observation)
    K = utils.compute_kalman_gain(jacobian, oldCovariance, Q)

    new_landmark = utils.compute_new_landmark(measurement, predicted_measurement, K, landmark_pos)

    new_cov = utils.compute_new_covariance(K, jacobian, oldCovariance)

    #remove the landmark from the particle's list of landmarks then change landmark and add it back
    particle.landmarks.remove(landmark)
    landmark = (new_landmark, new_cov)
    particle.landmarks.append(landmark)

def update_map(particle, measurements): 

    """ associates new measurements with old landmarks

    given a list of measurements to landmarks, determine whether a 
    landmark is a new landmark or a re-observed landmark. If it is
    a new one, call add_landmark(). If it is an existing landmark, 
    call update_landmark(), and update the weight. 
    
    
    traditionally done maximizing the likelihood of observation at
    that particular correspondance

    args: 
        particle: the particle to perform the data association on
        measurements: a list of (distance, bearing) where a landmark was observed

    returns: 
        index pairs? none if not matched

    """
    # retrieve the newly sampled robot pos
    robot_x,robot_y,robot_theta = particle.robot_position

    
    #some measurements were [] so need to loop through and only work with the valid ones
    valid_measurements = []
    for m in measurements:
        if len(m)>0:
            valid_measurements.append(m)

    weight_as_log_sum = 0
    #loop through non-empty measurements to find the best landmark and best probability
    for measurement in valid_measurements:
        best_landmark = None
        best_prob = 0
        for landmark in particle.landmarks:
            landmark_pos = landmark[0]
            oldCovariance = landmark[1]
            landmark_x, landmark_y = landmark_pos
            x_dif = landmark_x-robot_x
            y_dif = landmark_y-robot_y
            m_r = math.sqrt(x_dif*x_dif + y_dif*y_dif)
            m_phi = math.atan2(y_dif,x_dif)-robot_theta
            #calculate where the landmark should be 
            predicted_measurement = (m_r, m_phi)

            jacobian = utils.calculate_jacobian((robot_x, robot_y), (landmark_x, landmark_y))
            Q = utils.compute_measurement_covariance(jacobian, oldCovariance, sigma_observation)
            likelihood = utils.calculate_likelihood_probability(measurement, predicted_measurement, Q)
            if likelihood > best_prob:
                best_prob = likelihood
                best_landmark = landmark
        #if the landmark is likely to be a new landmark then add it 
        if best_prob < prob_threshold:
            add_landmark(particle, measurement)
            weight_as_log_sum += np.log(prob_threshold)
        #otherwise if it is likely the same as an observed landmark, update it
        else: 
            update_landmark(particle, best_landmark, measurement)
            weight_as_log_sum += np.log(best_prob)

    #update the particles weight, will take e^weight later
    particle.weight = weight_as_log_sum  




def resample_particles(particles): 
    """ resample particles according to weight

    Sample (with replacement) from the list of particles 
    according to their weight, which was assigned in the 
    update_map() section. Be sure to copy each particle

    args: A transformation for extracting new descriptors of shape H Blum, Models for the perception of speech and visual form, 1967
        particles: a list of particles to sample from

    returns: 
        a list of particles of the same length as the 
        input, sampled according to the particle's weight
    """

    num_particles = len(particles)
    
    #find the sum of the particle weights to normalize vector
    particle_weight_sum = 0
    for particle in particles:
        particle_weight_sum += np.exp(particle.weight)


    #normalize particle_weights
    norm_particle_weights = []
    for i in range (0, num_particles):
        norm_particle_weights.append(np.exp(particles[i].weight)/particle_weight_sum)
    
    #sample_results will be a list where the ith element represetns how many time the ith particle
    #was choosen to be in the new sample
    sample_results = list(np.random.multinomial(num_particles, norm_particle_weights))

    new_particles = []
    #loop though all the particles, adding it to new_particles the number of times specified by sample_results
    for k in range(0,num_particles):
        for i in range (0, sample_results[k]):
            to_add = copy.deepcopy(particles[k])
            to_add.weight = 0
            new_particles.append(to_add)

    return new_particles


def SLAM(init_particles, controls, measurements, true_position, true_landmarks, lidar_map, vis=1): 
    """ Runs the fastSLAM algorithm on the given inputs

    Args: 
        init_particles: an Nx3 array of (x-coordinate, y-coordinate, phi), the initial candidate points.
        controls: a Tx2 array of (degree change, velocity), the movement at each time step.
        measurements: a TxMx2 array of (distance, angle), the measured distance to a landmark in that direction at that timestep.
        true_position - a (T+1)x3 array representing the true position of the robot from start to end
        true_landmarks - a TxK array of landmarks representing the true locations of the landmarks
        lidar_map: an XxY array representing a discretized map. Each position in the array represents whetehr a space is empty, represented by a 0, or filled (an obstacle) represented by a 1. Generated by calling utils.load_map on the appropriate .pgm file.
	vis: a flag dictating whether to run the visualizer or not. You should NOT make any changes that result	in the visualizer being called when vis=False
        
    Returns: 
        None
    """

    if vis > 0:
        visualizer.init_vis(lidar_map, true_landmarks, init_particles, true_position[0,:], vis)
    particles = init_particles

    #for a timestep t
    for t in range(len(measurements)): 
        
        #update the particles by calling update_robot to get the new particle for every particle
        #then for every particle call update map with the measurement at time t
        new_particles = []
        for particle in particles:
            new_particle = update_robot(particle, controls[t])
            new_particles.append(new_particle)
            update_map(new_particle, measurements[t])
        
        
        if vis > 0:
            visualizer.update_vis(lidar_map, true_landmarks, new_particles, true_position[t+1,:], vis)

        #resample the particles
        particles = resample_particles(new_particles)
        

def main(): 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help="A test directory containing map.pgm, measure.csv, control.csv, and ground.csv files", required=True)
    parser.add_argument('-v', '--visualizer', type=int, default=0, help='0 : off, 1 : default visualizer, 2 : particle visualizer')
    parser.add_argument('-n', '--numstart', type=int, default=50)
    parser.add_argument('-e', '--error', type=float, default=0.0)

    args = parser.parse_args()
    lmap = utils.load_map('tests/' + args.test + '/map.pgm')
    landmarks = utils.load_csv('tests/' + args.test + '/landmarks.csv')
    controls = utils.load_csv('tests/' + args.test + '/control.csv') #a Tx2 array of T (delta phi, velocity)'s
    measurements = [[np.random.multivariate_normal((distance, bearing), args.error*sigma_observation) for distance, bearing in measurement] for measurement in utils.load_measurements('tests/' + args.test + '/measure.csv')]
    true_start = utils.load_csv('tests/' + args.test + '/ground.csv')
    start_posns = generate_initial_particles(lmap, args.numstart, true_start[0])
    if debug:
        print("Running SLAM...")
    SLAM(start_posns, controls, measurements, true_start, landmarks, lmap, args.visualizer)

if __name__ == "__main__": 
    main()
