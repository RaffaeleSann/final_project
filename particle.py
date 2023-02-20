import numpy as np

class Particle:

    def __init__(self, position, velocity ):
        '''position and velocity are vectors (numpy array)'''

        self.position = position
        self.velocity = velocity
        self.bestp = position

        self.iteration = 0


    def FitnessCalculator(self, position, accuracy):
        '''It takes as input the model and the parameter (which isthe particle position).
         Calculates the accuracy (or loss, we need to decide) of the model and return it. '''

        # Ho messo model.accuracy, poi quando definiamo il modello credo qui vada messa la parte proprio del training per ritornare poi l'accuracy o la loss in base a come vogliamo fare noi.

        self.fitness = accuracy(position)
        return self

    def inertia_coefficient(self, c1, c2, random_1, random_2, max_iter = None, old_w = None, schedule_type = 'constant'):
        
        min_w = (c1+c2)*(1/2)-1
        
        if schedule_type == 'costant':
            # w is set to be constant
            w = c1*random_1 + c2*random_2

        if schedule_type == 'random':
            # w is randomly selected ad each iteration from a gaussian distribution with center 0.72 and σ small enough to ensure that w is not predominantly greater than one
            w = np.random.normal(0.72, 0.4)

        if schedule_type == 'linearly decreasing':
            if max_iter != None:
                w = ((0.9-min_w)*(max_iter-self.iteration)/max_iter) + min_w
            else: 
                raise Exception('ERROR YOU MUST SPECIFY THE MAXIMUM NUMBER OF ITERATION')
        
        if schedule_type == 'nonlinearly decreasing':
            if old_w != None:
                w = 0.975*old_w
            else: 
                raise Exception('ERROR YOU MUST SPECIFY W AT PREVIOUS ITERATION')
        
        if schedule_type not in ['constant','random','linearly decreasing', 'nonlinearly decreasing']:
            raise Exception('You must specify a valid type for w')

        # w > min_w guarantees convergent particle trajectories. If this condition is not satisfied, divergent or cyclic behavior may occur.
        if (w < min_w).any:
            w = min_w
        return w


    def VelocityCalculator(self, c1, c2, best_glob_pos, w_schedule, w = 0.9, v_max = None):

        random_1 = np.random.random(len(self.position))
        random_2 = np.random.random(len(self.position))

        velocity = w*self.velocity + c1*random_1*(self.bestp - self.position) + c2*random_2*(best_glob_pos - self.position)

        # velocity quickly explodes to large values, especially for particles far from the neighborhood best and personal best positions. Consequently, particles have large position updates, which result in particles leaving the boundaries of the search space – the particles diverge. To control the global exploration of particles, velocities are clamped to stay within boundary constraints.

        if v_max != None:
            new_velocity = np.zeros(len(velocity)) #I inizialize an array with all zeros and then I change the elements

            for i in range (len(velocity)):
                if velocity[i] > v_max[i]:
                    new_velocity[i] = v_max[i]
                else:
                    new_velocity[i] = velocity[i]
            self.velocity = new_velocity

        else:
            self.velocity = velocity
        
        #let's update w
        w = self.inertia_coefficient(c1, c2, random_1, random_2, old_w = w, schedule_type = w_schedule)

        return self


    def PositionCalculator(self, new_vel):
        
        self.iteration += 1

        self.position = self.position + new_vel
        return self

    def BestLocal(self, problem):
        '''Takes as input the particle and the type of optimization problem (problem could be minimum or maximum) and calculates best fitness and best position'''
        if self.iteration == 0:
            self.bestfit = self.fitness

        if problem == 'minimum':
            if self.fitness < self.bestfit:
                self.bestfit = self.fitness
                self.bestp = self.position
        elif problem == 'maximum':
            if self.fitness > self.bestfit:
                self.bestfit = self.fitness
                self.bestp = self.position
        else:
            return "Error! problem must be: 'minimum' or 'maximum'"    
        return self