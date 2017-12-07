import mod_hyper as mod, math
from scipy.special import expit
from matplotlib import pyplot as plt
import numpy as np, os
from random import randint
from torch.autograd import Variable
import torch
import random
from torch.utils import data as util
plt.switch_backend('Qt4Agg')


class Tracker(): #Tracker
    def __init__(self, parameters):
        self.foldername = parameters.save_foldername + '/0000_CSV'
        self.fitnesses = []; self.avg_fitness = 0; self.tr_avg_fit = []
        self.hof_fitnesses = []; self.hof_avg_fitness = 0; self.hof_tr_avg_fit = []
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.file_save = 'Controller.csv'

    def add_fitness(self, fitness, generation):
        self.fitnesses.append(fitness)
        if len(self.fitnesses) > 100:
            self.fitnesses.pop(0)
        self.avg_fitness = sum(self.fitnesses)/len(self.fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/train_' + self.file_save
            self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
            np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

    def add_hof_fitness(self, hof_fitness, generation):
        self.hof_fitnesses.append(hof_fitness)
        if len(self.hof_fitnesses) > 100:
            self.hof_fitnesses.pop(0)
        self.hof_avg_fitness = sum(self.hof_fitnesses)/len(self.hof_fitnesses)
        if generation % 10 == 0: #Save to csv file
            filename = self.foldername + '/valid_' + self.file_save
            self.hof_tr_avg_fit.append(np.array([generation, self.hof_avg_fitness]))
            np.savetxt(filename, np.array(self.hof_tr_avg_fit), fmt='%.3f', delimiter=',')

    def save_csv(self, generation, filename):
        self.tr_avg_fit.append(np.array([generation, self.avg_fitness]))
        np.savetxt(filename, np.array(self.tr_avg_fit), fmt='%.3f', delimiter=',')

class Parameters:
    def __init__(self):
            self.pop_size = 100
            self.load_seed = False #Loads a seed population from the save_foldername
                                  # IF FALSE: Runs Backpropagation, saves it and uses that
            # Determine the nerual archiecture
            self.arch_type = 2
            self. output_activation = None

            #Controller choices
            self.target_sensor = 11 #Turbine speed the sensor to control
            self.run_time = 300 #Controller Run time

            #Controller noise
            self.sensor_noise = 0.1
            self.sensor_failure = 0.0
            self.actuator_noise = 0.0

            # Reconfigurability parameters
            self.is_random_initial_state = True  # Start state of controller
            self.num_profiles = 3
            self.reconf_shape = 2 #1 Periodic shape, #2 Mimicking real shape

            #GD Stuff
            self.total_epochs = 12
            self.batch_size = 100

            #SSNE stuff
            self.num_input = 20
            self.num_hnodes = 20
            self.num_mem = self.num_hnodes
            self.num_output = 2
            self.elite_fraction = 0.1
            self.crossover_prob = 0.1
            self.mutation_prob = 0.75
            self.weight_magnitude_limit = 10000000
            self.extinction_prob = 0.004  # Probability of extinction event
            self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
            self.mut_distribution = 3  # 1-Gaussian, 2-Laplace, 3-Uniform, ELSE-all 1s
            self.total_gens = 100000
            self.num_evals = 7 #Number of independent evaluations before getting a fitness score

            if self.arch_type == 1: self.arch_type = 'FF'
            elif self.arch_type == 2: self.arch_type = 'GRU-MB'
            self.save_foldername = 'R_Reconfigurable_Controller/'

class Fast_Simulator(): #TF Simulator individual (One complete simulator genome)
    def __init__(self):
        self.W = None

    def predict(self, input):
        # Feedforward operation
        h_1 = expit(np.dot(input, self.W[0]) + self.W[1])
        return np.dot(h_1, self.W[2]) + self.W[3]

    def from_tf(self, tf_sess):
        self.W = tf_sess.run(tf.trainable_variables())

class Task_Controller: #Reconfigurable Control Task
    def __init__(self, parameters):
        self.parameters = parameters
        self.num_input = parameters.num_input; self.num_hidden = parameters.num_hnodes; self.num_output = parameters.num_output

        self.train_data, self.valid_data = self.data_preprocess() #Get simulator data
        self.ssne = mod.SSNE(parameters) #Initialize SSNE engine

        # Save folder for checkpoints
        self.marker = 'TF_ANN'
        self.save_foldername = self.parameters.save_foldername
        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)

        #Load simulator
        self.simulator = mod.unpickle('Champion_Simulator')
        #mod.simulator_results(self.simulator)


        #####Create Reconfigurable controller population
        self.pop = []
        for i in range(self.parameters.pop_size):
            # Choose architecture
            self.pop.append(mod.PT_GRUMB(self.num_input, self.num_hidden, parameters.num_mem, self.num_output,
                                             output_activation=self.parameters.output_activation))




        ###Initialize Controller Population
        if self.parameters.load_seed: #Load seed population
            self.pop[0] = mod.unpickle('R_Controller/seed_controller') #Load PT_GRUMB object
        else: #Run Backprop
            self.run_bprop(self.pop[0])
        self.pop[0].to_fast_net()  # transcribe neurosphere to its Fast_Net

    def save(self, individual, filename ):
        mod.pickle_object(individual, filename)

    def predict(self, individual, input): #Runs the individual net and computes and output by feedforwarding
        return individual.predict(input)


    def run_bprop(self, model):
        all_train_x = self.train_data[0:-1,0:-2]
        sensor_target = self.train_data[1:,self.parameters.target_sensor:self.parameters.target_sensor+1]
        all_train_x = np.concatenate((all_train_x, sensor_target), axis=1) #Input training data
        all_train_y = self.train_data[0:-1,-2:] #Target Controller Output
        # criterion = torch.nn.L1Loss(False)
        #criterion = torch.nn.SmoothL1Loss(False)
        # criterion = torch.nn.KLDivLoss()
        criterion = torch.nn.MSELoss()
        # criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.1)
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum = 0.5, nesterov = True)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.005, momentum=0.1)

        seq_len = 1
        #eval_train_y = all_train_y[:]  # Copy just the list to evaluate batch
        all_train_x = torch.Tensor(all_train_x).cuda(); all_train_y = torch.Tensor(all_train_y).cuda()
        #eval_train_x = all_train_x[:]  # Copy tensor to evaluate batch
        train_dataset = util.TensorDataset(all_train_x, all_train_y)
        train_loader = util.DataLoader(train_dataset, batch_size=self.parameters.batch_size, shuffle=True)
        model.cuda()
        for epoch in range(1, self.parameters.total_epochs + 1):

            epoch_loss = 0.0
            for data in train_loader:  # Each Batch
                net_inputs, targets = data

                model.reset(len(net_inputs))  # Reset memory and recurrent out for the model
                for i in range(seq_len):  # For the length of the sequence
                    #net_inp = Variable(net_inputs[:, i], requires_grad=True).unsqueeze(0)
                    net_inp = Variable(net_inputs, requires_grad=True)
                    net_inp = torch.t(net_inp)
                    net_out = model.forward(net_inp)
                    target_T = Variable(targets)
                    loss = criterion(net_out, target_T)
                    loss.backward(retain_variables=True)
                    epoch_loss += loss.cpu().data.numpy()[0]

            optimizer.step()  # Perform the gradient updates to weights for the entire set of collected gradients
            optimizer.zero_grad()

            if epoch % 1 == 0:
                #test_x = torch.Tensor(test_x).cuda()
                #train_fitness = self.batch_evaluate(self.model, eval_train_x, eval_train_y)
                #valid_fitness = self.batch_evaluate(self.model, test_x, test_y)
                print 'Epoch: ', epoch, ' Loss: ', epoch_loss
                #print ' Train_Performance:', "%0.2f" % train_fitness,
                #print ' Valid_Performance:', "%0.2f" % valid_fitness
                #tracker.update([epoch_loss, train_fitness, valid_fitness], epoch)
                #torch.save(self.model, self.save_foldername + 'seq_classifier_net')





    def plot_controller(self, individual, data_setpoints=False):
        setpoints = self.get_setpoints()
        if self.parameters.is_random_initial_state:
            start_sim_input = np.copy(self.train_data[randint(0, len(self.train_data))])
        else:
            start_sim_input = np.reshape(np.copy(self.train_data[0]), (1, len(self.train_data[0])))
        start_controller_input = np.reshape(np.zeros(self.ssne_param.num_input), (1, self.ssne_param.num_input))
        for i in range(start_sim_input.shape[-1] - 2): start_controller_input[0][i] = start_sim_input[0][i]

        if data_setpoints: #Bprop test
            setpoints = self.train_data[0:, 11:12]
            mod.controller_results_bprop(individual, setpoints, start_controller_input, start_sim_input, self.simulator, self.train_data[0:,0:-2])
        else:
            mod.controller_results(individual, setpoints, start_controller_input, start_sim_input, self.simulator)

    def get_setpoints(self):
        if self.parameters.reconf_shape == 1:
            desired_setpoints = np.reshape(np.zeros(self.parameters.run_time), (parameters.run_time, 1))
            for profile in range(parameters.num_profiles):
                multiplier = randint(1, 5)
                #print profile, multiplier
                for i in range(self.parameters.run_time/self.parameters.num_profiles):
                    turbine_speed = math.sin(i * 0.2 * multiplier)
                    turbine_speed *= 0.3 #Between -0.3 and 0.3
                    turbine_speed += 0.5  #Between 0.2 and 0.8 centered on 0.5
                    desired_setpoints[profile * self.parameters.run_time/self.parameters.num_profiles + i][0] = turbine_speed

        elif self.parameters.reconf_shape == 2:
            desired_setpoints = np.reshape(np.zeros(self.parameters.run_time), (parameters.run_time, 1)) + random.uniform(0.4, 0.6)
            noise = np.random.uniform(-0.01, 0.01, (parameters.run_time, 1))
            desired_setpoints += noise

            for profile_id in range(self.parameters.num_profiles):
                phase_len = self.parameters.run_time/self.parameters.num_profiles
                phase_start = profile_id * phase_len; phase_end = phase_start + (phase_len)

                start = random.randint(phase_start, phase_end-35)
                end = random.randint(start+10, start + 35)
                magnitude = random.uniform(-0.25, 0.25)

                for i in range(start, end):
                    desired_setpoints[i][0] += magnitude


        plt.plot(desired_setpoints, 'r--', label='Setpoints')
        plt.show()
        sys.exit()
        return desired_setpoints

    def compute_fitness(self, individual, setpoints, start_controller_input, start_sim_input): #Controller fitness
        weakness = 0.0
        individual.fast_net.reset()

        control_input = np.copy(start_controller_input) #Input to the controller
        sim_input = np.copy(start_sim_input) #Input to the simulator

        for example in range(len(setpoints) - 1):  # For all training examples
            # Fill in the setpoint to control input
            control_input[0][-1] = setpoints[example][0]

            # Add noise to the state input to the controller
            if self.parameters.sensor_noise != 0:  # Add sensor noise
                for i in range(19):
                    std = self.parameters.sensor_noise * abs(control_input[0][i])
                    if std != 0:
                        control_input[0][i] += np.random.normal(0, std )

            if self.parameters.sensor_failure != None:  # Failed sensor outputs 0 regardless
                    if random.random() < self.parameters.sensor_failure:
                        control_input[0][11] = 0.0
            #

            #RUN THE CONTROLLER TO GET CONTROL OUTPUT
            control_out = individual.fast_net.predict(control_input)
            #
            # Add actuator noise (controls)
            if self.parameters.actuator_noise != 0:
                for i in range(len(control_out[0])):
                    std = self.parameters.actuator_noise * abs(control_out[0][i])
                    if std != 0:
                        control_out[0][i] += np.random.normal(0, std)

            #Fill in the controls
            sim_input[0][19] = control_out[0][0]
            sim_input[0][20] = control_out[0][1]

            # Use the simulator to get the next state
            simulator_out = self.simulator.predict(sim_input)

            # Calculate error (weakness)
            weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

            # Fill in the simulator inputs and control inputs
            for i in range(simulator_out.shape[-1]):
                sim_input[0][i] = simulator_out[0][i]
                control_input[0][i] = simulator_out[0][i]

        return -weakness

    def evolve(self, gen):

        #Fitness evaluation list for the generation
        fitness_evals = [0.0] * (self.parameters.population_size)

        for eval in range(parameters.num_evals): #Take multiple samples
            #Figure initial position and setpoints for the generation
            setpoints = self.get_setpoints()
            if self.parameters.is_random_initial_state:
                start_sim_input = np.reshape(np.copy(np.copy(self.train_data[randint(0,len(self.train_data)-2)])), (1, len(self.train_data[0])))
            else: start_sim_input = np.reshape(np.copy(self.train_data[0]), (1, len(self.train_data[0])))
            start_controller_input = np.reshape(np.zeros(self.ssne_param.num_input), (1, self.ssne_param.num_input))
            for i in range(start_sim_input.shape[-1]-2):
                start_controller_input[0][i] = start_sim_input[0][i]


            #Test all individuals and assign fitness
            for index, individual in enumerate(self.pop): #Test all genomes/individuals
                fitness = self.compute_fitness(individual, setpoints, start_controller_input, start_sim_input)
                fitness_evals[index] += fitness/(1.0*parameters.num_evals)
        gen_best_fitness = max(fitness_evals)

        #Champion Individual
        champion_index = fitness_evals.index(max(fitness_evals))
        valid_score = 0.0
        for eval in range(parameters.num_evals):  # Take multiple samples
            setpoints = self.get_setpoints()
            if self.parameters.is_random_initial_state: start_sim_input = start_sim_input = np.reshape(np.copy(np.copy(self.valid_data[randint(0,len(self.valid_data)-2)])), (1, len(self.valid_data[0])))
            else: start_sim_input = np.reshape(np.copy(self.valid_data[0]), (1, len(self.valid_data[0])))
            start_controller_input = np.reshape(np.zeros(self.ssne_param.num_input), (1, self.ssne_param.num_input))
            for i in range(start_sim_input.shape[-1]-2): start_controller_input[0][i] = start_sim_input[0][i]
            valid_score += self.compute_fitness(self.pop[champion_index], setpoints, start_controller_input, start_sim_input)/(1.0*parameters.num_evals)


        #Save population and Champion
        if gen % 50 == 0:
            #for index, individual in enumerate(self.pop): #Save population
                #self.save(individual, self.save_foldername + 'Controller_' + str(index))
            self.save(self.pop[champion_index], self.save_foldername + 'Champion_Controller') #Save champion
            np.savetxt(self.save_foldername + '/gen_tag', np.array([gen + 1]), fmt='%.3f', delimiter=',')

        #SSNE Epoch: Selection and Mutation/Crossover step
        self.ssne.epoch(self.pop, fitness_evals)

        return gen_best_fitness, valid_score

    def data_preprocess(self, filename='ColdAir.csv', downsample_rate=25, split = 1000):
        # Import training data and clear away the two top lines
        data = np.loadtxt(filename, delimiter=',', skiprows=2)

        # Splice data (downsample)
        ignore = np.copy(data)
        data = data[0::downsample_rate]
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i != data.shape[0] - 1):
                    data[i][j] = ignore[(i * downsample_rate):(i + 1) * downsample_rate,
                                       j].sum() / downsample_rate
                else:
                    residue = ignore.shape[0] - i * downsample_rate
                    data[i][j] = ignore[(i * downsample_rate):i * downsample_rate + residue, j].sum() / residue

        # Normalize between 0-0.99
        normalizer = np.zeros(data.shape[1])
        min = np.zeros(len(data[0]))
        max = np.zeros(len(data[0]))
        for i in range(len(data[0])):
            min[i] = np.amin(data[:, i])
            max[i] = np.amax(data[:, i])
            normalizer[i] = max[i] - min[i] + 0.00001
            data[:, i] = (data[:, i] - min[i]) / normalizer[i]

        #Train/Valid split
        train_data = data[0:split]
        valid_data = data[split:len(data)]

        return train_data, valid_data

    def test_restore(self, individual):
        train_x = self.train_data[0:-1]
        train_y = self.train_data[1:,0:-2]
        print individual.sess.run(self.cost, feed_dict={self.input: train_x, self.target: train_y})
        self.load(individual, 'Controller_' + str(98))
        print individual.sess.run(self.cost, feed_dict={self.input: train_x, self.target: train_y})

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    tracker = Tracker(parameters)  # Initiate tracker
    print 'Running Reconfigurable Controller Training ', parameters.arch_type

    control_task = Task_Controller(parameters)
    for gen in range(1, parameters.total_gens):
        gen_best_fitness, valid_score = control_task.evolve(gen)
        print 'Generation:', gen, ' Epoch_reward:', "%0.2f" % gen_best_fitness, ' Valid Score:', "%0.2f" % valid_score, '  Cumul_Valid_Score:', "%0.2f" % tracker.hof_avg_fitness
        tracker.add_fitness(gen_best_fitness, gen)  # Add average global performance to tracker
        tracker.add_hof_fitness(valid_score, gen)  # Add best global performance to tracker














