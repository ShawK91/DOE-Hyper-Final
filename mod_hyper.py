from random import randint
import math
from scipy.special import expit
import os, cPickle
#import tensorflow as tf
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, torch
from copy import deepcopy
import torch.nn.functional as F
from matplotlib import pyplot as plt
plt.switch_backend('Qt4Agg')
#plt.switch_backend('TKAgg')





###########TEST###########

class PT_GRUMB(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, output_activation):
        super(PT_GRUMB, self).__init__()

        self.input_size = input_size; self.hidden_size = hidden_size; self.memory_size = memory_size; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None


        #Input gate
        self.w_inpgate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( memory_size, output_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Block Input
        self.w_inp = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(memory_size, output_size), requires_grad=1)

        #Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Output weights
        self.w_hid_out = Parameter(torch.rand(output_size, memory_size), requires_grad=1)

        #Biases
        self.w_input_gate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_block_input_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_readgate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)
        self.w_writegate_bias = Parameter(torch.zeros(memory_size, 1), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            #torch.nn.init.orthogonal(param)
            #torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def graph_compute(self, input, rec_output, mem):
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))# + self.w_block_input_bias)
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(
            rec_output))# + self.w_input_gate_bias)
        inp_out = block_inp * inp_gate

        mem_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(mem))# + self.w_readgate_bias) * mem

        hidden_act = mem_out + inp_out

        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(rec_output))# + self.w_writegate_bias)
        mem = mem + write_gate_out * F.tanh(hidden_act)

        output = self.w_hid_out.mm(hidden_act)
        if self.output_activation != None: output = self.output_activation(output)

        return output, mem

    def forward(self, input):
        self.out, self.mem = self.graph_compute(input, self.out, self.mem)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

    def to_fast_net(self):
        self.reset(1)
        keys = self.state_dict().keys()  # Get all keys
        params = self.state_dict()  # Self params
        fast_net_params = self.fast_net.param_dict  # Fast Net params

        for key in keys:
            fast_net_params[key][:] = params[key].cpu().numpy()

    def from_fast_net(self):
        keys = self.state_dict().keys() #Get all keys
        params = self.state_dict() #Self params
        fast_net_params = self.fast_net.param_dict #Fast Net params

        for key in keys:
            params[key][:] = torch.from_numpy(fast_net_params[key])

class SSNE:
    def __init__(self, parameters):
        self.parameters = parameters;
        self.population_size = self.parameters.pop_size;
        self.num_elitists = int(self.parameters.elite_fraction * parameters.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1
        self.num_input = self.parameters.num_input;
        self.num_hidden = self.parameters.num_hnodes;
        self.num_output = self.parameters.num_output

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight):
        if weight > self.parameters.weight_magnitude_limit:
            weight = self.parameters.weight_magnitude_limit
        if weight < -self.parameters.weight_magnitude_limit:
            weight = -self.parameters.weight_magnitude_limit
        return weight

    def crossover_inplace(self, gene_1, gene_2):

        keys = list(gene_1.param_dict.keys())

        # References to the variable tensors
        W1 = gene_1.param_dict
        W2 = gene_2.param_dict
        num_variables = len(W1)
        if num_variables != len(W2): print 'Warning: Genes for crossover might be incompatible'

        # Crossover opertation [Indexed by column, not rows]
        num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = fastrand.pcg32bounded(num_variables)  # Choose which tensor to perturb
            receiver_choice = random.random()  # Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = fastrand.pcg32bounded(W1[keys[tensor_choice]].shape[-1])  #
                W1[keys[tensor_choice]][:, ind_cr] = W2[keys[tensor_choice]][:, ind_cr]
                #W1[keys[tensor_choice]][ind_cr, :] = W2[keys[tensor_choice]][ind_cr, :]
            else:
                ind_cr = fastrand.pcg32bounded(W2[keys[tensor_choice]].shape[-1])  #
                W2[keys[tensor_choice]][:, ind_cr] = W1[keys[tensor_choice]][:, ind_cr]
                #W2[keys[tensor_choice]][ind_cr, :] = W1[keys[tensor_choice]][ind_cr, :]

    def mutate_inplace(self, gene):
        mut_strength = 0.2
        num_mutation_frac = 0.2
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        # References to the variable keys
        keys = list(gene.param_dict.keys())
        W = gene.param_dict
        num_structures = len(keys)
        ssne_probabilities = np.random.uniform(0,1,num_structures)*2

        for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure
            if random.random()<ssne_prob:

                num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                for _ in range(num_mutations):
                    ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                                                                                      W[key][
                                                                                          ind_dim1, ind_dim2])
                    elif random_num < reset_prob:  # Reset probability
                        W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                    else:  # mutauion even normal
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                                                                                          ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                        W[key][ind_dim1, ind_dim2])

    def copy_individual(self, master, replacee):  # Replace the replacee individual with master
            keys = master.param_dict.keys()
            for key in keys:
                replacee.param_dict[key][:] = master.param_dict[key]

    def reset_genome(self, gene):
            keys = gene.param_dict
            for key in keys:
                dim = gene.param_dict[key].shape
                gene.param_dict[key][:] = np.mat(np.random.uniform(-1, 1, (dim[0], dim[1])))

    def epoch(self, pop, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        #Extinction step (Resets all the offsprings genes; preserves the elitists)
        if random.random() < self.parameters.extinction_prob: #An extinction event
            print
            print "######################Extinction Event Triggered#######################"
            print
            for i in offsprings:
                if random.random() < self.parameters.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
                    self.reset_genome(pop[i])

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=pop[i], replacee=pop[replacee])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.copy_individual(master=pop[off_i], replacee=pop[i])
            self.copy_individual(master=pop[off_j], replacee=pop[j])
            self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.parameters.crossover_prob: self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists plus homozenize
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.parameters.mutation_prob: self.mutate_inplace(pop[i])



def simulator_test_perfect(model, filename = 'ColdAir.csv', downsample_rate=25):
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

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

    print ('TESTING NOW')
    input = np.reshape(data[0], (1, 21))  # First input to the simulatior
    track_target = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))
    track_output = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))

    for example in range(len(data)-1):  # For all training examples
        model_out = model.predict(input)

        # Track index
        for index in range(19):
            track_output[index][example] = model_out[0][index]# * normalizer[index] + min[index]
            track_target[index][example] = data[example+1][index]# * normalizer[index] + min[index]

        # Fill in new input data
        for k in range(len(model_out)):
            input[k] = input = np.reshape(data[example+1], (1, 21))
        # Fill in two control variables
        input[0][19] = data[example + 1][19]
        input[0][20] = data[example + 1][20]


    for index in range(19):
        plt.plot(track_target[index], 'r--',label='Actual Data: ' + str(index))
        plt.plot(track_output[index], 'b-',label='TF_Simulator: ' + str(index))
        #np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        #np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend( loc='upper right',prop={'size':6})
        #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        #print track_output[index]
        plt.show()

def simulator_results(model, filename = 'ColdAir.csv', downsample_rate=25):


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

    print ('TESTING NOW')
    input = np.reshape(data[0], (1, 21))  # First input to the simulatior
    track_target = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))
    track_output = np.reshape(np.zeros((len(data) - 1) * 19), (19, len(data) - 1))

    for example in range(len(data)-1):  # For all training examples
        model_out = model.predict(input)

        # Track index
        for index in range(19):
            track_output[index][example] = model_out[0][index]# * normalizer[index] + min[index]
            track_target[index][example] = data[example+1][index]# * normalizer[index] + min[index]

        # Fill in new input data
        for k in range(len(model_out[0])):
            input[0][k] = model_out[0][k]
        # Fill in two control variables
        input[0][19] = data[example + 1][19]
        input[0][20] = data[example + 1][20]


    for index in range(19):
        x = [1,2,3]; y = [0,0,0]
        plt.plot(x, y)
        plt.show()
        plt.plot(track_target[index], 'r--',label='Actual Data: ' + str(index))
        plt.plot(track_output[index], 'b-',label='TF_Simulator: ' + str(index))
        #np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
        #np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
        plt.legend( loc='upper right',prop={'size':6})
        #plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
        #print track_output[index]
        plt.show()

def controller_results(individual, setpoints, start_controller_input, start_sim_input, simulator):  # Controller fitness
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    control_input = np.copy(start_controller_input)  # Input to the controller
    sim_input = np.copy(start_sim_input)  # Input to the simulator
    track_output = np.reshape(np.zeros(len(setpoints) - 1), (len(setpoints) - 1, 1))

    for example in range(len(setpoints) - 1):  # For all training examples
        # Fill in the setpoint to control input
        control_input[0][-1] = setpoints[example][0]

        # # Add noise to the state input to the controller
        # if self.parameters.sensor_noise != 0:  # Add sensor noise
        #     for i in range(19):
        #         std = self.parameters.sensor_noise * abs(noise_input[0][i]) / 100.0
        #         if std != 0:
        #             noise_input[0][i] += np.random.normal(0, std / 2.0)
        #
        # if self.parameters.sensor_failure != None:  # Failed sensor outputs 0 regardless
        #     for i in self.parameters.sensor_failure:
        #         noise_input[0][i] = 0
        #

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
        control_out = individual.predict(control_input)
        #
        # # Add actuator noise (controls)
        # if self.parameters.actuator_noise != 0:
        #     for i in range(len(control_out[0])):
        #         std = self.parameters.actuator_noise * abs(control_out[0][i]) / 100.0
        #         if std != 0:
        #             control_out[0][i] += np.random.normal(0, std / 2.0)
        #
        # if self.parameters.actuator_failure != None:  # Failed actuator outputs 0 regardless
        #     for i in self.parameters.actuator_failure:
        #         control_out[0][i] = 0


        # Fill in the controls
        sim_input[0][19] = control_out[0][0]
        sim_input[0][20] = control_out[0][1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output[example][0] = simulator_out[0][11]
        #weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        for i in range(simulator_out.shape[-1]):
            sim_input[0][i] = simulator_out[0][i]
            control_input[0][i] = simulator_out[0][i]

        #decorator = np.reshape(np.arange(len(setpoints) - 1) + 1, (len(setpoints) - 1, 1))
        #setpoints = np.array(setpoints[0:-1])
        #setpoints = np.concatenate((decorator, setpoints))
        #track_output = np.concatenate((decorator, track_output))

    plt.plot(setpoints, 'r--', label='Desired Turbine Speed' )
    plt.plot(track_output, 'b-', label='Achieved Turbine Speed')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='upper right', prop={'size': 15})
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()

def pt_controller_results(individual, setpoints, start_controller_input, start_sim_input,
                           simulator, sensor_noise, actuator_noise, sensor_failure):  # Controller fitness
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    control_input = np.copy(start_controller_input)  # Input to the controller
    sim_input = np.copy(start_sim_input)  # Input to the simulator
    track_output = np.reshape(np.zeros(len(setpoints) - 1), (len(setpoints) - 1, 1))
    individual.fast_net.reset()

    for example in range(len(setpoints) - 1):  # For all training examples
        # Fill in the setpoint to control input
        control_input[0][-1] = setpoints[example][0]

        # Add noise to the state input to the controller
        if sensor_noise != 0:  # Add sensor noise
            for i in range(19):
                std = sensor_noise * abs(control_input[0][i])
                if std != 0:
                    control_input[0][i] += np.random.normal(0, std)

        if sensor_failure != None:  # Failed sensor outputs 0 regardless
            if random.random() < sensor_failure:
                control_input[0][11] = 0.0

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
        control_out = individual.fast_net.predict(control_input)
        #
        # Add actuator noise (controls)
        if actuator_noise != 0:
            for i in range(len(control_out[0])):
                std = actuator_noise * abs(control_out[0][i])
                if std != 0:
                    control_out[0][i] += np.random.normal(0, std)



        # Fill in the controls
        sim_input[0][19] = control_out[0][0]
        sim_input[0][20] = control_out[0][1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output[example][0] = simulator_out[0][11]
        # weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        for i in range(simulator_out.shape[-1]):
            sim_input[0][i] = simulator_out[0][i]
            control_input[0][i] = simulator_out[0][i]

            # decorator = np.reshape(np.arange(len(setpoints) - 1) + 1, (len(setpoints) - 1, 1))
            # setpoints = np.array(setpoints[0:-1])
            # setpoints = np.concatenate((decorator, setpoints))
            # track_output = np.concatenate((decorator, track_output))

    plt.plot(setpoints, 'r--', label='Desired Turbine Speed')
    plt.plot(track_output, 'b-', label='Achieved Turbine Speed')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='upper right', prop={'size': 15})
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()

def controller_results_bprop(individual, setpoints, start_controller_input, start_sim_input, simulator, train_x):  # Controller fitness
    from matplotlib import pyplot as plt
    plt.switch_backend('Qt4Agg')

    #Normalizer #TODO DONT DO THIS
    if True:
        # Import training data and clear away the two top lines
        downsample_rate = 25
        data = np.loadtxt('ColdAir.csv', delimiter=',', skiprows=2)

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

        control_input = np.copy(start_controller_input)  # Input to the controller
        sim_input = np.copy(start_sim_input)  # Input to the simulator
        track_output = np.reshape(np.zeros(len(setpoints) - 1), (len(setpoints) - 1, 1))

    for example in range(len(setpoints) - 1):  # For all training examples
        # Fill in the setpoint to control input
        control_input[0][-1] = setpoints[example][0]

        # # Add noise to the state input to the controller
        # if self.parameters.sensor_noise != 0:  # Add sensor noise
        #     for i in range(19):
        #         std = self.parameters.sensor_noise * abs(noise_input[0][i]) / 100.0
        #         if std != 0:
        #             noise_input[0][i] += np.random.normal(0, std / 2.0)
        #
        # if self.parameters.sensor_failure != None:  # Failed sensor outputs 0 regardless
        #     for i in self.parameters.sensor_failure:
        #         noise_input[0][i] = 0
        #

        # RUN THE CONTROLLER TO GET CONTROL OUTPUT
        control_out = individual.predict(control_input)

        #
        # # Add actuator noise (controls)
        # if self.parameters.actuator_noise != 0:
        #     for i in range(len(control_out[0])):
        #         std = self.parameters.actuator_noise * abs(control_out[0][i]) / 100.0
        #         if std != 0:
        #             control_out[0][i] += np.random.normal(0, std / 2.0)
        #
        # if self.parameters.actuator_failure != None:  # Failed actuator outputs 0 regardless
        #     for i in self.parameters.actuator_failure:
        #         control_out[0][i] = 0


        # Fill in the controls
        sim_input[0][19] = control_out[0][0]
        sim_input[0][20] = control_out[0][1]

        # Use the simulator to get the next state
        simulator_out = simulator.predict(sim_input)

        # Calculate error (weakness)
        track_output[example][0] = simulator_out[0][11] * normalizer[11] + min[11]
        setpoints[example][0] = setpoints[example][0] * normalizer[11] + min[11]
        #weakness += math.fabs(simulator_out[0][self.parameters.target_sensor] - setpoints[example][0])  # Time variant simulation

        # Fill in the simulator inputs and control inputs
        for i in range(simulator_out.shape[-1]):
            sim_input[0][i] = train_x[example+1][i]
            control_input[0][i] = train_x[example+1][i]

            #sim_input[0][i] = simulator_out[0][i]
            #control_input[0][i] = simulator_out[0][i]



        #decorator = np.reshape(np.arange(len(setpoints) - 1) + 1, (len(setpoints) - 1, 1))
        #setpoints = np.array(setpoints[0:-1])
        #setpoints = np.concatenate((decorator, setpoints))
        #track_output = np.concatenate((decorator, track_output))

    plt.plot(setpoints[0:-10,0:], 'r--', label='Target Turbine Speed' )
    plt.plot(track_output[0:-10,0:], 'b-', label='Controller Turbine Speed ')
    # np.savetxt('R_Simulator/output_' + str(index) + '.csv', track_output[index])
    # np.savetxt('R_Simulator/target_' + str(index) + '.csv', track_target[index])
    plt.legend(loc='lower right', prop={'size': 15})
    plt.xlabel("Time (min)", fontsize = 15)
    plt.ylabel("ST-502 (Turbine Speed)", fontsize=15)
    # plt.savefig('Graphs/' + 'Index' + str(index) + '.png')
    # print track_output[index]
    plt.show()






def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)




import fastrand
import math
import  cPickle
import random
import numpy as np
from scipy.special import expit
#from scipy.sparse import random as scipy_rand








