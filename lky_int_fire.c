#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>



struct leakyNeuron{
    double rest_potential, membrane_potential, refactory_potential;
    double threshold;
    double resistance;
    double capacity;
    double current_injected;
    double time_constant;
    int spiked;
};

struct leakyNeuron* create_neuron(double rest_potential, double refactory_potential, double threshold, double resistance, double capacity){
    struct leakyNeuron* neuron = (struct leakyNeuron*)malloc(sizeof(struct leakyNeuron));
    neuron -> rest_potential = rest_potential;
    neuron -> refactory_potential = refactory_potential;
    neuron -> threshold = threshold;
    neuron -> resistance = resistance;
    neuron -> capacity = capacity;
    neuron -> current_injected = 0;
    neuron -> time_constant = resistance * capacity;
    neuron -> spiked = 0;
    return neuron;
}

// Sloppy normal distribution
double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

void write_to_file(double data[], int N){
    FILE *file = fopen("output/neuron_current", "a");
    fprintf(file, "%f", data[0]);
    for (int i=1;i<N;i++){
        fprintf(file, ", %f", data[i]);
    }
    fclose(file);
}

double RandomFloat(float min, float max){
   return ((max - min) * ((double)rand() / RAND_MAX)) + min;
}

double generate_spike(struct leakyNeuron* neuron){
    neuron -> membrane_potential = neuron ->refactory_potential;
    neuron -> spiked = 1;
    return 1.0;
}

double leaky_integrator_model(double membrane_potential, double rest_p, double resistance, double external_current, double time_constant){
    double new_potential = (-(membrane_potential - rest_p) + resistance * external_current)/time_constant;
    // printf("Here is new potential: %f", new_potential);
    return new_potential;
}

double euler_ode_solver(int N, int time_s, int time_e, double membrane_potential, double resistance, double rest_p, 
double external_current, double time_constant, double (*leaky_integrator_model)(double, double, double, double, double)){

    double step_size_h = (double)(time_e-time_s)/N;
    //printf("Stepsize: %f", step_size_h);

    double step_value_old = time_s;
    double step_value = 0;
    double func_aprox_old = membrane_potential;
    double func_aprox = 0;

    for (int i=0;i<N;i++){
        step_value = time_s + step_size_h * i;
        func_aprox = func_aprox_old + step_size_h * leaky_integrator_model(func_aprox_old, rest_p, resistance, external_current, time_constant);
        step_value_old = step_value;
        func_aprox_old = func_aprox;
    }

    return func_aprox;
}

double RK4_solver(int N, int time_s, int time_e, double membrane_potential, double resistance, double rest_p, 
double external_current, double time_constant, double (*leaky_integrator_model)(double, double, double, double, double)){
    double step_size_h = (double)(time_e-time_s)/N;

    double func_aprox_old = membrane_potential;
    double func_aprox = 0;

    for (int i=0;i<N;i++){
        double k1 = leaky_integrator_model(func_aprox_old, rest_p, resistance, external_current, time_constant);
        double k2 = leaky_integrator_model(func_aprox_old + step_size_h*(k1/2), rest_p, resistance, external_current, time_constant);
        double k3 = leaky_integrator_model(func_aprox_old + step_size_h*(k2/2), rest_p, resistance, external_current, time_constant);
        double k4 = leaky_integrator_model(func_aprox_old + step_size_h*k3, rest_p, resistance, external_current, time_constant);
        func_aprox = func_aprox_old + (1.0/6.0) * step_size_h*(k1+2*k2+2*k3+k4);
        //printf("func_aprox: %f \n", func_aprox);
        func_aprox_old = func_aprox;
    }
    return func_aprox;


}

double update_neuron(struct leakyNeuron* neuron, int current_time, double input_current){
    if (neuron -> spiked >= 3){
        neuron -> spiked = 0;
    }
    if(neuron -> spiked > 0){
        neuron -> spiked += 1;
    }
    double new_membrane = RK4_solver(100, current_time, current_time+1, neuron->membrane_potential, neuron->resistance,neuron->rest_potential,
    input_current, neuron->time_constant, leaky_integrator_model);
    neuron -> membrane_potential = new_membrane;
    if (neuron -> membrane_potential > neuron -> threshold){
        neuron -> spiked = 1;
        neuron -> membrane_potential = neuron -> refactory_potential;
        return 1.0;
    }
    //printf("Neuron potential: %f \n", neuron ->membrane_potential);
    //printf("Threshold = %f\n", neuron->threshold);
    return 0.0;
}

void simulate_neuron(int sim_steps){
    struct leakyNeuron* neuron = create_neuron(1.0, 0, 3.0, 2, 10);
    neuron -> membrane_potential = neuron -> rest_potential;
    double current = 0;
    double results[sim_steps];

    for (int i=1; i<sim_steps;i++){
        double membrane_potential = update_neuron(neuron, i, current);
        results[i] = membrane_potential;
        //current = RandomFloat(-0.75,2.5);
        current = sampleNormal()+.8;

    }
    // printf("[%f", results[0]);
    // for (int j=1; j<sim_steps;j++){
    //     printf(", %f",results[j]);
    // }
    // printf("]\n");
    write_to_file(results, sim_steps);
}

void create_weights(int* connection_array, int topology[], int n_layers, int n_neurons){

    int start_index = 0;
    for (int i=0;i<n_layers-1;i++){

        int n_neurons_in_layer = topology[i];
        int n_neurons_connected = topology[i+1];

        for (int j=0;j<n_neurons_in_layer;j++){
            for (int k=0;k<n_neurons_connected;k++){
                connection_array[(start_index+j)*n_neurons + start_index+n_neurons_in_layer+k]=1;
            }
        }
        start_index += n_neurons_in_layer;
    }
    for (int z=0;z<10;z++){
        printf("[%d", connection_array[z*n_neurons]);
        for(int l=1;l<10;l++){
            printf(", %d", connection_array[z*n_neurons+l]);
        }
        printf("]\n");
    }
}

void initialise_neurons(struct leakyNeuron** neurons, int num_neurons){
    for (int i=0;i<num_neurons;i++){
        neurons[i] = create_neuron(1.0, 0, 2.0, 2, 10);
    }
}

void forward_pass(double* input_current, int num_layers, int time_step, int num_neurons,
struct leakyNeuron **neurons, int* connectivity_array, int* spike_outputs, int* topology){
    for (int i=0; i<topology[0]; i++){
        spike_outputs[i]=update_neuron(neurons[i], time_step, input_current[i]);
    }
    int start_index = topology[0];
    // Loop over layers and iteratively update them
    for (int i=1;i<num_layers;i++){
        // Loop over neurons and update them
        for (int j=0;j<topology[i];j++){
            int input_current = 0;
            // Inefficient, lots of zero multiplications
            // Multiply the spike outputs from previous layer with the connectivity, this should be done using a BLAS function
            for (int k=0;k<num_neurons;k++){
                input_current += spike_outputs[k] * connectivity_array[k*num_neurons+start_index+k];
            }
            spike_outputs[start_index + j] = update_neuron(neurons[start_index+j], time_step, input_current);
        }
        start_index += topology[i];
    }
    return spike_outputs;
}

int main( int argc, char ** argv ) {
    
    //struct leakyNeuron* neuron = create_neuron(1.0, 0, 2.0, 2, 10);
    //simulate_neuron(200);

    srand(time(NULL));
    int num_neurons = 10;

    struct leakyNeuron **neurons = malloc(num_neurons * sizeof(struct leakyNeuron));
    int* connectivity_array = calloc(num_neurons * sizeof(int)*num_neurons, sizeof(int));
    int* spike_outputs = calloc(num_neurons*sizeof(double), sizeof(double));
    int topology[3] = {5,4,1};
    double input_current[5] = {3, 3, 4, 2, 2.5};

    initialise_neurons(neurons, num_neurons);
    create_weights(connectivity_array, topology, 3, num_neurons);
    for (int i=0;i<100;i++){
        forward_pass(input_current, 3, 0, num_neurons, neurons, connectivity_array, spike_outputs, topology);
        printf("%f ", neurons[9]->membrane_potential);
    }

    free(neurons);
    free(connectivity_array);
    free(spike_outputs);

    return EXIT_SUCCESS;
}
