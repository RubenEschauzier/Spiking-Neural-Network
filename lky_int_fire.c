#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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

double generate_spike(struct leakyNeuron* neuron){
    neuron -> membrane_potential = neuron ->refactory_potential;
    neuron -> spiked = 1;
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

double update_neuron(struct leakyNeuron* neuron, int current_time, double input_current){
    if (neuron -> spiked >= 3){
        neuron -> spiked = 0;
    }
    if(neuron -> spiked > 0){
        neuron -> spiked += 1;
        return neuron -> membrane_potential;
    }
    double new_membrane = euler_ode_solver(1000, current_time, current_time+1, neuron->membrane_potential, neuron->resistance,neuron->rest_potential,
    input_current, neuron->time_constant, leaky_integrator_model);
    neuron -> membrane_potential = new_membrane;
    if (neuron -> membrane_potential > neuron -> threshold){
        neuron -> spiked = 1;
        neuron -> membrane_potential = neuron -> refactory_potential;
    }
    printf("Neuron potential: %f \n", neuron ->membrane_potential);
    //printf("Threshold = %f\n", neuron->threshold);
    return neuron -> membrane_potential;
}

void simulate_neuron(int sim_steps){
    struct leakyNeuron* neuron = create_neuron(1.0, 0, 2.0, 2, 10);
    neuron -> membrane_potential = neuron -> rest_potential;

    double results[sim_steps];
    for (int i=1; i<sim_steps;i++){
        double membrane_potential = update_neuron(neuron, i, 1);
        results[i] = membrane_potential;
    }
    printf("[%f", results[0]);
    for (int j=1; j<sim_steps;j++){
        printf(", %f",results[j]);
    }
    printf("]\n");
}

int main( int argc, char ** argv ) {

    struct leakyNeuron* neuron = create_neuron(1.0, 0, 2.0, 2, 10);
    simulate_neuron(100);

    return EXIT_SUCCESS;
}
