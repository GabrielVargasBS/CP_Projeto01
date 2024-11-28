#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <iostream>
#include <math.h>
#include <vector>
#include <omp.h>
//#include <mpi.h>
#include <cuda.h>

using namespace std;

namespace Neural
{

    class Network
    {

        struct ForwardPropagation
        {
            vector<double> sum_input_weight;           // Armazena a soma dos produtos entre as entradas e os pesos das entradas para cada neurônio na camada oculta
            vector<double> sum_output_weigth;          // Armazena a soma dos produtos entre as saídas da camada oculta e os pesos de saída para cada neurônio na camada de saída
            vector<double> sum_input_weight_ativation; // Armazena o resultado da função de ativação aplicada ao sum_input_weight
            vector<double> output;                     // Armazena a saída final da rede neural

            ForwardPropagation() {}
            ForwardPropagation(int size_input, int size_output)
            {
                sum_input_weight.resize(size_input);
                sum_output_weigth.resize(size_output);
                fill(sum_input_weight.begin(), sum_input_weight.end(), 0);
                fill(sum_output_weigth.begin(), sum_output_weigth.end(), 0);
            }
        };

        struct BackPropagation
        {
            vector<double> delta_output_sum; // Armazena o erro na saída multiplicado pela derivada da função de ativação na saída
            vector<double> delta_input_sum;  // Armazena o erro na entrada (calculado com base no delta_output_sum e nos pesos de saída) multiplicado pela derivada da função de ativação na entrada

            BackPropagation() {}
            BackPropagation(int size_input)
            {
                delta_input_sum.resize(size_input);
                fill(delta_input_sum.begin(), delta_input_sum.end(), 0);
            }
        };

        struct network
        {
            int epoch = 0;                        // Número de épocas necessárias para treinar a melhor rede
            int hidden_layer = 0;                 // Tamanho da camada oculta da melhor rede
            double learning_rate = 0;             // Taxa de aprendizado da melhor rede
            vector<vector<double>> weight_input;  // Pesos de entrada da melhor rede
            vector<vector<double>> weight_output; // Pesos de saída da melhor rede
        };

    private:
        int input_layer_size;  // Tamanho da camada de entrada
        int output_layer_size; // Tamanho da camada de saída
        int hidden_layer_size; // Tamanho da camada oculta

        vector<vector<double>> input;         // Dados de entrada
        vector<vector<double>> output;        // Dados de saída
        vector<vector<double>> weight_input;  // Pesos de entrada da rede atual
        vector<vector<double>> weight_output; // Pesos de saída da rede atual

        network best_network; // Melhor rede encontrada durante o treinamento

        int epoch;     // Número atual de épocas
        int max_epoch; // Número máximo de épocas permitidas para o treinamento

        int correct_output; // Contador do número de previsões corretas
        int hit_percent;    // Porcentagem de acertos atual

        double desired_percent; // Porcentagem de acertos desejada
        double learning_rate;   // Taxa de aprendizado atual
        double error_tolerance; // Tolerância de erro para considerar uma previsão como correta

    public:

		static bool mpi_finalized;

        Network();
        Network(vector<vector<double>>, vector<vector<double>>);

        void run();

        void trainingClassification();
        void autoTraining(int, double);
        void initializeWeight();
        void hitRateCount(vector<double>, unsigned int);
        void hitRateCalculate();

        ForwardPropagation forwardPropagation(vector<double>);
        void backPropagation(ForwardPropagation, vector<double>, vector<double>);

        double sigmoid(double);
        double sigmoidPrime(double);

        void setInput(vector<vector<double>>);
        void setOutput(vector<vector<double>>);
        void setMaxEpoch(int);
        void setDesiredPercent(int);
        void setHiddenLayerSize(int);
        void setLearningRate(double);
        void setErrorTolerance(double);
        void setParameter(int, int, double, double = 1, int = 1);

        __global__ void autoTrainingKernel(Network*, int, double, Network*)
    };

}

#endif