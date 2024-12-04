#include "../include/Network.hpp"

namespace Neural{

    Network::Network(){}

    Network::Network(vector<vector<double>> user_input, vector<vector<double>> user_output){

        setInput(user_input);
        setOutput(user_output);
        output_layer_size = 3;

    }

    void Network::setParameter( int user_max_epoch, int user_desired_percent, double user_error_tolerance, double user_learning_rate, int user_hidden_layer_size){

        setMaxEpoch(user_max_epoch);
        setLearningRate(user_learning_rate);
        setErrorTolerance(user_error_tolerance);
        setDesiredPercent(user_desired_percent);
        setHiddenLayerSize(user_hidden_layer_size);
        best_network.epoch = max_epoch;

        initializeWeight();
    }

    void Network::run(){

        for (unsigned int data_row = 0; data_row < input.size(); data_row++){
            ForwardPropagation forward = forwardPropagation(input[data_row]);
            hitRateCount(forward.output, data_row);            
        }
        hitRateCalculate();    
    }

    __device__ void Network::trainingClassification(){

        for (epoch = 0; epoch < max_epoch && hit_percent < desired_percent; epoch++) {
            for (unsigned int data_row = 0; data_row < input.size(); data_row++){
                ForwardPropagation forward = forwardPropagation(input[data_row]);
                backPropagation(forward, input[data_row], output[data_row]);
            }
            run();
        }

        cout << "Hidden Layer Size: " << hidden_layer_size 
            << "\tLearning Rate: " << learning_rate 
            << "\tHit Percent: " << hit_percent << "%" 
            << "\tEpoch: " << epoch << endl;
    }

    // Variáveis globais
    __device__ double best_weights_input[MAX_THREADS][INPUT_SIZE][HIDDEN_LAYER_SIZE];
    __device__ double best_weights_output[MAX_THREADS][HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];

    __global__ void autoTrainingKernel(int hidden_layer_limit, double learning_rate_increase, int* best_network_epoch, double* best_network_learning_rate, int* best_network_hidden_layer) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

        for (int i = index; i < hidden_layer_limit; i += stride){
            for (double learning_rate = learning_rate_increase; learning_rate <= 1; learning_rate = learning_rate + learning_rate_increase){
                initializeWeight();
                trainingClassification();
                if (epoch < best_network_epoch[index]){
                    best_network_epoch[index] = epoch;
                    best_network_learning_rate[index] = learning_rate;
                    best_network_hidden_layer[index] = hidden_layer_size;
                    
                    // Copie os pesos da melhor rede
                    for (int j = 0; j < INPUT_SIZE; j++) {
                        for (int k = 0; k < HIDDEN_LAYER_SIZE; k++) {
                            best_weights_input[index][j][k] = weight_input[j][k];
                        }
                    }
                    for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                        for (int k = 0; k < OUTPUT_LAYER_SIZE; k++) {
                            best_weights_output[index][j][k] = weight_output[j][k];
                        }
                    }
                }
            }
        }
    }

    void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase){
        int* d_best_network_epoch;
        double* d_best_network_learning_rate;
        int* d_best_network_hidden_layer;

        // Aloque memória no dispositivo para as variáveis da melhor rede.
        cudaMalloc((void**)&d_best_network_epoch, sizeof(int) * MAX_THREADS);
        cudaMalloc((void**)&d_best_network_learning_rate, sizeof(double) * MAX_THREADS);
        cudaMalloc((void**)&d_best_network_hidden_layer, sizeof(int) * MAX_THREADS);

        // Inicialize as variáveis na memória do dispositivo.
        int initial_epoch = INT_MAX;
        for (int i = 0; i < MAX_THREADS; i++) {
            cudaMemcpy(&d_best_network_epoch[i], &initial_epoch, sizeof(int), cudaMemcpyHostToDevice);
        }
        double initial_learning_rate = 0.0;
        cudaMemcpy(d_best_network_learning_rate, &initial_learning_rate, sizeof(double), cudaMemcpyHostToDevice);
        int initial_hidden_layer_size = 0;
        cudaMemcpy(d_best_network_hidden_layer, &initial_hidden_layer_size, sizeof(int), cudaMemcpyHostToDevice);

        // Lançar o kernel CUDA.
        int blockSize = 256;
        int numBlocks = (hidden_layer_limit + blockSize - 1) / blockSize;
        autoTrainingKernel<<<numBlocks, blockSize>>>(hidden_layer_limit, learning_rate_increase, d_best_network_epoch, d_best_network_learning_rate, d_best_network_hidden_layer);

        // Copie os resultados de volta para o host.
        int* best_network_epoch = new int[MAX_THREADS];
        double* best_network_learning_rate = new double[MAX_THREADS];
        int* best_network_hidden_layer = new int[MAX_THREADS];

        cudaMemcpy(best_network_epoch, d_best_network_epoch, sizeof(int) * MAX_THREADS, cudaMemcpyDeviceToHost);
        cudaMemcpy(best_network_learning_rate, d_best_network_learning_rate, sizeof(double) * MAX_THREADS, cudaMemcpyDeviceToHost);
        cudaMemcpy(best_network_hidden_layer, d_best_network_hidden_layer, sizeof(int) * MAX_THREADS, cudaMemcpyDeviceToHost);

        // Encontre a melhor rede entre todas as threads.
        int best_thread = 0;
        for (int i = 1; i < MAX_THREADS; i++) {
            if (best_network_epoch[i] < best_network_epoch[best_thread]) {
                best_thread = i;
            }
        }

        epoch = best_network_epoch[best_thread];
        learning_rate = best_network_learning_rate[best_thread];
        hidden_layer_size = best_network_hidden_layer[best_thread];

        // Copie os pesos da melhor rede de volta para o host.
        double* best_weights_input_host = new double[INPUT_SIZE][HIDDEN_LAYER_SIZE];
        double* best_weights_output_host = new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE];

        cudaMemcpy(best_weights_input_host, best_weights_input[best_thread], sizeof(double) * INPUT_SIZE * HIDDEN_LAYER_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(best_weights_output_host, best_weights_output[best_thread], sizeof(double) * HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE, cudaMemcpyDeviceToHost);

        // Agora, você pode usar best_weights_input_host e best_weights_output_host no seu código.
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                best_network.weight_input[i][j] = best_weights_input_host[i][j];
            }
        }

        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < OUTPUT_LAYER_SIZE; j++) {
                best_network.weight_output[i][j] = best_weights_output_host[i][j];
            }
        }

        cudaFree(d_best_network_epoch);
        cudaFree(d_best_network_learning_rate);
        cudaFree(d_best_network_hidden_layer);

        delete[] best_network_epoch;
        delete[] best_network_learning_rate;
        delete[] best_network_hidden_layer;

        delete[] best_weights_input_host;
        delete[] best_weights_output_host;

        cout << "Best Network --> Hidden Layer Size: " << hidden_layer_size 
            << "\tLearning Rate: " << learning_rate 
            << "\tEpoch: " << epoch << endl;
    }

    Network::ForwardPropagation Network::forwardPropagation(vector<double> input_line){

        input_line.push_back(1); // bias

        ForwardPropagation forward(hidden_layer_size, output_layer_size);

        // somatório dos produtos entre, entrada e peso das entradas em cada neurônio da camada oculta
        for (int i = 0; i < hidden_layer_size; i++ ){
            for (int j = 0; j < input_layer_size; j++ ){
                forward.sum_input_weight[i] += input_line[j] * weight_input[j][i];
            }
        }

        // aplica função de ativação, em cada somatório encontrado, ou em cada neurônio da camada oculta  (sigmoid)
        for (int i = 0; i < hidden_layer_size; i++ ){
            forward.sum_input_weight_ativation.push_back(sigmoid(forward.sum_input_weight[i]));
        }

        // somatório dos produtos entre, o somatório dos neurônios na camada oculta e o peso das saídas
        for (int i = 0; i < output_layer_size; i++ ){
            for (int j = 0; j < hidden_layer_size; j++ ){
                forward.sum_output_weigth[i] += forward.sum_input_weight_ativation[j] * weight_output[j][i];
            }
        }

        // aplica função de ativação em cada somatório encontrado, ou em cada nerônio da camada de saída (sigmoidPrime), saída da rede neural
        for (int i = 0; i < output_layer_size; i++ ){
            forward.output.push_back(sigmoid(forward.sum_output_weigth[i]));
        }

        return forward;
    }

    void Network::backPropagation(ForwardPropagation forward, vector<double> input_line, vector<double> output_line){

        input_line.push_back(1); // bias
        
        BackPropagation back(hidden_layer_size);

        // erro entre a saída esperada e a calculada, multiplicado pela taxa de mudança da função de ativação no somatório de saída (derivada)
        for (int i = 0; i < output_layer_size; i++ ){
            back.delta_output_sum.push_back((output_line[i] - forward.output[i]) * sigmoidPrime(forward.sum_output_weigth[i]));
        }

        // erro da saída multiplicado pelos pesos de saída, aplicando a taxa de mudança da função de ativação no somatório da camada oculta (derivada)
        for (int i = 0; i < hidden_layer_size; i++ ){
            for (int j = 0; j < output_layer_size; j++ ){
                back.delta_input_sum[i] += back.delta_output_sum[j] * weight_output[i][j];
            }
            back.delta_input_sum[i] *= sigmoidPrime(forward.sum_input_weight[i]);
        }

        // corrigindo os valores dos pesos de saída
        for (unsigned int i = 0; i < weight_output.size(); i++){
            for (unsigned int j = 0; j < weight_output[i].size(); j++){
                weight_output[i][j] += back.delta_output_sum[j] * forward.sum_input_weight_ativation[i] * learning_rate;
            }        
        }

        // corrigindo os valores dos pesos de entrada
        for (unsigned int i = 0; i < weight_input.size(); i++){
            for (unsigned int j = 0; j < weight_input[i].size(); j++){
                weight_input[i][j] += back.delta_input_sum[j] * input_line[i] * learning_rate;
            }        
        }
    }

    void Network::hitRateCount(vector<double> neural_output, unsigned int data_row){

        for (int i = 0; i < output_layer_size; i++ ){
            if (abs(neural_output[i] - output[data_row][i]) < error_tolerance)
                correct_output++;
        }
    }

    void Network::hitRateCalculate(){

        hit_percent = (correct_output*100) / (output.size() * output_layer_size);
        correct_output = 0;
    }

    __device__ void Network::initializeWeight(){

        weight_input.resize(input_layer_size);
        weight_output.resize(hidden_layer_size);

        curandState_t state;
        curand_init(clock64(), 0, 0, &state);

        for (unsigned int i = 0; i < weight_input.size(); i++ ){
            weight_input[i].clear();
            for ( int j = 0; j < hidden_layer_size; j++ ){
                weight_input[i].push_back(curand_uniform_double(&state));
            }
        }

        for (unsigned int i = 0; i < weight_output.size(); i++ ){
            weight_output[i].clear();        
            for ( int j = 0; j < output_layer_size; j++ ){
                weight_output[i].push_back(curand_uniform_double(&state));
            }
        }

        hit_percent = 0;
        correct_output = 0;
    }

    double Network::sigmoid(double z){
        return 1/(1+exp(-z));
    }	

    double Network::sigmoidPrime(double z){
        return exp(-z) / ( pow(1+exp(-z),2) );
    }

    void Network::setMaxEpoch(int m){
        max_epoch = m;
    }

    void Network::setDesiredPercent(int d){
        desired_percent = d;
    }

    void Network::setHiddenLayerSize(int h){
        hidden_layer_size = h;
    }

    void Network::setLearningRate(double l){
        learning_rate = l;
    }

    void Network::setErrorTolerance(double e){
        error_tolerance = e;
    }

    void Network::setInput(vector<vector<double>> i){
        input = i;
        input_layer_size = i[0].size() + 1; // +1 bias
    }

    void Network::setOutput(vector<vector<double>> o){
        output = o;
        output_layer_size = o[0].size();    
    }

}
