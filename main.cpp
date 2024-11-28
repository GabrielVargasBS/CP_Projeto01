#include "include/Network.hpp"
#include "include/Dataset.hpp"

int main(int argc, char *argv[])
{
	int num_atributos = 4;
	int num_classes = 3;

	Neural::Dataset data_learning;
	data_learning.loadInputOutputData(num_atributos, num_classes, "database/iris.txt");

	vector<vector<double>> input = data_learning.getInput();
	vector<vector<double>> output = data_learning.getOutput();

	int maximo_epocas_para_testar = 1000;
	int taxa_acerto_desejada = 95;
	double tolerancia_maxima_de_erro = 0.05;
	int maximo_camadas_escondidas = 15;
	double taxa_de_aprendizado = 0.25;

	Neural::Network neural_network(input, output);
	neural_network.setParameter(
		maximo_epocas_para_testar,
		taxa_acerto_desejada,
		tolerancia_maxima_de_erro,
		maximo_camadas_escondidas,
		taxa_de_aprendizado
	);

	neural_network.autoTraining(maximo_camadas_escondidas, taxa_de_aprendizado);
	return 0;
}






