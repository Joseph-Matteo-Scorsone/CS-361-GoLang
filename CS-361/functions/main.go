package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Go doesn't have classes, but we can still make our own types
type NeuralNetwork struct {
	inputSize       int
	hiddenSize      int
	outputSize      int
	numHiddenLayers int
	learningRate    float64
	weights         [][][]float64
	biases          [][]float64
	biasOutput      float64
}

func NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers int, learningRate float64, biasOutput float64) NeuralNetwork {
	r := rand.New(rand.NewSource(42)) // for replicability

	// getting it as a useable variable
	nn := NeuralNetwork{inputSize,
		hiddenSize,
		outputSize,
		numHiddenLayers,
		learningRate,
		make([][][]float64, numHiddenLayers+1),
		make([][]float64, numHiddenLayers+1),
		r.Float64(),
	}

	// input layer
	nn.weights[0] = make([][]float64, hiddenSize)
	nn.biases[0] = make([]float64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		nn.weights[0][i] = make([]float64, inputSize)
		nn.biases[0][i] = r.Float64()

		for j := 0; j < inputSize; j++ {
			nn.weights[0][i][j] = r.Float64()
		}
	}

	// hidden layer
	for layer := 1; layer < numHiddenLayers; layer++ {
		nn.weights[layer] = make([][]float64, hiddenSize)
		nn.biases[layer] = make([]float64, hiddenSize)

		for i := 0; i < hiddenSize; i++ {
			nn.weights[layer][i] = make([]float64, hiddenSize)
			nn.biases[layer][i] = r.Float64()

			for j := 0; j < hiddenSize; j++ {
				nn.weights[layer][i][j] = r.Float64()
			}
		}
	}

	// output layer
	nn.weights[numHiddenLayers] = make([][]float64, outputSize)
	nn.biases[numHiddenLayers] = make([]float64, outputSize)

	for i := 0; i < outputSize; i++ {
		nn.weights[numHiddenLayers][i] = make([]float64, hiddenSize)
		nn.biases[numHiddenLayers][i] = r.Float64()

		for j := 0; j < hiddenSize; j++ {
			nn.weights[numHiddenLayers][i][j] = r.Float64()
		}
	}

	return nn
}

// function to get the neural network output
func (nn *NeuralNetwork) ForwardPass(input []float64) (float64, [][]float64) {
	layerOutputs := [][]float64{input}
	current := input

	//Forward pass
	for layer := 0; layer < nn.numHiddenLayers; layer++ {
		nextLayer := make([]float64, nn.hiddenSize)

		for j := 0; j < nn.hiddenSize; j++ {
			sum := nn.biases[layer][j]

			for k := 0; k < len(current); k++ {
				sum += current[k] * nn.weights[layer][j][k]
			}
			nextLayer[j] = Tanh(sum)
		}
		current = nextLayer
		layerOutputs = append(layerOutputs, current)
	}

	//Output layer
	output := nn.biasOutput
	for j := 0; j < nn.hiddenSize; j++ {
		output += current[j] * nn.weights[nn.numHiddenLayers][0][j] // 0 for 1 output neuron
	}
	layerOutputs = append(layerOutputs, []float64{output})

	return output, layerOutputs
}

func Tanh(x float64) float64 {
	return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
}

func TanhDeriv(x float64) float64 {
	return 1 - math.Pow(Tanh(x), 2)
}

func ReLUDeriv(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func MSE(y, yPred float64) float64 {
	return 0.5 * (y - yPred) * (y - yPred)
}

func MSEDeriv(y, yPred float64) float64 {
	return yPred - y
}

func main() {

	// our hyper parameters
	inputSize := 5
	hiddenSize := 4
	outputSize := 1
	numHiddenLayers := 3
	learningRate := 0.02

	// getting it as a useable variable
	nn := NewNeuralNetwork(inputSize,
		hiddenSize,
		outputSize,
		numHiddenLayers,
		learningRate,
		rand.Float64(),
	)

	// print with +v for any kind of type
	fmt.Printf("We have a new Neural Network struct now! %+v\n", nn)
}
