package main

import (
	"fmt"
	"math"
	"math/rand"
)

// NeuralNetworkInterface defines the behavior of a neural network
type NeuralNetworkInterface interface {
	Train(input []float64, target float64)
	Predict(input []float64) float64
}

// Layer defines a generic type for a neural network layer
type Layer[T any] struct {
	weights [][]T
	biases  []T
}

// NeuralNetwork represents the structure of a neural network
type NeuralNetwork struct {
	inputSize       int
	hiddenSize      int
	outputSize      int
	numHiddenLayers int
	learningRate    float64
	layers          []Layer[float64] // Using float64 as the type for layers
	biasOutput      float64
}

func NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers int, learningRate float64, biasOutput float64) *NeuralNetwork {
	r := rand.New(rand.NewSource(42)) // for replicability

	// Create the neural network structure
	nn := &NeuralNetwork{
		inputSize:       inputSize,
		hiddenSize:      hiddenSize,
		outputSize:      outputSize,
		numHiddenLayers: numHiddenLayers,
		learningRate:    learningRate,
		layers:          make([]Layer[float64], numHiddenLayers+1), // Use float64 for the layer type
		biasOutput:      biasOutput,
	}

	// Initialize layers with random weights and biases
	// Input layer
	nn.layers[0] = Layer[float64]{
		weights: make([][]float64, hiddenSize),
		biases:  make([]float64, hiddenSize),
	}
	for i := 0; i < hiddenSize; i++ {
		nn.layers[0].weights[i] = make([]float64, inputSize)
		nn.layers[0].biases[i] = r.Float64()

		for j := 0; j < inputSize; j++ {
			nn.layers[0].weights[i][j] = r.Float64()
		}
	}

	// Hidden layers
	for layer := 1; layer < numHiddenLayers; layer++ {
		nn.layers[layer] = Layer[float64]{
			weights: make([][]float64, hiddenSize),
			biases:  make([]float64, hiddenSize),
		}

		for i := 0; i < hiddenSize; i++ {
			nn.layers[layer].weights[i] = make([]float64, hiddenSize)
			nn.layers[layer].biases[i] = r.Float64()

			for j := 0; j < hiddenSize; j++ {
				nn.layers[layer].weights[i][j] = r.Float64()
			}
		}
	}

	// Output layer
	nn.layers[numHiddenLayers] = Layer[float64]{
		weights: make([][]float64, outputSize),
		biases:  make([]float64, outputSize),
	}

	for i := 0; i < outputSize; i++ {
		nn.layers[numHiddenLayers].weights[i] = make([]float64, hiddenSize)
		nn.layers[numHiddenLayers].biases[i] = r.Float64()

		for j := 0; j < hiddenSize; j++ {
			nn.layers[numHiddenLayers].weights[i][j] = r.Float64()
		}
	}

	return nn
}

func (nn *NeuralNetwork) Train(input []float64, target float64) {
	// Forward pass
	output, layerOutputs := nn.ForwardPass(input)

	// Backpropagation
	// Calculate the derivative of the loss with respect to the output (MSE derivative)
	deltaOutput := MSEDeriv(target, output)

	// Gradient for the output layer
	for i := 0; i < nn.hiddenSize; i++ {
		// Calculate the gradient of the weights and update them
		nn.layers[nn.numHiddenLayers].weights[0][i] -= nn.learningRate * deltaOutput * layerOutputs[nn.numHiddenLayers][i]
	}
	// Update the output layer bias
	nn.biasOutput -= nn.learningRate * deltaOutput

	// Backpropagate through the hidden layers
	deltaHidden := make([]float64, nn.hiddenSize)
	for layer := nn.numHiddenLayers - 1; layer >= 0; layer-- {
		for i := 0; i < nn.hiddenSize; i++ {
			// Compute the gradient of the hidden layer
			deltaHidden[i] = deltaOutput * TanhDeriv(layerOutputs[layer+1][i])

			// Update the weights and biases for this layer
			for j := 0; j < len(layerOutputs[layer]); j++ {
				nn.layers[layer].weights[i][j] -= nn.learningRate * deltaHidden[i] * layerOutputs[layer][j]
			}
			nn.layers[layer].biases[i] -= nn.learningRate * deltaHidden[i]
		}
	}
}

func (nn *NeuralNetwork) Predict(input []float64) float64 {
	// Forward pass
	output, _ := nn.ForwardPass(input)
	return output
}

// function to get the neural network output
func (nn *NeuralNetwork) ForwardPass(input []float64) (float64, [][]float64) {
	layerOutputs := [][]float64{input}
	current := input

	// Forward pass
	for layer := 0; layer < nn.numHiddenLayers; layer++ {
		nextLayer := make([]float64, nn.hiddenSize)

		for j := 0; j < nn.hiddenSize; j++ {
			sum := nn.layers[layer].biases[j]

			for k := 0; k < len(current); k++ {
				sum += current[k] * nn.layers[layer].weights[j][k]
			}
			nextLayer[j] = Tanh(sum)
		}
		current = nextLayer
		layerOutputs = append(layerOutputs, current)
	}

	// Output layer
	output := nn.biasOutput
	for j := 0; j < nn.hiddenSize; j++ {
		output += current[j] * nn.layers[nn.numHiddenLayers].weights[0][j] // 0 for 1 output neuron
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

func MSE(y, yPred float64) float64 {
	return 0.5 * (y - yPred) * (y - yPred)
}

func MSEDeriv(y, yPred float64) float64 {
	return yPred - y
}

func main() {
	// Our hyperparameters
	inputSize := 5
	hiddenSize := 5
	outputSize := 1
	numHiddenLayers := 4
	learningRate := 0.02
	epochs := 1000
	sinLength := 1000

	// Create the neural network
	nn := NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers, learningRate, rand.Float64())

	// Sin data for predicting
	sinData := []float64{}
	dataLength := sinLength + inputSize
	for i := 0; i < dataLength; i++ {
		value := math.Sin(float64(i))
		sinData = append(sinData, value)
	}

	// Training data
	inputs := [][]float64{}
	targets := []float64{}
	for i := inputSize; i < sinLength; i++ {
		// Create an input vector with size inputSize
		input := make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			input[j] = sinData[i-inputSize+j]
		}
		inputs = append(inputs, input)
		targets = append(targets, sinData[i+1]) // Target is the next value
	}

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < len(inputs); i++ {
			nn.Train(inputs[i], targets[i])
		}
	}

	actual := math.Sin(float64(sinLength+inputSize+1))
	fmt.Printf("Actual next: %.2f\n", actual)

	testInput := make([]float64, inputSize)
	for i := 0; i < inputSize; i++ {
		testInput[i] = sinData[sinLength-inputSize+i]
	}

	// Make prediction
	prediction := nn.Predict(testInput)
	fmt.Printf("Predicted: %.2f\n", prediction)

	// Calculate error
	pred_err := math.Abs(prediction - actual)
	fmt.Printf("Error: %.2f\n", pred_err)
}
