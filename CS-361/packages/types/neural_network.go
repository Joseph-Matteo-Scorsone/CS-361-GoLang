package types

import (
	"NeuralNetworks/utils"
	"errors"
	"fmt"
	"math/rand"
)

// NeuralNetworkInterface defines the behavior of a neural network
type NeuralNetworkInterface interface {
	Train(input []float64, target float64) error
	Predict(input []float64) (float64, error)
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

// ErrorInvalidLayerSize is returned when a layer size is invalid
var ErrorInvalidLayerSize = errors.New("invalid layer size")

// NewNeuralNetwork creates a new neural network
func NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers int, learningRate float64, biasOutput float64) (*NeuralNetwork, error) {
	// Validate inputs
	if inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0 || numHiddenLayers <= 0 {
		return nil, fmt.Errorf("invalid network dimensions: %w", ErrorInvalidLayerSize)
	}

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

	return nn, nil
}

// Train trains the neural network using input and target
func (nn *NeuralNetwork) Train(input []float64, target float64) error {
	// Validate input size
	if len(input) != nn.inputSize {
		return fmt.Errorf("input size mismatch: expected %d, got %d", nn.inputSize, len(input))
	}

	// Forward pass
	output, layerOutputs, err := nn.ForwardPass(input)
	if err != nil {
		return err
	}

	// Backpropagation
	// Calculate the derivative of the loss with respect to the output (MSE derivative)
	deltaOutput := utils.MSEDeriv(target, output)

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
			deltaHidden[i] = deltaOutput * utils.TanhDeriv(layerOutputs[layer+1][i])

			// Update the weights and biases for this layer
			for j := 0; j < len(layerOutputs[layer]); j++ {
				nn.layers[layer].weights[i][j] -= nn.learningRate * deltaHidden[i] * layerOutputs[layer][j]
			}
			nn.layers[layer].biases[i] -= nn.learningRate * deltaHidden[i]
		}
	}
	return nil
}

// Predict makes a prediction based on the input
func (nn *NeuralNetwork) Predict(input []float64) (float64, error) {
	// Validate input size
	if len(input) != nn.inputSize {
		return 0, fmt.Errorf("input size mismatch: expected %d, got %d", nn.inputSize, len(input))
	}

	// Forward pass
	output, _, err := nn.ForwardPass(input)
	if err != nil {
		return 0, err
	}
	return output, nil
}

// ForwardPass computes the forward pass through the network
func (nn *NeuralNetwork) ForwardPass(input []float64) (float64, [][]float64, error) {
	if len(input) != nn.inputSize {
		return 0, nil, fmt.Errorf("input size mismatch: expected %d, got %d", nn.inputSize, len(input))
	}

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
			nextLayer[j] = utils.Tanh(sum)
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

	return output, layerOutputs, nil
}
