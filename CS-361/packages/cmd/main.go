package main

import (
	"NeuralNetworks/types"
	"fmt"
	"math"
	"math/rand"
	"time"
)

func main() {

	// Start timing
	start := time.Now()

	// Our hyperparameters
	inputSize := 5
	hiddenSize := 5
	outputSize := 1
	numHiddenLayers := 4
	learningRate := 0.02
	epochs := 1000
	sinLength := 1000

	// Create the neural network
	nn, err := types.NewNeuralNetwork(inputSize, hiddenSize, outputSize, numHiddenLayers, learningRate, rand.Float64())
	if err != nil {
		fmt.Println("Error creating neural network:", err)
		return
	}

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
			if err := nn.Train(inputs[i], targets[i]); err != nil {
				fmt.Printf("Error training at epoch %d, input %d: %v\n", epoch, i, err)
			}
		}
	}

	actual := math.Sin(float64(sinLength + inputSize + 1))
	fmt.Printf("Actual next: %.2f\n", actual)

	testInput := make([]float64, inputSize)
	for i := 0; i < inputSize; i++ {
		testInput[i] = sinData[sinLength-inputSize+i]
	}

	// Make prediction
	prediction, err := nn.Predict(testInput)
	if err != nil {
		fmt.Println("Error making prediction:", err)
		return
	}
	fmt.Printf("Predicted: %.2f\n", prediction)

	// Calculate error
	pred_err := math.Abs(prediction - actual)
	fmt.Printf("Error: %.2f\n", pred_err)

	// End timing and output
	elapsed := time.Since(start)
	fmt.Printf("Elapsed time: %.6f seconds\n", elapsed.Seconds())
}
