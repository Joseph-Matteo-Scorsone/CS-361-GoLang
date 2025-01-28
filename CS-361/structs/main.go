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

	// our hyper parameters
	inputSize := 5
	hiddenSize := 4
	outputSize := 1
	numHiddenLayers := 3
	learningRate := 0.001

	// getting it as a useable variable
	nn := NeuralNetwork{inputSize,
		hiddenSize,
		outputSize,
		numHiddenLayers,
		learningRate,
		make([][][]float64, numHiddenLayers+1),
		make([][]float64, numHiddenLayers+1),
		rand.Float64(),
	}

	// print with +v for any kind of type
	fmt.Printf("We have a new Neural Network struct now! %+v\n", nn)
}

// Our weights and baises is empty though, we will next go through and use a function to initialize them.
