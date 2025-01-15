package main

import (
	"NeuralNetworks/types"
	"NeuralNetworks/utils"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"
)

// TimeSeriesData represents our processed dataset
type TimeSeriesData struct {
	Features [][]float64
	Targets  []float64
}

// prepareTimeSeriesData creates sliding window sequences for prediction
func prepareTimeSeriesData(data [][]float64, windowSize int) (*TimeSeriesData, error) {
	if len(data) < windowSize+1 {
		return nil, fmt.Errorf("not enough data points for window size %d", windowSize)
	}

	var features [][]float64
	var targets []float64

	// Create sliding windows
	for i := 0; i <= len(data)-windowSize-1; i++ {
		// Flatten window data into a single feature vector
		window := make([]float64, 0, windowSize*len(data[0]))
		for j := 0; j < windowSize; j++ {
			window = append(window, data[i+j]...)
		}
		features = append(features, window)
		// Target is the LGR of the next day
		targets = append(targets, data[i+windowSize][0])
	}

	return &TimeSeriesData{
		Features: features,
		Targets:  targets,
	}, nil
}

// splitData splits data into training and testing sets
func splitData(data *TimeSeriesData, trainRatio float64) (*TimeSeriesData, *TimeSeriesData) {
	splitIdx := int(float64(len(data.Features)) * trainRatio)

	return &TimeSeriesData{
			Features: data.Features[:splitIdx],
			Targets:  data.Targets[:splitIdx],
		}, &TimeSeriesData{
			Features: data.Features[splitIdx:],
			Targets:  data.Targets[splitIdx:],
		}
}

func main() {
	start := time.Now()

	// Hyperparameters
	const (
		windowSize      = 5
		trainRatio      = 0.8
		hiddenSize      = 32
		numHiddenLayers = 3
		learningRate    = 0.01
		epochs          = 1000
		batchSize       = 10
	)

	// Load data
	ticker := "CRWD"
	path := fmt.Sprintf("../CSVs/%s_features.csv", ticker)
	csvData, err := utils.LoadCSVData(path)
	if err != nil {
		log.Fatal(err)
	}

	// Create and fit scaler
	scaler := utils.NewScaler(csvData)

	// Scale the data
	scaledData := scaler.ScaleData(csvData)

	// Prepare time series data with scaled values
	timeSeriesData, err := prepareTimeSeriesData(scaledData, windowSize)
	if err != nil {
		log.Fatal(err)
	}

	// Split into training and testing sets
	trainData, testData := splitData(timeSeriesData, trainRatio)

	// Calculate input size based on window size and features per timestep
	inputSize := windowSize * len(csvData[0])

	// Create neural network
	nn, err := types.NewNeuralNetwork(
		inputSize,
		hiddenSize,
		1,
		numHiddenLayers,
		learningRate,
		rand.Float64(),
	)
	if err != nil {
		log.Fatal("Error creating neural network:", err)
	}

	// Training loop (same as before)
	numBatches := len(trainData.Features) / batchSize
	for epoch := 0; epoch < epochs; epoch++ {
		var wg sync.WaitGroup
		var totalError float64
		var errorLock sync.Mutex

		for batch := 0; batch < numBatches; batch++ {
			wg.Add(1)
			go func(batch int) {
				defer wg.Done()
				startIdx := batch * batchSize
				endIdx := min((batch+1)*batchSize, len(trainData.Features))

				batchError := 0.0
				for i := startIdx; i < endIdx; i++ {
					err := nn.Train(trainData.Features[i], trainData.Targets[i])
					if err != nil {
						log.Printf("Training error at epoch %d, batch %d: %v\n", epoch, batch, err)
						continue
					}

					// Calculate error for this sample
					pred, _ := nn.Predict(trainData.Features[i])
					// Unscale predictions and targets for error calculation
					unscaledPred := scaler.UnscaleValue(pred, 0) // 0 is the index for LGR
					unscaledTarget := scaler.UnscaleValue(trainData.Targets[i], 0)
					batchError += math.Abs(unscaledPred - unscaledTarget)
				}

				errorLock.Lock()
				totalError += batchError
				errorLock.Unlock()
			}(batch)
		}
		wg.Wait()

		if epoch%100 == 0 {
			avgError := totalError / float64(len(trainData.Features))
			fmt.Printf("Epoch %d - Average Error: %.6f\n", epoch, avgError)
		}
	}

	// Evaluate on test set
	var totalTestError float64
	for i := 0; i < len(testData.Features); i++ {
		prediction, err := nn.Predict(testData.Features[i])
		if err != nil {
			log.Printf("Prediction error for test sample %d: %v\n", i, err)
			continue
		}

		// Unscale prediction and target for error calculation
		unscaledPred := scaler.UnscaleValue(prediction, 0)
		unscaledTarget := scaler.UnscaleValue(testData.Targets[i], 0)
		totalTestError += math.Abs(unscaledPred - unscaledTarget)
	}
	avgTestError := totalTestError / float64(len(testData.Features))

	fmt.Printf("\nTest Set Results:\n")
	fmt.Printf("Average Error: %.6f\n", avgTestError)

	// Make a prediction for the next day
	lastWindow := testData.Features[len(testData.Features)-1]
	nextDayPrediction, err := nn.Predict(lastWindow)
	if err != nil {
		log.Fatal("Error predicting next day:", err)
	}

	// Unscale the prediction
	unscaledPrediction := scaler.UnscaleValue(nextDayPrediction, 0)
	fmt.Printf("Next Day LGR Prediction: %.6f\n", unscaledPrediction)

	fmt.Printf("Training Time: %.2f seconds\n", time.Since(start).Seconds())

	// Load new data file for prediction
	newFilePath := fmt.Sprintf("../CSVs/%s_trade.csv", ticker)
	newCsvData, err := utils.LoadCSVData(newFilePath)
	if err != nil {
		log.Fatalf("Error loading new data file: %v\n", err)
	}

	// Scale the new data using the existing scaler
	newScaledData := scaler.ScaleData(newCsvData)

	// Ensure there are enough data points for prediction
	if len(newScaledData) < windowSize {
		log.Fatalf("Not enough data points in new file for prediction. Require at least %d data points.\n", windowSize)
	}

	// Extract the last windowSize entries for prediction
	lastWindow = make([]float64, 0, windowSize*len(newCsvData[0]))
	for i := len(newScaledData) - windowSize; i < len(newScaledData); i++ {
		lastWindow = append(lastWindow, newScaledData[i]...)
	}

	// Predict the next value
	newPrediction, err := nn.Predict(lastWindow)
	if err != nil {
		log.Fatalf("Error predicting for new data: %v\n", err)
	}

	// Unscale the prediction
	unscaledNewPrediction := scaler.UnscaleValue(newPrediction, 0)
	fmt.Printf("Prediction for new data: %.6f\n", unscaledNewPrediction)

}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
