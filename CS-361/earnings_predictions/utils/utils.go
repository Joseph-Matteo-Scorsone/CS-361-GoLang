package utils

import (
	"encoding/csv"
	"io"
	"math"
	"os"
	"strconv"
)

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

func LoadCSVData(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Read() // Skip header

	var data [][]float64
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		row := make([]float64, 3)
		for i, value := range record[1:4] { // LGR, Volume, Volatility
			row[i], err = strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
		}
		data = append(data, row)
	}
	return data, nil
}

// Scaler holds the parameters needed for scaling and unscaling data
type Scaler struct {
	min []float64
	max []float64
}

// NewScaler creates a new scaler for the data
func NewScaler(data [][]float64) *Scaler {
	if len(data) == 0 || len(data[0]) == 0 {
		return &Scaler{}
	}

	numFeatures := len(data[0])
	min := make([]float64, numFeatures)
	max := make([]float64, numFeatures)

	// Initialize min and max with first row
	copy(min, data[0])
	copy(max, data[0])

	// Find min and max for each feature
	for _, row := range data {
		for j, val := range row {
			min[j] = math.Min(min[j], val)
			max[j] = math.Max(max[j], val)
		}
	}

	return &Scaler{
		min: min,
		max: max,
	}
}

// Scale scales a single data point
func (s *Scaler) Scale(data []float64) []float64 {
	scaled := make([]float64, len(data))
	for i, val := range data {
		// Handle case where max equals min to avoid division by zero
		if s.max[i] == s.min[i] {
			scaled[i] = 0.5 // Set to middle of range if all values are the same
		} else {
			scaled[i] = (val - s.min[i]) / (s.max[i] - s.min[i])
		}
	}
	return scaled
}

// ScaleData scales all data points
func (s *Scaler) ScaleData(data [][]float64) [][]float64 {
	scaled := make([][]float64, len(data))
	for i, row := range data {
		scaled[i] = s.Scale(row)
	}
	return scaled
}

// Unscale converts a scaled value back to its original scale
func (s *Scaler) Unscale(data []float64) []float64 {
	unscaled := make([]float64, len(data))
	for i, val := range data {
		unscaled[i] = val*(s.max[i]-s.min[i]) + s.min[i]
	}
	return unscaled
}

// UnscaleValue unscales a single value for a specific feature index
func (s *Scaler) UnscaleValue(value float64, featureIndex int) float64 {
	return value*(s.max[featureIndex]-s.min[featureIndex]) + s.min[featureIndex]
}