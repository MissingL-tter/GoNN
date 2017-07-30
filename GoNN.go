package GoNN

import (
	"math"
	"math/rand"
	"time"
)

// Network defines the structure of the network
type Network struct {
	Biases  [][]float64
	Weights [][]float64
}

// CreateNetwork returns a Network struct.
//
// inputNodes is the number of inputs, hiddenLayers is the number of hidden layers, hiddenNodes is an array representing the number of nodes for each hidden layer.
func CreateNetwork(inputNodes int, hiddenLayers int, hiddenNodes []int, outputNodes int) *Network {

	// Init layers and fill with for the number of nodes at that layer
	biases := make([][]float64, hiddenLayers+2)

	biases[0] = make([]float64, inputNodes)
	biases[len(biases)-1] = make([]float64, outputNodes)

	for i := range hiddenNodes {
		biases[i+1] = make([]float64, hiddenNodes[i])
	}

	// Init weights and fill with weights for each node at each layer
	weights := make([][]float64, len(biases)-1)

	for i := range weights {
		weights[i] = make([]float64, len(biases[i])*len(biases[i+1]))
	}

	return &Network{biases, weights}
}

// InitDebug initializes the network where all biases = 1 and all weights = 2
func (n Network) InitDebug() {

	for i := range n.Biases {
		if i != 0 {
			for j := range n.Biases[i] {
				n.Biases[i][j] = 1
			}
			for j := range n.Weights[i-1] {
				n.Weights[i-1][j] = 2
			}
		}
	}

}

// Init initializes the network with normally distributed biases adn weights
func (n Network) Init() {

	rand.Seed(time.Now().Unix())

	for i := range n.Biases {
		if i != 0 {
			for j := range n.Biases[i] {
				n.Biases[i][j] = rand.NormFloat64()
			}
			for j := range n.Weights[i-1] {
				n.Weights[i-1][j] = rand.NormFloat64()
			}
		}
	}
}

// ForwardProp takes some input and processes it through the network to return an output
// Basic concept implemented
func (n Network) ForwardProp() []float64 {

	input := n.Biases[0]

	for l := 1; l < len(n.Biases); l++ {
		var output []float64
		for b := 0; b < len(n.Biases[l]); b++ {
			sum := 0.0
			for i := range input {
				idx := len(input)*b + i
				sum += input[i] * n.Weights[l-1][idx]
			}
			output = append(output, sigmoid(sum+n.Biases[l][b]))
		}
		input = output
	}

	return input

}

// BackProp takes some output compared to what was expected by the input and uses stochastic gradient descent to minimize the loss function
func (n Network) BackProp() []float64 {
	return []float64{0.0, 0.0}
}

// Stochastic Gradient Descent
func sgd() {

}

func sigmoid(z float64) float64 {

	return 1 / (1 + math.Exp(-z))

}

func sigPrime(z float64) float64 {

	return sigmoid(z) * (1 - sigmoid(z))

}
