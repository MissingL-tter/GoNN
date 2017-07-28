package GoNN

import (
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

// Init initializes the network with normally distributed biases
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

func sigmoid(z []float64) {

	// output of sigmoid neuron is
	// where z is the input vector
	//x := 1/1 + math.Exp(-z)

}

func forwardPropagate(n Network) {

	// weights := n.Weights
	// layers := n.Layers

	// for i := range weights {
	// 	for j := range weights[i] {

	// 	}
	// }

}
