package GoNN

// Network defines the structure of the network
type Network struct {
	layers  [][]float32
	weights [][][]float32
}

// CreateNetwork returns a Network struct.
//
// inputNodes is the number of inputs, hiddenLayers is the number of hidden layers, hiddenNodes is an array representing the number of nodes for each hidden layer.
func CreateNetwork(inputNodes int, hiddenLayers int, hiddenNodes []int, outputNodes int) *Network {

	// Init layers and fill with for the number of nodes at that layer
	layers := make([][]float32, hiddenLayers+2)

	layers[0] = make([]float32, inputNodes)
	layers[len(layers)-1] = make([]float32, outputNodes)

	for i := 0; i < len(hiddenNodes); i++ {
		layers[i+1] = make([]float32, hiddenNodes[i])
	}

	// Init weights and fill with weights for each node at each layer
	weights := make([][][]float32, len(layers)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = make([][]float32, len(layers[i]))

		for j := 0; j < len(weights[i]); j++ {
			weights[i][j] = make([]float32, len(layers[i+1]))
		}
	}

	return &Network{layers, weights}
}

// Train iteratively propagates over the network to train it to a data set
func (n Network) Train() {

}

// ForwardPropagate performs forward propagation on the network to determine an output
func (n Network) ForwardPropagate() {

}

// BackPropagate performs back propagation on the network to adjust weights
func (n Network) BackPropagate() {

}
