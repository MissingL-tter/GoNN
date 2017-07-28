package GoNN

// Network defines the structure of the network
type Network struct {
	Layers  [][]float64
	Weights [][][]float64
}

// CreateNetwork returns a Network struct.
//
// inputNodes is the number of inputs, hiddenLayers is the number of hidden layers, hiddenNodes is an array representing the number of nodes for each hidden layer.
func CreateNetwork(inputNodes int, hiddenLayers int, hiddenNodes []int, outputNodes int) *Network {

	// Init layers and fill with for the number of nodes at that layer
	layers := make([][]float64, hiddenLayers+2)

	layers[0] = make([]float64, inputNodes)
	layers[len(layers)-1] = make([]float64, outputNodes)

	for i := range hiddenNodes {
		layers[i+1] = make([]float64, hiddenNodes[i])
	}

	// Init weights and fill with weights for each node at each layer
	weights := make([][][]float64, len(layers)-1)

	for i := range weights {
		weights[i] = make([][]float64, len(layers[i]))

		for j := range weights[i] {
			weights[i][j] = make([]float64, len(layers[i+1]))
		}
	}

	return &Network{layers, weights}
}

func sigmoid(z []float64) {

	// output of sigmoid neuron is
	// where z is the input vector
	//x := 1/1 + math.Exp(-z)

}

func forwardPropagate(n Network) {

	weights := n.Weights
	layers := n.Layers

	// for i := range weights {
	// 	for j := range weights[i] {

	// 	}
	// }

}
