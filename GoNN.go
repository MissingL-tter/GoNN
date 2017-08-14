package GoNN

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// Network defines the structure of the network
type Network struct {
	Layers [][]*Node
}

// Node defines a node in the network
type Node struct {
	In   []*Edge
	Bias float64
	Out  []*Edge
}

// Edge defines a connection between nodes in the network
type Edge struct {
	Start  *Node
	Weight float64
	End    *Node
}

// CreateNetwork returns a Network struct.
//
// layers is the description of the network such that CreateNetwork(5, 3, 2) returns a network with 5 input nodes, 1 hidden layer with 3 nodes, and 2 output nodes
// layers requires at least two inputs but a variadic has been used for clarity in building the network
func CreateNetwork(nodeCounts ...int) (*Network, error) {

	// First verify that we have at least input and output nodes for out network.
	if len(nodeCounts) < 2 {
		return nil, errors.New("Cannot build an empty network. CreateNetwork() must have at least 2 parameters")
	}
	for _, l := range nodeCounts {
		if l < 1 {
			return nil, errors.New("Cannot build a layer of zero nodes. All parameters must be greater than 0")
		}
	}

	// Create an array to represent each layer and
	layers := make([][]*Node, len(nodeCounts))

	// for each layer make the array for the nodes in that layer.
	for l := range layers {
		layers[l] = make([]*Node, nodeCounts[l])

		// Initialize each node's bias from a normal distribution
		for n := range layers[l] {
			if l != 0 {
				layers[l][n] = &Node{nil, rand.NormFloat64(), nil}
			} else {
				// unless they are input nodes, which get initialized at 0.
				layers[l][n] = &Node{nil, 0, nil}
			}
		}
	}

	// In every layer that is not an input,
	for l := 1; l < len(layers); l++ {

		// for each node in that layer
		for n := range layers[l] {

			// create an edge to each node in the layer before it and initialize its weight from a normal distribution
			for k := range layers[l-1] {
				edge := &Edge{layers[l-1][k], rand.NormFloat64(), layers[l][n]}
				layers[l-1][k].Out = append(layers[l-1][k].Out, edge)
				layers[l][n].In = append(layers[l][n].In, edge)
			}
		}
	}

	// Return a pointer to the network
	return &Network{layers}, nil
}

// createNetworkDebug returns a Network struct where every bias is initialized at 1, and every weight at 2
func createNetworkDebug(nodeCounts ...int) (*Network, error) {

	// First verify that we have at least input and output nodes for out network.
	if len(nodeCounts) < 2 {
		return nil, errors.New("Cannot build an empty network. CreateNetwork() must have at least 2 parameters")
	}
	for _, l := range nodeCounts {
		if l < 1 {
			return nil, errors.New("Cannot build a layer of zero nodes. All parameters must be greater than 0")
		}
	}

	// Create an array to represent each layer and
	layers := make([][]*Node, len(nodeCounts))

	// for each layer make the array for the nodes in that layer.
	for l := range layers {
		layers[l] = make([]*Node, nodeCounts[l])

		// Initialize each node's bias at 1
		for n := range layers[l] {
			if l != 0 {
				layers[l][n] = &Node{nil, 1, nil}
			} else {
				// unless they are input nodes, which get initialized at 0.
				layers[l][n] = &Node{nil, 0, nil}
			}
		}
	}

	// In every layer that is not an input,
	for l := 1; l < len(layers); l++ {

		// for each node in that layer
		for n := range layers[l] {

			// create an edge to each node in the layer before it and initialize its weight at 2
			for k := range layers[l-1] {
				edge := &Edge{layers[l-1][k], 2, layers[l][n]}
				layers[l-1][k].Out = append(layers[l-1][k].Out, edge)
				layers[l][n].In = append(layers[l][n].In, edge)
			}
		}
	}

	// Return a pointer to the network
	return &Network{layers}, nil
}

// ForwardPropagate takes some input and processes it through the network to return the output
//
// The output returned will be an array of size corresponding to the number of nodes in the output layer
func (net Network) ForwardPropagate(layerIn []float64) ([]float64, error) {

	// Verify that we have enough inputs to satisfy the input layer
	if len(layerIn) != len(net.Layers[0]) {
		return nil, fmt.Errorf("Length of input array must be the length of the input layer. Expected:%v Got:%v", len(net.Layers[0]), len(layerIn))
	}

	// For each layer that is not the input layer
	for _, layer := range net.Layers[1:] {

		// create an array to store its output
		layerOut := []float64{}

		// and for each node in that layer
		for _, node := range layer {

			// calculate the dot product of all incoming nodes, and their edges weights.
			dotProd := 0.
			for e, edge := range node.In {
				dotProd += layerIn[e] * edge.Weight
			}

			// Then add the node's bias to the dot product, apply the sigmoid function,
			// and store the result in the layer's output
			layerOut = append(layerOut, sigmoid(dotProd+node.Bias))
		}

		// At the end of the layer, set the input for the next layer as this layer's output.
		layerIn = layerOut
	}

	// Return the final output once we have traversed all layers
	return layerIn, nil

}

// BackPropagate takes some output compared to what was expected by the input and uses stochastic gradient descent to minimize the loss function
func (net Network) BackPropagate() []float64 {
	return []float64{0.}
}

// Stochastic Gradient Descent
func sgd() {

}

// The activation function used to squash the input to between 0 and 1
func sigmoid(z float64) float64 {

	return 1 / (1 + math.Exp(-z))

}

// The derivative of the sigmoid function
func sigPrime(z float64) float64 {

	return sigmoid(z) * (1 - sigmoid(z))

}
