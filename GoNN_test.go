package GoNN

import (
	"testing"
)

func TestNetworkStruct(t *testing.T) {

	n := CreateNetwork(3, 1, []int{4}, 1)

	for i := range n.Weights {

		if i == 0 {
			print(len(n.Layers[i]))
			print(" input  nodes with bias: ")
			for k := range n.Layers[i] {
				print(int(n.Layers[i][k]))
			}
			println()

		}

		if i < len(n.Weights)-1 {
			print(len(n.Weights[i]) / len(n.Layers[i]))
			print(" hidden nodes with bias: ")
			for k := range n.Layers[i+1] {
				print(int(n.Layers[i+1][k]))
			}
			print(" and weights to: ")

		} else {
			print(len(n.Weights[i]) / len(n.Layers[i]))
			print(" output nodes with bias: ")
			for k := range n.Layers[i+1] {
				print(int(n.Layers[i+1][k]))
			}
			print(" and weights to: ")
		}
		for j := range n.Weights[i] {
			print(int(n.Weights[i][j]))
		}
		println()
	}
	println()
}
