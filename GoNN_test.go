package GoNN

import (
	"fmt"
	"testing"
)

func TestNetworkStruct(t *testing.T) {

	n := CreateNetwork(3, 1, []int{4}, 1)
	n.Init()

	for i := range n.Weights {

		if i == 0 {
			print(len(n.Biases[i]))
			print(" input  nodes with bias: ")
			for k := range n.Biases[i] {
				fmt.Printf("%.2f ", n.Biases[i][k])
			}
			println()

		}

		if i < len(n.Weights)-1 {
			print(len(n.Weights[i]) / len(n.Biases[i]))
			print(" hidden nodes with bias: ")
			for k := range n.Biases[i+1] {
				fmt.Printf("%.2f ", n.Biases[i+1][k])
			}
			print(" and weights to: ")

		} else {
			print(len(n.Weights[i]) / len(n.Biases[i]))
			print(" output nodes with bias: ")
			for k := range n.Biases[i+1] {
				fmt.Printf("%.2f ", n.Biases[i+1][k])
			}
			print(" and weights to: ")
		}
		for j := range n.Weights[i] {
			fmt.Printf("%.2f ", n.Weights[i][j])
		}
		println()
	}
	println()
}
