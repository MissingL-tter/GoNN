package GoNN

import (
	"fmt"
	"testing"
)

func TestNetworkStruct(t *testing.T) {

	n := CreateNetwork(2, 2, []int{2, 2}, 2)
	n.InitDebug()

	for i := range n.Weights {

		if i == 0 {
			fmt.Print(len(n.Biases[i]))
			fmt.Printf("%25s", " input nodes with bias: ")
			for k := range n.Biases[i] {
				fmt.Printf("%+.2f ", n.Biases[i][k])
			}
			fmt.Println()
		}

		if i < len(n.Weights)-1 {
			fmt.Print(len(n.Weights[i]) / len(n.Biases[i]))
			fmt.Print(" hidden nodes with bias: ")
			for k := range n.Biases[i+1] {
				fmt.Printf("%+.2f ", n.Biases[i+1][k])
			}
			fmt.Print("and weight: ")
			for j := range n.Weights[i] {
				fmt.Printf("%+.2f ", n.Weights[i][j])
			}

		} else {
			fmt.Print(len(n.Weights[i]) / len(n.Biases[i]))
			fmt.Print(" output nodes with bias: ")
			for k := range n.Biases[i+1] {
				fmt.Printf("%+.2f ", n.Biases[i+1][k])
			}
			fmt.Print("and weight: ")
			for j := range n.Weights[i] {
				fmt.Printf("%+.2f ", n.Weights[i][j])
			}
		}
		fmt.Println()
	}
}

func TestForwardPropagate(t *testing.T) {

	n := CreateNetwork(2, 2, []int{2, 2}, 2)
	n.InitDebug()

	for i := range n.Biases[0] {
		n.Biases[0][i] = 1
	}

	out := n.ForwardPropagate()

	for o := range out {
		fmt.Printf("%.2f ", out[o])
	}
	fmt.Println()
}
