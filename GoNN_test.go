package GoNN

import (
	"fmt"
	"os"
	"testing"
	"text/tabwriter"
)

func TestNetworkStruct(t *testing.T) {

	structure := []int{2, 3, 2}

	n, err := createNetworkDebug(structure...)
	if err != nil {
		t.Errorf(err.Error())
	}

	fmt.Print("Testing on:")
	for _, i := range structure {
		fmt.Printf(" %v", i)
	}
	fmt.Println()

	for i := range n.Layers {
		if len(n.Layers[i]) != structure[i] {
			t.Errorf("Incorrect size of Layer %v | Have: %v | Want: %v", i, len(n.Layers[i]), structure[i])
		}
	}

	if !t.Failed() {
		fmt.Println("No errors found. All layers correctly sized.")
	}

}

func TestNetworkPrint(t *testing.T) {

	n, err := createNetworkDebug(2, 3, 2)
	if err != nil {
		t.Errorf(err.Error())
	}

	writer := tabwriter.NewWriter(os.Stdout, 0, 0, 0, ' ', tabwriter.Debug)
	fmt.Fprintf(writer, "Nodes \t Bias \t Weights \t \n")
	fmt.Fprint(writer, "\t \t \t \n")

	for i := range n.Layers {

		if i == 0 {
			fmt.Fprintf(writer, "%v Input ", len(n.Layers[i]))
			for j := range n.Layers[i] {
				fmt.Fprint(writer, "\t \t ")
				for k := range n.Layers[i][j].Out {
					fmt.Fprintf(writer, "%+.2f ", n.Layers[i][j].Out[k].Weight)
				}
				fmt.Fprint(writer, "\t \n")
			}
			fmt.Fprint(writer, "\t \t \t \n")

		} else if i < len(n.Layers)-1 {
			fmt.Fprintf(writer, "%v Hidden ", len(n.Layers[i]))
			for j := range n.Layers[i] {
				fmt.Fprintf(writer, "\t %+.2f \t ", n.Layers[i][j].Bias)
				for k := range n.Layers[i][j].Out {
					fmt.Fprintf(writer, "%+.2f ", n.Layers[i][j].Out[k].Weight)
				}
				fmt.Fprint(writer, "\t \n")
			}
			fmt.Fprint(writer, "\t \t \t \n")
		} else {
			fmt.Fprintf(writer, "%v Output ", len(n.Layers[i]))
			for j := range n.Layers[i] {
				fmt.Fprintf(writer, "\t %+.2f \t \t \n", n.Layers[i][j].Bias)
			}
		}
	}
	writer.Flush()
}

func TestForwardPropagate(t *testing.T) {

	n, _ := createNetworkDebug(2, 3, 2)

	input := []float64{1, 1}

	out, err := n.ForwardPropagate(input)
	if err != nil {
		t.Errorf(err.Error())
	}

	writer := tabwriter.NewWriter(os.Stdout, 0, 0, 0, ' ', tabwriter.Debug)
	fmt.Fprint(writer, "Outputs ")
	for o := range out {
		fmt.Fprintf(writer, "\t %.5f \t \n", out[o])
	}
	writer.Flush()
}
