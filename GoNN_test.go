package GoNN

import (
	"testing"
)

func TestGoNN(t *testing.T) {

	var n = CreateNetwork(2, 2, []int{3, 3}, 1)

	for i := 0; i < len(n.Weights); i++ {
		for j := 0; j < len(n.Weights[i]); j++ {
			for k := 0; k < len(n.Weights[i][j]); k++ {
				print(int(n.Weights[i][j][k]))
			}
			println()
		}
		println()
	}
}
