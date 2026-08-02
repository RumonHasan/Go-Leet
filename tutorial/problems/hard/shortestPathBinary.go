package hard

func shortestPathBinaryMatrix(grid [][]int) int {
	rowLen := len(grid)
	colLen := len(grid[0])
	// 8 way directions for bfs traversal
	directions := [][2]int{{0, 1}, {0, -1}, {-1, 0}, {1, 0}, {1, 1}, {-1, -1}, {-1, 1}, {1, -1}}
	queue := [][3]int{{0, 0, 1}} // main queue has 3 dimenions per cell since we store the distance at each cell
	if grid[0][0] == 0 {
		grid[0][0] = 1 // need to mark the starting cell
	} else {
		return -1
	}

	// bfs keeps going till the there is a queue length
	for len(queue) > 0 {
		queueSize := len(queue)
		for index := 0; index < queueSize; index++ {
			currentCell := queue[0]
			currRowIndex, currColIndex, currDist := currentCell[0], currentCell[1], currentCell[2]
			queue = queue[1:] // trimming the queue up front from header cell

			if currRowIndex == rowLen-1 && currColIndex == colLen-1 { // if target is reached then return the min distance
				return currDist
			}
			// check in all 8 directions
			for _, dir := range directions {
				newRow, newCol := dir[0]+currRowIndex, dir[1]+currColIndex
				// base case for boundary checks and occupied cell value
				if newRow < 0 || newCol < 0 || newRow >= rowLen || newCol >= colLen || grid[newRow][newCol] == 1 {
					continue
				}
				if grid[newRow][newCol] == 0 {
					grid[newRow][newCol] = 1                                    // marked by visited
					queue = append(queue, [3]int{newRow, newCol, currDist + 1}) // adding the distance to the new chain
				}
			}
		}
	}
	return -1
}
