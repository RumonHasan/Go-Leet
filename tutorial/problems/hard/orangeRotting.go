package hard

// bfs based structure problem with prefixed queue structure
func orangesRotting(grid [][]int) int {
	rowLen := len(grid)
	colLen := len(grid[0])
	queue := [][2]int{}                               // for storing indices of the rotten oranges
	freshCount := 0                                   // fresh orange count
	dir := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}} // 4 directional movement from affected cell
	minutes := 0
	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			currCell := grid[row][col]
			if currCell == 0 {
				continue
			}
			if currCell == 2 {
				queue = append(queue, [2]int{row, col})
			}
			if currCell == 1 {
				freshCount++
			}
		}
	}
	if freshCount == 0 {
		return 0
	}
	for len(queue) > 0 && freshCount > 0 {
		queueSize := len(queue) // to control and fix the queue size here so this minute layer is preserved
		for index := 0; index < queueSize; index++ {
			currQueueCell := queue[0] // already rotten
			poppedQueue := queue[1:]  // removing the source infected top cell since it will be processed
			for _, direction := range dir {
				currRow, currCol := currQueueCell[0], currQueueCell[1]
				row, col := direction[0], direction[1]
				newRow := currRow + row
				newCol := currCol + col
				if newRow < 0 || newCol < 0 || newRow >= rowLen || newCol >= colLen || grid[newRow][newCol] == 0 { // adding boundaries
					continue
				}
				possibleFreshCell := grid[newRow][newCol]
				if possibleFreshCell == 1 {
					grid[newRow][newCol] = 2
					freshCount--
					poppedQueue = append(queue, [2]int{newRow, newCol}) // new potenttial start for rotten cell check
				}
			}
			queue = poppedQueue
		}
		minutes++
	}
	if freshCount == 0 {
		return minutes
	}
	return -1
}
