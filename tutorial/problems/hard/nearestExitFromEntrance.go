package hard

func nearestExit(maze [][]byte, entrance []int) int {
	rowLen := len(maze)
	colLen := len(maze[0])
	floor := '.'
	marked := '+' // for marking cells
	shortestDistanceToExit := 0
	directions := [][2]int{{0, 1}, {1, 0}, {-1, 0}, {0, -1}} // 4- directional path
	queue := [][2]int{[2]int(entrance)}                      // initializing the queue with the entrance
	maze[entrance[0]][entrance[1]] = byte(marked)            // marking the initial cell at first
	noPath := -1

	for len(queue) > 0 {
		shortestDistanceToExit++
		queueLen := len(queue)
		for index := 0; index < queueLen; index++ {
			currRow, currCol := queue[0][0], queue[0][1] // from current cell
			queue = queue[1:]                            // slicing the queue from the first cell
			// check in all 4 directions
			for _, dir := range directions {
				newRow := dir[0] + currRow
				newCol := dir[1] + currCol
				// general border edge case for stopping
				if newRow >= rowLen || newCol >= colLen || newRow < 0 || newCol < 0 || maze[newRow][newCol] == byte(marked) {
					continue
				}
				// if we reach goal just return the distance
				if (newRow == rowLen-1 || newCol == colLen-1 || newRow == 0 || newCol == 0) && maze[newRow][newCol] == byte(floor) {
					return shortestDistanceToExit
				}

				// for moving and appending to queue
				if maze[newRow][newCol] == byte(floor) {
					maze[newRow][newCol] = byte(marked)
					queue = append(queue, [2]int{newRow, newCol})
				}

			}
		}
	}

	return noPath
}
