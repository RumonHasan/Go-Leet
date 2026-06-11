package problems

// getting the longest wiggle value for alternating signs
func wiggleSubsequence(nums []int) int {
	moveUp := 1
	moveDown := 1
	lockUp := false
	lockDown := false

	// booleans are not doing antyhing

	for index := 1; index < len(nums); index++ {
		currVal := nums[index]
		prevVal := nums[index-1]

		if currVal > prevVal && !lockUp {
			moveUp = moveDown + 1 // its cumulative
			lockUp = true
			lockDown = false
		} else if prevVal > currVal && !lockDown {
			moveDown = moveUp + 1
			lockDown = true
			lockUp = false
		}

	}

	return max(moveDown, moveUp)
}

// [0, 0]
