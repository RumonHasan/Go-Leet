package problems

import "sort"

// medium problem using prefix indices and using binary search for getting the smallest earliest index from the previous point
func numberOfMatchingSub(s string, words []string) int {
	indexMap := make(map[byte][]int) // will contain the byte index of each letter
	count := 0
	// adding indices
	for index, char := range s {
		indexMap[byte(char)] = append(indexMap[byte(char)], index) // should be in relative order
	}
	// main iteration to go word by word to check the presence
	for index := 0; index < len(words); index++ {
		currWord := words[index]
		currWordMatch := true
		prevIndex := -1
		for _, currChar := range currWord {
			// check for individual char

			if _, found := indexMap[byte(currChar)]; found {
				currIndexList := indexMap[byte(currChar)]
				// binary search block
				nextPrevIndex := sort.Search(len(currIndexList), func(i int) bool {
					currIndexFromList := currIndexList[i]
					return prevIndex < currIndexFromList
				})
				// final edge case
				if nextPrevIndex == len(currIndexList) { // when the current range is bigger
					currWordMatch = false
					break
				} else {
					prevIndex = currIndexList[nextPrevIndex]
				}
			} else {
				currWordMatch = false
			}
		}
		if currWordMatch {
			count++
		}
	}

	return count
}
