package problems

import (
	"slices"
	"strings"
)

// medium level : hashtable table basics
func largestWordCount(messages []string, senders []string) string {
	n := len(messages) // general length
	wordMap := make(map[string]int)
	maxCount := 0
	maxNameArray := []string{}
	wordCount := func(word string) int {
		return len(strings.Split(word, " "))
	}
	// main iteration to check for similar names and words
	for index := 0; index < n; index++ {
		currSender := senders[index]
		currMessage := messages[index]

		if existingMsgWordCount, found := wordMap[currSender]; found {
			currMessageWordCount := wordCount(currMessage)
			newCount := existingMsgWordCount + currMessageWordCount
			wordMap[currSender] = newCount
		} else {
			wordMap[currSender] = wordCount(currMessage)
		}

		maxCount = max(maxCount, wordMap[currSender]) // updating to the biggest

	}
	for key, value := range wordMap {
		if value == maxCount {
			maxNameArray = append(maxNameArray, key)
		}
	}
	slices.Sort(maxNameArray)
	return maxNameArray[len(maxNameArray)-1]
}
