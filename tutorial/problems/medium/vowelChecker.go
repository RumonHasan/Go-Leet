package problems

import "strings"

func vowelSpellchecker(wordList, queries []string) []string {
	var result []string
	lowMap := make(map[string]string)
	maskMap := make(map[string]string)
	wordMap := make(map[string]bool)

	for _, word := range wordList {
		wordMap[word] = true
	}
	// vowel map for vowel checking
	vowelMap := map[rune]bool{
		'a': true,
		'e': true,
		'i': true,
		'o': true,
		'u': true,
	}

	// function to replace vowel positions with asterisk
	mask := func(word string) string {
		var maskedWord string
		for _, currChar := range word {
			if _, found := vowelMap[currChar]; found {
				maskedWord += "*"
			} else {
				maskedWord += string(currChar)
			}
		}
		return maskedWord
	}

	// get the lower case words to be the key with their main word
	for _, word := range wordList {
		lowWord := strings.ToLower(word)
		if _, ok := lowMap[lowWord]; !ok {
			lowMap[lowWord] = word
		}
		maskedWord := mask(lowWord)
		if _, ok := maskMap[maskedWord]; !ok { // update only if it does not exist
			maskMap[maskedWord] = word
		}
	}

	// main iteration for queries and insertion
	for _, query := range queries {
		lowWord := strings.ToLower(query)
		if wordMap[query] {
			result = append(result, query)
		} else if value, found := lowMap[lowWord]; found {
			result = append(result, value)
		} else if value, found := maskMap[mask(lowWord)]; found {
			result = append(result, value)
		} else {
			result = append(result, "")
		}
	}

	return result
}
