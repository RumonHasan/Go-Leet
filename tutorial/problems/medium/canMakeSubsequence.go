package problems

func canMakeSubsequence(str1 string, str2 string) bool {
	strLen := len(str1)
	strLenTwo := len(str2)
	// function to locate the next cyclic char
	cyclicCharFinder := func(currChar byte) byte {
		return (((currChar - 'a') + 1) % 26) + 'a'
	}
	strIndex := 0
	strIndexTwo := 0
	// main iteration
	for strIndex < strLen && strIndexTwo < strLenTwo {
		currStrOneChar := str1[strIndex]
		currStrTwoChar := str2[strIndexTwo]

		// main check whether the character is a next or not
		if currStrOneChar == currStrTwoChar {
			strIndex++
			strIndexTwo++
		} else if cyclicCharFinder(currStrOneChar) == currStrTwoChar {
			strIndexTwo++
			strIndex++
		} else {
			strIndex++
		}
	}
	if strIndexTwo != strLenTwo {
		return false
	}

	return true
}
