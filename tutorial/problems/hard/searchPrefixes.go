package hard

// main basic struct for creating root trie and pass int score
type Node struct {
	children map[byte]*Node
	pass     int
}

// new root making function
func NewRootNode() *Node {
	return &Node{
		children: make(map[byte]*Node),
		pass:     0,
	}
}

// primary function for injecting the letter by letter to trie and create the prefix tree
func (r *Node) addWord(word string) {
	currNode := r
	for _, currChar := range word {
		byteChar := byte(currChar - 'a')
		// if its not found
		if _, ok := currNode.children[byteChar]; !ok {
			currNode.children[byteChar] = NewRootNode()
		}
		// update the pass int on every root if its part of the same parent word
		currNode = currNode.children[byteChar] // next node
		currNode.pass++
	}
}

// function to make prefix tries and store the letters of the word per root
func sumPrefixScores(words []string) []int {
	passResult := make([]int, len(words))
	rootNode := NewRootNode()
	for _, word := range words {
		rootNode.addWord(word)
	}
	for index, word := range words {
		rootNodePassValue := 0
		currRootNode := rootNode
		for index := 0; index < len(word); index++ {
			currChar := byte(word[index] - 'a')
			currRootNode = currRootNode.children[currChar]
			currRootNodePassVal := currRootNode.pass
			rootNodePassValue += currRootNodePassVal

		}
		passResult[index] = rootNodePassValue
	}

	return passResult
}
