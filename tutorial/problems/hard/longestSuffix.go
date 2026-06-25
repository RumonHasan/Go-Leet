package hard

type RootTrieNode struct {
	children  map[byte]*RootTrieNode
	bestIndex int
}

func NewTrieNode() *RootTrieNode {
	return &RootTrieNode{
		children:  make(map[byte]*RootTrieNode),
		bestIndex: -1, // initializing the root with -1 first for non missing case
	}
}

// adding the based on suffix based insertion
func (r *RootTrieNode) addWord(word string, currWordIndex int, wordsContainer []string) {
	root := r
	// initial check
	if root.bestIndex == -1 || len(word) < len(wordsContainer[root.bestIndex]) {
		root.bestIndex = currWordIndex
	}
	// reverse iteration for suffix tree
	for index := len(word) - 1; index >= 0; index-- {
		currChar := word[index]
		byteChar := byte(currChar)
		// if the node is found
		if _, ok := root.children[byteChar]; !ok {
			root.children[byteChar] = NewTrieNode()
		}
		// iterate to the next root char
		root = root.children[byteChar]
		// updating the length
		if root.bestIndex == -1 || len(word) < len(wordsContainer[root.bestIndex]) { // updating with the index of the shortest word
			root.bestIndex = currWordIndex
		}
	}
}

func stringIndices(wordsContainer []string, wordsQuery []string) []int {
	result := make([]int, len(wordsQuery))
	// adding each word and building the suffix trie
	containerNode := NewTrieNode()
	for index, word := range wordsContainer {
		containerNode.addWord(word, index, wordsContainer)
	}
	// check the suffix tree words
	for index, query := range wordsQuery {
		// explore every root children from the beginning
		currRootNode := containerNode

		for index := len(query) - 1; index >= 0; index-- {
			currQueryByteChar := byte(query[index])
			// if no child is found simply break out since the word suffix can no longer be found
			if _, found := currRootNode.children[currQueryByteChar]; !found {
				break
			}
			currRootNode = currRootNode.children[currQueryByteChar] // iterate to the child node if its there
		}
		result[index] = currRootNode.bestIndex // adding the best index so for from the suffix tree

	}
	return result

}
