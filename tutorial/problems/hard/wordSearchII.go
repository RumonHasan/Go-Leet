package hard

// Recursive trie node cluster for checking the prefix char chain of each word
type WordTrieNode struct {
	children map[byte]*WordTrieNode
	isWord   bool
	word     string // word for storage at the source of each node cluster
}

// func to make a new trie node
func NewWordTrie() *WordTrieNode {
	return &WordTrieNode{
		children: make(map[byte]*WordTrieNode),
	}
}

// this function extracts the char of every word then creates the node cluster
func (w *WordTrieNode) addNewWord(word string) {
	currNode := w // holds the currnode
	// now for all character in the word
	for _, currChar := range word {
		currByteChar := byte(currChar)
		// if its not found u add a new cluster of char from this point
		if _, found := currNode.children[currByteChar]; !found {
			currNode.children[currByteChar] = NewWordTrie() // new word trie method
		}
		// general node iteration to the next one
		currNode = currNode.children[currByteChar]
	}
	// after the char end we can simply delcare true for word found and store the word at that point of node cluster
	currNode.isWord = true
	currNode.word = word // storing the word at the point of node cluster
}

// primary function that will create the trie nodes then return the string array
func findWords(board [][]byte, words []string) []string {
	// declare new word trie then start injecting the trie pattern
	rootNode := NewWordTrie()
	for _, word := range words {
		rootNode.addNewWord(word)
	}
	// col and row limit borders
	rowLen := len(board)
	colLen := len(board[0])
	foundWords := make(map[string]bool) // will contain the found words that are to be injected into the final collection
	marker := '#'

	// main dfs functionality which will be similar to that of word search
	var dfs func(currRow, currCol int, currNode *WordTrieNode)
	dfs = func(currRow, currCol int, currNode *WordTrieNode) {
		// base case for when its out of boundary or its a marked char chain then return
		if currRow < 0 || currCol < 0 || currRow >= rowLen || currCol >= colLen || board[currRow][currCol] == byte(marker) {
			return
		}
		// get char
		currChar := board[currRow][currCol]
		//if char is not in trie node then return
		if _, found := currNode.children[currChar]; !found {
			return
		}
		// advance to the next node if the char is found in the cluster
		nextNode := currNode.children[currChar]
		if nextNode.isWord {
			foundWords[nextNode.word] = true // update the found word but do not return since the last char could be other chars
		}
		// declare the char as marked
		board[currRow][currCol] = byte(marker)
		directions := [][]int{{0, 1}, {1, 0}, {-1, 0}, {0, -1}}
		// quadri directional dfs call in alll 4 direction
		for _, dir := range directions {
			row, col := dir[0], dir[1]
			nextRow := currRow + row
			nextCol := currCol + col
			dfs(nextRow, nextCol, nextNode) // making it as next node for iteration
		}
		// backtracking redeclaring char for previous location exploration
		board[currRow][currCol] = currChar
	}

	// main for loop to traverse the board and inject the char
	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			dfs(row, col, rootNode)
		}
	}
	// collecting the final found words collection and injecting it into the final collection
	finalCollection := []string{}
	for word, _ := range foundWords {
		finalCollection = append(finalCollection, word)
	}
	// returning the final collection
	return finalCollection

}
