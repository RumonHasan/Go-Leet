package hard

// primary struct
type WordDictionary struct {
	children map[byte]*WordDictionary
	isWord   bool
}

// constructor supporter
func CreateNewWordTrie() *WordDictionary {
	return &WordDictionary{
		children: make(map[byte]*WordDictionary),
	}
}

// references the main CreateNewWordTrie function to create a new trie
func Constructor() WordDictionary {
	return *CreateNewWordTrie()
}

// breaks down the word and adds its respective cluster based on chars
func (this *WordDictionary) AddWord(word string) {
	currNode := this
	for _, curr := range word {
		currChar := byte(curr)
		if _, found := currNode.children[currChar]; !found {
			currNode.children[currChar] = CreateNewWordTrie() // create new char node cluster
		}
		// append the node to the next location
		currNode = currNode.children[currChar]
	}

	currNode.isWord = true // marking the final word as true
}

func (this *WordDictionary) Search(word string) bool {
	// main base case
	if len(word) == 0 {
		return this.isWord // checking whether the word actually ended at this word or not
	}
	currWordChar := word[0]
	if currWordChar == '.' {
		// explore all the child nodes
		for _, child := range this.children {
			if child.Search(word[1:]) {
				return true
			}
		}
		return false
	}
	next, ok := this.children[currWordChar] // next node
	// if it does not exists
	if !ok {
		return false
	}
	return next.Search(word[1:]) // iterates and moves to the next char cluster
}

/**
 * Your WordDictionary object will be instantiated and called as such:
 * obj := Constructor();
 * obj.AddWord(word);
 * param_2 := obj.Search(word);
 */
