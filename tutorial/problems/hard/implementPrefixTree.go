package hard

type Trie struct {
	children map[byte]*Trie
	isEnd    bool
}

// will use this method for creating the children since it will give the reference
func NewTrie() *Trie {
	return &Trie{
		children: make(map[byte]*Trie),
	}
}

// no reference
func Constructor() Trie {
	return Trie{
		children: make(map[byte]*Trie),
	}
}

func (this *Trie) Insert(word string) {
	root := this
	for _, char := range word {
		byteChar := byte(char - 'a')

		if _, found := root.children[byteChar]; !found {
			root.children[byteChar] = NewTrie()
		}
		root = root.children[byteChar] // moving down the root
	}
	root.isEnd = true // end of word
}

// entire distinct word
func (this *Trie) Search(word string) bool {
	root := this
	for _, char := range word {
		byteChar := byte(char - 'a')
		if _, found := root.children[byteChar]; !found {
			return false
		}
		root = root.children[byteChar] // keep going till the end
	}

	return root.isEnd
}

// thi is on every step
func (this *Trie) StartsWith(prefix string) bool {
	root := this
	for _, char := range prefix {
		byteChar := byte(char - 'a')
		// returns on every found
		nextNode, found := root.children[byteChar]
		if !found {
			return false
		}
		root = nextNode
	}
	return true // thats all I need since it is a complete prefix
}

/**
 * Your Trie object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Insert(word);
 * param_2 := obj.Search(word);
 * param_3 := obj.StartsWith(prefix);
 */
