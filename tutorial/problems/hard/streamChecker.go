package hard

// will contain the premade root and the stream of bytes per search
type StreamChecker struct {
	root   *RootTrie
	stream []byte
}

// main root trie
type RootTrie struct {
	children map[byte]*RootTrie
	isWord   bool
}

func NewStreamTrie() *RootTrie {
	return &RootTrie{
		children: make(map[byte]*RootTrie),
	}
}

// will create the root trie here and inject all the words and form the suffix tree
func Constructor(words []string) StreamChecker {
	root := NewStreamTrie()
	for _, word := range words {
		nodeRoot := root
		//rev injection of every single char
		for index := len(word) - 1; index >= 0; index-- {
			byteChar := byte(word[index])

			if _, ok := nodeRoot.children[byteChar]; !ok {
				nodeRoot.children[byteChar] = NewStreamTrie()
			}
			nodeRoot = nodeRoot.children[byteChar]

		}
		nodeRoot.isWord = true
	}

	return StreamChecker{root: root, stream: []byte{}}
}

func (this *StreamChecker) Query(letter byte) bool {
	currRoot := this.root
	this.stream = append(this.stream, byte(letter)) // adding it letter by letter for query
	streamLen := len(this.stream)

	for streamIndex := streamLen - 1; streamIndex >= 0; streamIndex-- {
		currByteChar := byte(this.stream[streamIndex])

		if _, ok := currRoot.children[currByteChar]; ok {
			currRoot = currRoot.children[currByteChar]
		} else {
			return false
		}
		// only return true if the full word suffix is found.
		if currRoot.isWord {
			return true
		}
	}

	return false
}

/**
 * Your StreamChecker object will be instantiated and called as such:
 * obj := Constructor(words);
 * param_1 := obj.Query(letter);
 */
