package hard

type RootNode struct {
	children [26]*RootNode
	sum      int
}

type MapSum struct {
	rootNode *RootNode
	vals     map[string]int
}

func NewNodeRoot() *RootNode {
	return &RootNode{}
}

// func Constructor() MapSum {
// 	return MapSum{
// 		rootNode: NewNodeRoot(),
// 		vals:     make(map[string]int),
// 	}
// }

func (this *MapSum) Insert(key string, val int) {
	existingKeyVal := this.vals[key]
	delta := val - existingKeyVal // record the delta value

	rootNode := this.rootNode
	for index := 0; index < len(key); index++ {
		keyIdx := key[index] - 'a'

		// if the key does not exist then yea
		if rootNode.children[keyIdx] == nil {
			rootNode.children[keyIdx] = NewNodeRoot()
		}

		rootNode = rootNode.children[keyIdx]
		rootNode.sum += delta
	}
	this.vals[key] = val // updating the value to the latest
}

func (this *MapSum) Sum(prefix string) int {
	rootNode := this.rootNode
	prefLen := len(prefix)

	for index := 0; index < prefLen; index++ {
		keyIdx := prefix[index] - 'a'

		if rootNode.children[keyIdx] == nil {
			return 0
		}

		rootNode = rootNode.children[keyIdx]
	}

	return rootNode.sum
}
