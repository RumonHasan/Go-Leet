package problems

import "sort"

type SearchTrie struct {
	children    map[byte]*SearchTrie // for byte range between 0 -26
	suggestions []string
}

func NewSearchTrie() *SearchTrie {
	return &SearchTrie{
		children: make(map[byte]*SearchTrie),
	}
}

// creating the trie but injecting suggestion with limit based on every prefix
func (r *SearchTrie) addProduct(product string) {
	currNode := r
	for _, char := range product {
		currByte := byte(char - 'a')
		if _, found := currNode.children[currByte]; !found {
			currNode.children[currByte] = NewSearchTrie()
		}
		currNode = currNode.children[currByte] // moving the pointer to the next child pointer
		// injecting suggestion words based on length constraints
		if len(currNode.suggestions) < 3 {
			currNode.suggestions = append(currNode.suggestions, product)
		}
	}
}

func suggestedProducts(products []string, searchWord string) [][]string {
	sort.Strings(products)
	rootTrie := NewSearchTrie()
	result := [][]string{}
	// creating the root trie for search lookups
	for _, word := range products {
		rootTrie.addProduct(word)
	}
	// main iteration to get the search results
	for _, char := range searchWord {
		byteChar := byte(char - 'a')
		if rootTrie != nil {
			rootTrie = rootTrie.children[byteChar] // descent first then append the suggestions
		}
		if rootTrie != nil {
			result = append(result, rootTrie.suggestions)
		}
		if rootTrie == nil {
			result = append(result, []string{})
		}
	}
	return result
}
