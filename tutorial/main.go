package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"time"
)

type gasEngine struct {
	mpg       uint8
	gallons   uint8
	ownerInfo owner
}

type owner struct {
	name string
}

func (e gasEngine) milesLeft() uint8 {
	return e.gallons * e.mpg
}

var m = sync.RWMutex{} // RWMutex (can also use sync.Mutex)
var wg = sync.WaitGroup{}
var dbData = []string{"id1", "id2", "id3", "id4", "id5"}
var results = []string{}

func solutionWithMutex() {
	fmt.Println("=== THE SOLUTION: MUTEX LOCK/UNLOCK ===")

	t0 := time.Now()
	for i := 0; i < len(dbData); i++ {
		wg.Add(1)
		go dbCall(i)
	}
	wg.Wait()

	fmt.Printf("Total execution time: %v\n", time.Since(t0))
	fmt.Printf("The results are: %v\n", results)
	fmt.Printf("Successfully got %d results (no race condition!)\n", len(results))
	fmt.Println()
}

func dbCall(i int) {
	defer wg.Done()

	// Simulate DB delay
	var delay float32 = 2000 // Fixed delay for predictable output
	time.Sleep(time.Duration(delay) * time.Millisecond)

	fmt.Printf("The result from the database is: %s\n", dbData[i])

	// CRITICAL SECTION: Only one goroutine can execute this at a time
	m.Lock()                             // 🔒 LOCK: "I'm using the shared resource"
	results = append(results, dbData[i]) // Safe modification
	m.Unlock()                           // 🔓 UNLOCK: "I'm done, others can use it"
}

// ====================
// 3. HOW MUTEX WORKS INTERNALLY
// ====================

func howMutexWorks() {
	fmt.Println("=== HOW MUTEX WORKS INTERNALLY ===")

	var mutex sync.Mutex
	var counter int
	var wg2 sync.WaitGroup

	// Start 5 goroutines that increment a counter
	for i := 1; i <= 5; i++ {
		wg2.Add(1)
		go func(id int) {
			defer wg2.Done()

			fmt.Printf("Goroutine %d: Trying to acquire lock...\n", id)

			mutex.Lock() // 🔒 Only ONE goroutine can pass this line at a time
			fmt.Printf("Goroutine %d: 🔒 Got the lock! Current counter: %d\n", id, counter)

			// Critical section - only one goroutine executes this at a time
			oldValue := counter
			time.Sleep(100 * time.Millisecond) // Simulate work
			counter = oldValue + 1

			fmt.Printf("Goroutine %d: Incremented counter to %d\n", id, counter)
			fmt.Printf("Goroutine %d: 🔓 Releasing lock...\n", id)

			mutex.Unlock() // 🔓 Other goroutines can now acquire the lock

		}(i)
	}

	wg2.Wait()
	fmt.Printf("Final counter value: %d (should be 5)\n", counter)
	fmt.Println()
}

var MAX_CHICKEN_PRICE float64 = 5

func checkChickenPrices(website string, chickenChannel chan string) {
	for {
		time.Sleep(time.Second + 1)
		var chickenPrice = rand.Float32() * 20
		if chickenPrice <= float32(MAX_CHICKEN_PRICE) {
			chickenChannel <- website
			break
		}
	}
}

func main() {

	// testing channels
	var chickenChannel = make(chan string)
	var tofuChannel = make(chan string)
	var websites = []string{"google", "walmart"}
	for i := range websites {
		go checkChickenPrices(websites[i], chickenChannel)
	}

	var newChannel = make(chan int) // this channel can only a single syntax value
	newChannel <- 1                 // adding one to a channel
	var extractedChannelVal = <-newChannel

	var myEngine gasEngine = gasEngine{25, 15, owner{"Rumon"}}

	// fmt.Println("Hello World")
	array()

	// var intNum int = 32676

	// fmt.Println((intNum))

	// var myString string = "Hello" + "" + "World"
	// fmt.Println(myString)

	// fmt.Println(len("test"))

	// fmt.Println(utf8.RuneCountInString("Y"))

	// var my bool = false

	// myVar := "string"
	var printValue string = "Hellow World"
	var num int = 11
	var denominator int = 2
	var res, remainder, err = intDivision(num, denominator)
	if err != nil {
		fmt.Println((err.Error()))
	} else if remainder == 0 {
		fmt.Println("The result is", res)
	}
	fmt.Printf("The result of the int division is %v with remainder %v", res, remainder)
	printMe(printValue)

}

func printMe(printValue string) {
	fmt.Println(printValue)
}

func intDivision(numerator int, denominator int) (int, int, error) {
	var err error
	if denominator == 0 {
		err = errors.New("cannot divide by zero")
		return 0, 0, err
	}
	var result int = numerator / denominator
	var remainder int = numerator % denominator
	return result, remainder, err
}

func array() {
	var intArr []int32 = []int32{1, 3, 444} // not adding length value turns it into a slice
	intArr = append(intArr, 7)

	var intNewSlice []int32 = []int32{3, 4, 4}
	intNewSlice = append(intNewSlice, intArr...)
	fmt.Println(intNewSlice)

	// var intSlice3 []int32 = make([]int32, 3, 8)

	// // maps
	// var myMap map[string]uint8 = make(map[string]uint8)

	var myMap2 = map[string]uint8{"Adam": 23, "Sarah": 43}

	var age, ok = myMap2["Adam"]

	if ok {
		fmt.Printf("The age is %v\n", age)
	} else {
		fmt.Println("Invalid key")
	}

	for name := range myMap2 {
		fmt.Printf("%v\n", name)
	}

	// for i, v := range intArr {

	// }

	// while loop equivalent in go using for
	var i int = 0
	for i < 10 {
		fmt.Println((i))
		i = i + 1
	}

	// omitting the condition and using break
	for {
		if i >= 10 {
			break
		}
		fmt.Println(i)
		i = i + 1
	}

	// traditional
	var indexCheck int = 100
	for i := range indexCheck {
		fmt.Println(i)
	}
}

// an array with make preset capacity will take shorter time to append elements

func timeLoop(slice []int, n int) time.Duration {
	var t0 = time.Now()
	for len(slice) < n {
		slice = append(slice, 1)
	}
	return time.Since(t0)
}

func stringCheck() {
	var myString = "resume"
	var indexed = myString[0]
	fmt.Printf("%v, %T\n", indexed, indexed)
	for i, v := range myString {
		fmt.Println(i, v)
	}
}

// getting the max and min rows in order to calculate the area of hte rectangle
func minimumArea(grid [][]int) int {
	var maxRow int = 0
	var minRow int = math.MaxInt32
	var maxCol int = 0
	var minCol int = math.MaxInt32

	for row := 0; row < len(grid); row++ {
		for col := 0; col < len(grid[row]); col++ {
			currCell := grid[row][col]
			if currCell == 1 {
				maxRow = max(maxRow, row)
				minRow = min(minRow, row)
				maxCol = max(maxCol, col)
				minCol = min(minCol, col)
			}
		}
	}

	height := maxRow - minRow + 1
	width := maxCol - minCol + 1

	var area int = height * width
	return area
}

// mininum deletion
func minimumDeleteSum(s1 string, s2 string) int {
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(indexOne, indexTwo int) int {
		cacheKey := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)
		if value, found := memo[cacheKey]; found {
			return value
		}
		if indexOne == len(s1) && indexTwo == len(s2) {
			return 0
		}
		if indexOne == len(s1) {
			remainingSum := 0
			for index := indexTwo; index < len(s2); index++ {
				remainingSum = remainingSum + int(s2[index])
			}
			return remainingSum
		}
		if indexTwo == len(s2) {
			remainingSum := 0
			for index := indexOne; index < len(s1); index++ {
				remainingSum = remainingSum + int(s1[index])
			}
			return remainingSum
		}
		minSum := math.MaxInt32
		if s1[indexOne] == s2[indexTwo] {
			minSum = recurse(indexOne+1, indexTwo+1)
		} else {
			deleteFromS1 := int(s1[indexOne]) + recurse(indexOne+1, indexTwo)
			deleteFromS2 := int(s2[indexTwo]) + recurse(indexOne, indexTwo+1)
			minSum = min(deleteFromS1, deleteFromS2)
		}

		memo[cacheKey] = minSum
		return minSum
	}
	return recurse(0, 0)
}

func findTargetSumWays(nums []int, target int) int {
	memo := make(map[[2]int]int)

	var dfs func(int, int) int // declaring a variable to store the dfs go function

	dfs = func(index, currSum int) int {
		var cacheKey = [2]int{index, currSum}
		// getting cached values
		if cachedVal, found := memo[cacheKey]; found {
			return cachedVal
		}
		// base case
		if index >= len(nums) {
			if target == currSum {
				return 1
			}
			return 0
		}

		totalWays := dfs(index+1, nums[index]+currSum) + dfs(index+1, currSum-nums[index])

		memo[cacheKey] = totalWays
		return totalWays
	}

	return dfs(0, 0)
}

func isRob(nums []int) int {
	memo := make(map[int]int)

	var recurse func(int) int

	recurse = func(currIndex int) int {
		var cacheKey int = currIndex

		if keyValue, found := memo[cacheKey]; found {
			return keyValue
		}

		if currIndex >= len(nums) {
			return 0
		}

		// skipping a house then adding the next one
		currentHouse := nums[currIndex]
		includeCurrent := currentHouse + recurse(currIndex+2)
		skipCurrent := recurse(currIndex + 1)

		totalCurrentVal := 0
		totalCurrentVal = max(includeCurrent, skipCurrent)

		memo[cacheKey] = totalCurrentVal
		return totalCurrentVal
	}

	return recurse(0)
}

// wild card matching in rust
func isMatch(s string, p string) bool {
	memo := make(map[string]bool)

	var dfs func(int, int) bool

	dfs = func(indexOne, indexTwo int) bool {
		cacheKey := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)

		if value, found := memo[cacheKey]; found {
			return value
		}

		if indexTwo >= len(p) {
			return indexOne >= len(s)
		}

		if indexOne >= len(s) {
			// checking the rest of the string
			for index := indexTwo; index < len(p); index++ {
				if p[index] != '*' {
					return false
				}
			}
			return true
		}

		// if the characters are equal then do nothing
		var path bool = false
		if s[indexOne] == p[indexTwo] {
			path = dfs(indexOne+1, indexTwo+1)
		} else {
			if p[indexTwo] == '?' {
				path = dfs(indexOne+1, indexTwo+1)
			}
			if p[indexTwo] == '*' {
				path = dfs(indexOne+1, indexTwo) || dfs(indexOne, indexTwo+1)
			}
		}

		memo[cacheKey] = path
		return path

	}

	return dfs(0, 0)
}

// go version of dfs problem of restoring Ip address
// no need for caching as it will explore all possible string combinations

func restoreIdAddresses(s string) []string {

	ipCollection := []string{}
	dotLimit := 4

	isValidIp := func(currIp string) bool {
		isIpNum, _ := strconv.Atoi(currIp) // this method returns two terms
		return isIpNum < 256
	}

	var recurse func(int, int, string)

	recurse = func(currIndex, currDots int, currIp string) {

		if currIndex >= len(s) && currDots == dotLimit {
			ipAddress := currIp[:len(currIp)-1] // slicing it from the last char to remove the additional dots
			ipCollection = append(ipCollection, ipAddress)
			return
		}

		for index := currIndex; index < min(len(s), currIndex+3); index++ {
			currIpSlice := s[currIndex : index+1]

			if index != currIndex && s[currIndex] == '0' {
				continue
			}
			if isValidIp(currIpSlice) {
				// string concatenation only works with double quotes not runes
				recurse(index+1, currDots+1, currIp+currIpSlice+".")
			}

		}
	}
	recurse(0, 0, "")
	return ipCollection
}

// concatenating words
func findAllConcatenatedWordsInADict(words []string) []string {
	var mainSet = make(map[string]bool)
	var cache = make(map[string]bool)
	for _, word := range words {
		mainSet[word] = true
	}
	finalCollection := []string{}

	var recurse func(int, string, map[string]bool, int) bool

	recurse = func(currIndex int, currWord string, currSet map[string]bool, currWordCount int) bool {
		cacheKey := strconv.Itoa(currIndex) + "-" + currWord
		if val, found := cache[cacheKey]; found {
			return val
		}
		if currIndex > len(currWord) {
			cache[cacheKey] = false
			return false
		}
		// main base case
		if currIndex >= len(currWord) && currWordCount >= 2 {
			cache[cacheKey] = true
			return true
		}
		var validPath bool = false
		for index := currIndex + 1; index <= len(currWord); index++ {
			currSlice := currWord[currIndex:index]
			if mainSet[currSlice] {
				var foundPath bool = recurse(index, currWord, mainSet, currWordCount+1)
				validPath = foundPath || validPath
				if validPath {
					break
				}
			}
		}
		cache[cacheKey] = validPath
		return validPath
	}
	for _, word := range words {
		mainSet[word] = false
		// only add the if the word returns true after recursive structure succeeds
		if recurse(0, word, mainSet, 0) {
			finalCollection = append(finalCollection, word)
		}
		mainSet[word] = true
	}

	return finalCollection
}

// generating parenthesis
func generateParenthesis(n int) []string {
	result := []string{}
	var recurse func(int, int, string)
	// will return all paths
	recurse = func(openCount, closeCount int, substring string) {
		if openCount > n || closeCount > n {
			return
		}
		if openCount == n && closeCount == n {
			result = append(result, substring)
			return
		}
		if openCount < n {
			recurse(openCount+1, closeCount, substring+"(")
		}
		if closeCount < openCount {
			recurse(openCount, closeCount+1, substring+")")
		}
	}
	recurse(0, 0, "")
	return result

}

// hard problem
func cherryPick(grid [][]int) int {
	cache := make(map[string]int)
	rowLen := len(grid)
	var recurse func(int, int, int, int) int
	recurse = func(row1, col1, row2, col2 int) int {
		var cacheKey string = strconv.Itoa(row1) + "," + strconv.Itoa(col1) + "," + strconv.Itoa(row2) + "," + strconv.Itoa(col2)
		if value, found := cache[cacheKey]; found {
			return value
		}
		if row1 >= rowLen || row2 >= rowLen || col1 >= rowLen || col2 >= rowLen || grid[row1][col1] == -1 || grid[row2][col2] == -1 {
			return -1
		}
		if row1 == rowLen-1 && row2 == rowLen-1 && col1 == rowLen-1 && col2 == rowLen-1 {
			return grid[row1][col1]
		}
		maxCount := -1
		cherryCount := 0

		if row1 == row2 && col1 == col2 {
			cherryCount = grid[row1][col1]
		} else {
			cherryCount = grid[row1][col1] + grid[row2][col2]
		}
		maxCount = max(maxCount, recurse(row1+1, col1, row2+1, col2))
		maxCount = max(maxCount, recurse(row1+1, col1, row2, col2+1))
		maxCount = max(maxCount, recurse(row1, col1+1, row2+1, col2))
		maxCount = max(maxCount, recurse(row1, col1+1, row2, col2+1))

		if maxCount == -1 {
			cache[cacheKey] = -1
			return -1
		}

		totalCount := maxCount + cherryCount
		cache[cacheKey] = totalCount
		return totalCount
	}
	result := recurse(0, 0, 0, 0)
	if result == -1 {
		return 0
	} else {
		return result
	}
}

// getting the obstacle grid using a recursive approach using dfs memoization to record paths that have already been visited
func uniquePaths(obstacleGrid [][]int) int {
	cachePath := make(map[string]int)
	obstacle := 1
	rowLen := len(obstacleGrid)
	colLen := len(obstacleGrid[0])

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + strconv.Itoa(col)
		if value, found := cachePath[cacheKey]; found {
			return value
		}
		// base path
		if row >= rowLen || col >= colLen || row < 0 || col < 0 || obstacleGrid[row][col] == obstacle {
			return 0
		}

		if row == rowLen-1 && col == colLen-1 {
			return 1
		}

		totalPath := 0

		currPathTree := recurse(row+1, col) + recurse(row, col+1)
		totalPath = currPathTree + totalPath

		cachePath[cacheKey] = totalPath
		return totalPath

	}

	return recurse(0, 0)
}

// function to get the min Path cost for a matrix
func minPathCost(grid [][]int, moveCost [][]int) int {
	cache := make(map[string]int)
	rowLen := len(grid)
	colLen := len(grid[0])
	minCost := math.MaxInt32

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if value, found := cache[cacheKey]; found {
			return value
		}

		if row < 0 || col < 0 || row >= rowLen || col >= colLen {
			return 0
		}
		// if its the last element then return the actual grid value
		if row == rowLen-1 {
			return grid[row][col]
		}
		minCost := math.MaxFloat32

		for nextCol := 0; nextCol < colLen; nextCol++ {
			currGridVal := grid[row][col]
			currMoveCost := moveCost[grid[row][col]][nextCol] // gets the exact index and col value of the movecost
			minCost = min(minCost, float64(currGridVal+currMoveCost+recurse(row+1, nextCol)))
		}
		cache[cacheKey] = int(minCost)
		return int(minCost)
	}

	for col := 0; col < colLen; col++ {
		minCost = min(recurse(0, col), minCost)
	}
	return minCost
}

// getting the min failing sum for path using recursiona and dfs approach TLE
func minFaillingSumHard(grid [][]int) int {
	cache := make(map[string]int)
	minCost := math.MaxFloat32
	rowLen := len(grid)
	colLen := len(grid[0])
	maxVal := 9999999

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if value, found := cache[cacheKey]; found {
			return value
		}
		// base border check
		if row < 0 || col < 0 || row >= rowLen || col >= colLen {
			return maxVal
		}
		if row == rowLen-1 {
			return grid[row][col]
		}
		minCost := maxVal

		// traversing all the cols and checking whether the next one or not
		for nextCol := 0; nextCol < colLen; nextCol++ {
			if nextCol != col {
				minCost = min(minCost, recurse(row+1, nextCol))
			}
		}
		result := grid[row][col] + minCost
		cache[cacheKey] = result
		return result
	}

	for col := 0; col < colLen; col++ {
		minCost = min(minCost, float64(recurse(0, col)))
	}

	return int(minCost)
}

// getting the longest increasing path
func longestIncreasingPath(matrix [][]int) int {
	cache := make(map[string]int)
	longestPathCount := 0
	rowLen := len(matrix)
	colLen := len(matrix[0])

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if cachedValue, found := cache[cacheKey]; found {
			return cachedValue
		}
		// base case - exceed boundary and return 0 since there is no path
		if row < 0 || col < 0 || row >= rowLen || col >= colLen {
			return 0
		}
		directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
		localMaxPathResult := 0
		for _, direction := range directions {
			// need to extrapolate next row or next col
			newRow := row + direction[0]
			newCol := col + direction[1]
			// border check then only recurse if the condition passes for valid path
			if newRow >= 0 && newCol >= 0 && newRow < rowLen && newCol < colLen {
				if matrix[newRow][newCol] > matrix[row][col] {
					// updates the particular direction and each recursive cycle
					localMaxPathResult = max(localMaxPathResult, recurse(newRow, newCol))
				}
			}
		}
		// adds one if the localResult returns a path..
		localResult := 1 + localMaxPathResult
		cache[cacheKey] = localResult
		return localResult
	}

	// traversing the matrix and updating the current max path
	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			currMaxPath := recurse(row, col)
			longestPathCount = max(longestPathCount, currMaxPath)
		}
	}

	return longestPathCount
}

// using dfs memo to calculate the count paths from a grid
func countPaths(grid [][]int) int {
	cache := make(map[string]int)
	paths := 0
	rowLen := len(grid)
	colLen := len(grid[0])
	const MOD = 1000000007 // to limit the mod path and control overflow
	// Optimized caching approach
	visited := make([][]int, len(grid))
	for index := 0; index < len(visited); index++ {
		visited[index] = make([]int, len(visited[index]))
		for subIndex := 0; subIndex < len(visited[index]); subIndex++ {
			visited[index][subIndex] = -1
		}
	}
	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if cachedValue, found := cache[cacheKey]; found {
			return cachedValue
		}
		if row < 0 || row >= rowLen || col < 0 || col >= colLen {
			return 0
		}
		var directions = [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}} // directions for the grid path to traverse in
		localPath := 1
		for _, direction := range directions {
			newRow := row + direction[0]
			newCol := col + direction[1]

			if newRow >= 0 && newRow < rowLen && newCol >= 0 && newCol < colLen && grid[newRow][newCol] > grid[row][col] {
				localPath = (localPath + recurse(newRow, newCol)) % MOD
			}
		}
		cache[cacheKey] = localPath
		return localPath
	}
	// passing each cell location in order to get the total number of paths
	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			paths = (paths + recurse(row, col)) % MOD
		}
	}
	return paths
}

// getting maxmimum gold with dfs memoization ... visiting a cell with 0 resets the path to return 0
func getMaximumGold(grid [][]int) int {
	maxAmount := 0
	rowLen := len(grid)
	colLen := len(grid[0])

	var recurse func(int, int, map[string]bool) int
	recurse = func(row, col int, visited_map map[string]bool) int {

		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if visited_map[cacheKey] {
			return 0
		}

		visited_map[cacheKey] = true // visited path

		if row < 0 || row >= rowLen || col < 0 || col >= colLen || grid[row][col] == 0 {
			return 0
		}

		directions := [][]int{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}
		localMax := 0

		for _, direction := range directions {
			newRow := row + direction[0]
			newCol := col + direction[1]

			if newRow >= 0 && newCol >= 0 && newRow < rowLen && newCol < colLen &&
				grid[newRow][newCol] != 0 {
				localMax = max(localMax, recurse(newRow, newCol, visited_map))
			}
		}
		visited_map[cacheKey] = false // unmark for backtracking
		totalPath := grid[row][col] + localMax
		return totalPath
	}

	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			if grid[row][col] != 0 {
				maxAmount = max(maxAmount, recurse(row, col, make(map[string]bool)))
			}
		}
	}

	return maxAmount
}

// difference is changing the grid in place
func optimizedPathGold(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	maxAmount := 0
	rows := len(grid)
	cols := len(grid[0])

	var backtrack func(int, int) int
	backtrack = func(row, col int) int {
		// Check bounds and validity
		if row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == 0 {
			return 0
		}

		// Store original value and mark as visited
		original := grid[row][col]
		grid[row][col] = 0

		// Explore all 4 directions
		directions := [][]int{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}
		maxPath := 0

		for _, dir := range directions {
			newRow := row + dir[0]
			newCol := col + dir[1]
			maxPath = max(maxPath, backtrack(newRow, newCol))
		}

		// Backtrack: restore original value
		grid[row][col] = original

		return original + maxPath
	}

	// Try starting from each cell with gold
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			if grid[row][col] != 0 {
				maxAmount = max(maxAmount, backtrack(row, col))
			}
		}
	}

	return maxAmount
}

// HARD Problem: getting number of steps to get to 0 using recursion
func numWays(steps int, arrLen int) int {
	numWays := 0
	const MOD = 1000000007
	// limit array length before creating memo
	if arrLen > steps+1 {
		arrLen = steps + 1
	}

	maxPos := min(steps, arrLen-1)
	memo := make([][]int, arrLen)
	for i := range memo {
		memo[i] = make([]int, steps+1)
		for j := range memo[i] {
			memo[i][j] = -1
		}
	}
	var recurse func(int, int) int
	recurse = func(currentPos, remainingSteps int) int {
		// returning a cached path
		if memo[currentPos][remainingSteps] != -1 {
			return memo[currentPos][remainingSteps]
		}
		// main base case to return a valid path
		if remainingSteps == 0 {
			// main case to return a valid path
			if currentPos == 0 {
				return 1
			}
			return 0
		}
		// if steps are exchausted then there is no turning back to 0
		if currentPos > remainingSteps || remainingSteps < 0 || currentPos > maxPos {
			memo[currentPos][remainingSteps] = 0
			return 0
		}
		totalPaths := 0
		// moving right within boundary
		movingRight := 0
		if currentPos+1 < arrLen {
			movingRight = (recurse(currentPos+1, remainingSteps-1)) % MOD
		}
		// moving left within boundary
		movingLeft := 0
		if currentPos-1 >= 0 {
			movingLeft = (recurse(currentPos-1, remainingSteps-1)) % MOD
		}
		samePos := 0
		samePos = (recurse(currentPos, remainingSteps-1)) % MOD

		totalPaths = (movingLeft + movingRight + samePos) % MOD
		memo[currentPos][remainingSteps] = totalPaths
		return totalPaths
	}
	numWays = recurse(0, steps) // initial recursive call to begin the recursion
	return numWays
}

// simialar question to reaching destination but above is hard problem
func numberOfWays(startPos int, endPos int, k int) int {
	cache := make(map[string]int)
	const MOD = 1000000007

	var recurse func(int, int) int
	recurse = func(currentPos, remainingSteps int) int {
		cacheKey := strconv.Itoa(currentPos) + "-" + strconv.Itoa(remainingSteps)
		if val, found := cache[cacheKey]; found {
			return val
		}
		if remainingSteps < 0 {
			return 0
		}

		if remainingSteps == 0 {
			if currentPos == endPos {
				return 1
			}
			return 0
		}
		// no more steps remaining
		if math.Abs(float64(currentPos)-float64(endPos)) > float64(remainingSteps) {
			return 0
		}

		totalWays := 0
		left := recurse(currentPos-1, remainingSteps-1)
		right := recurse(currentPos+1, remainingSteps-1)

		totalWays = (left + right) % MOD

		cache[cacheKey] = totalWays
		return totalWays
	}

	return recurse(startPos, k)
}

// minimum cost to cut stick... cutting from both sides using dfs memo interval technique
func minCostToCutStick(n int, cuts []int) int {
	cuts = append(cuts, 0, n)
	sort.Ints(cuts) // enables us to recurse between intervals
	minCost := math.MaxInt32

	var recurse func(int, int, map[string]int) int
	recurse = func(left, right int, cache map[string]int) int {
		// cached minimum path
		cacheKey := strconv.Itoa(left) + "-" + strconv.Itoa(right)
		if cacheMin, found := cache[cacheKey]; found {
			return cacheMin
		}
		// base case no possible cuts to check if there is any cut positions between the left and rigth boundary
		if right-left <= 1 {
			return 0
		}
		minCostLocal := math.MaxInt32

		// for loop to check through various intervals
		for currIndex := left + 1; currIndex < right; currIndex++ {
			currentCost := cuts[right] - cuts[left]
			minCost := currentCost + recurse(left, currIndex, cache) + recurse(currIndex, right, cache)
			minCostLocal = min(minCost, minCostLocal)
		}

		cache[cacheKey] = minCostLocal
		return minCostLocal
	}

	minCost = recurse(0, len(cuts)-1, make(map[string]int))
	// have to pass len cuts - 1 because its calculating till n value
	return minCost
}

// solving jump game V with dfs memo and recursion while keeping track of proper caching
func jumpGameV(arr []int, d int) int {
	maxCost := 0
	memo := make(map[int]int)
	// dfs function to check whether the inner elements are bigger than the current one or not
	var canJump func(int, int) bool
	canJump = func(from, to int) bool {
		// if the from side is smaller or equal then no more checks need to be performed
		if arr[from] <= arr[to] {
			return false
		}
		// checking elements within the from and to range
		start := min(from, to) + 1
		end := max(from, to)
		for i := start; i < end; i++ {
			currEl := arr[i]
			if currEl >= arr[from] {
				return false
			}
		}
		return true
	}
	// dfs recursive function
	var recurse func(int, int) int
	recurse = func(start, d int) int {
		// cached max cost
		cacheKey := start
		if cachedMaxCost, found := memo[cacheKey]; found {
			return cachedMaxCost
		}
		maxVisits := 1
		// checking left step jump count
		for step := 1; step <= d; step++ {
			newJumpIndex := start - step // goes left
			if newJumpIndex < 0 {        // bound checking
				break
			}
			if canJump(start, newJumpIndex) {
				visits := 1 + recurse(newJumpIndex, d)
				maxVisits = max(maxVisits, visits)
			}
		}
		// checking right step jump count
		for step := 1; step <= d; step++ {
			newJumpIndex := start + step  // for right
			if newJumpIndex >= len(arr) { // bound checking
				break
			}
			if canJump(start, newJumpIndex) {
				visits := 1 + recurse(newJumpIndex, d)
				maxVisits = max(maxVisits, visits)
			}
		}
		memo[cacheKey] = maxVisits
		return maxVisits
	}
	// starting from each position in the array and collect recursive paths
	for startIndex := 0; startIndex < len(arr); startIndex++ {
		result := recurse(startIndex, d)
		maxCost = max(maxCost, result)
	}
	return maxCost
}

// jump game ii reaching index with value 0
// record the visited index using a set

func jumpGameIII(arr []int, start int) bool {
	memo := make(map[int]bool)
	visitedIndex := make(map[int]bool)

	var recurse func(int) bool
	recurse = func(index int) bool {
		cacheKey := index
		if cachedPath, found := memo[cacheKey]; found {
			return cachedPath
		}
		visitedIndex[index] = true
		// main base case for true
		if arr[index] == 0 {
			return true
		}
		if index < 0 || index >= len(arr) {
			memo[cacheKey] = false
			return false
		}
		validPath := false
		currJumpValue := arr[index]
		// logic for left side
		var leftJump bool
		if index-currJumpValue >= 0 && !visitedIndex[index-currJumpValue] {
			leftJump = recurse(index - currJumpValue)
		}
		// logic for right side
		var rightJump bool
		if index+currJumpValue < len(arr) && !visitedIndex[index+currJumpValue] {
			rightJump = recurse(index + currJumpValue)
		}

		validPath = leftJump || rightJump
		memo[cacheKey] = validPath
		return validPath
	}

	state := recurse(start)
	return state
}

// frog can cross using dfs memo approach
func frogCanJump(stones []int) bool {
	// make boolean array
	maxPos := stones[len(stones)-1]
	memo := make(map[string]bool)

	if stones[1] != 1 {
		return false
	}

	hasStones := make(map[int]bool)
	for _, v := range stones {
		hasStones[v] = true
	}
	var recurse func(int, int) bool
	recurse = func(currIndex, prevJump int) bool {
		key := strconv.Itoa(currIndex) + "-" + strconv.Itoa(prevJump)
		if val, found := memo[key]; found {
			return val
		}
		// main base case
		if currIndex < 0 || currIndex > maxPos {
			memo[key] = false
			return false
		}
		if currIndex == stones[len(stones)-1] {
			return true
		}
		validJump := false
		// main check to see whether jump is valid or not
		directions := []int{prevJump - 1, prevJump, prevJump + 1}

		for _, direction := range directions {
			if direction > 0 {
				newDirection := direction + currIndex
				if hasStones[newDirection] { // checking stones
					jumpState := recurse(newDirection, direction) // direction is the previous direction
					validJump = jumpState || validJump
				}
			}
		}

		memo[key] = validJump
		return validJump
	}
	return recurse(1, 1) // passing current index and prevJump coords
}

// geting minimum side ways jump only jump if the obstacles position is not equal to lane
func minSideWaysJump(obstacles []int) int {
	cache := make(map[string]int)
	minJumps := math.MaxInt32

	var recurse func(int, int) int
	recurse = func(currPos, currLane int) int {
		// cached min jump path
		cacheKey := strconv.Itoa(currPos) + "-" + strconv.Itoa(currLane)
		if val, found := cache[cacheKey]; found {
			return val
		}
		// main base case if it exceeds length
		if currPos == len(obstacles)-1 {
			return 0
		}
		if currPos >= len(obstacles) {
			return 9999999
		}
		jumps := math.MaxInt32
		// keep the same path if there are no obstacles in the current lane
		if obstacles[currPos+1] != currLane && currPos+1 < len(obstacles) {
			jumps = min(recurse(currPos+1, currLane), jumps)
		}
		// checking for sideways jumps
		for lane := 1; lane <= 3; lane++ {
			// side jump plus moving forward in orer to avoid infinite recursion
			if lane != currLane && obstacles[currPos] != lane && obstacles[currPos+1] != lane {
				jumps = int(math.Min(float64(jumps), 1+float64(recurse(currPos+1, lane))))
			}
		}
		cache[cacheKey] = jumps
		return jumps
	}

	minJumps = recurse(0, 2)
	return minJumps
}

// using skip and reuse technique for dfs memoization
func combinationSum4(nums []int, target int) int {
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(index, target int) int {
		key := strconv.Itoa(index) + "-" + strconv.Itoa(target)
		if val, found := memo[key]; found {
			return val
		}
		if target == 0 {
			return 1
		}
		if index >= len(nums) || target < 0 {
			return 0
		}
		// means u can pick again from any combination
		ways := recurse(0, target-nums[index]) // resets index to make it work for any number of other numbers

		skip := recurse(index+1, target)

		totalWays := ways + skip
		memo[key] = totalWays
		return totalWays
	}

	return recurse(0, target)
}

// getting the cheapest price for flights remember.. k flight means k stops means k + 1 stops
func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
	memo := make(map[string]int)
	minCost := math.MaxInt32
	// organizing array into map structure
	// array restructing in order to collect the items based on the source being the key and the to and cost being the key pair values
	flightMap := make(map[int][][2]int)
	for _, flight := range flights {
		from, to, cost := flight[0], flight[1], flight[2]
		flightMap[from] = append(flightMap[from], [2]int{to, cost})
	}
	// the recursion will be traversing through a map hence you will not need to use an index for traversal
	var recurse func(int, int) int
	recurse = func(city, remainingStops int) int {
		// cached minCost for flight traversal
		key := strconv.Itoa(city) + "-" + strconv.Itoa(remainingStops)
		if cachedJumpValue, found := memo[key]; found {
			return cachedJumpValue
		}
		// base case
		if city == dst {
			return 0
		}
		// if there are no remaining stops then return the maximum int size value possible
		if remainingStops < 0 {
			return math.MaxInt32
		}
		localMinCost := math.MaxInt32
		// current local cost from each source city
		for _, flightDetails := range flightMap[city] {
			destination, cost := flightDetails[0], flightDetails[1]
			if remainingStops > 0 { // need to check before running the recursion
				recursiveResult := recurse(destination, remainingStops-1)
				if recursiveResult != math.MaxInt32 {
					totalCost := recursiveResult + cost
					localMinCost = min(localMinCost, totalCost)
				}
			}
		}
		memo[key] = localMinCost
		return localMinCost
	}

	// getting the recursive avlue
	minCost = recurse(src, k+1)
	if minCost == math.MaxInt32 {
		return -1
	}
	return minCost
}

// minimum number of distance to make word1 to word2
func minDistance(word1 string, word2 string) int {
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(indexOne, indexTwo int) int {
		// getting the cached value
		key := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)
		if val, found := memo[key]; found {
			return val
		}
		if indexOne == len(word1) && indexTwo == len(word2) {
			return 0
		}
		if indexOne == len(word1) {
			return len(word2) - indexTwo
		}
		if indexTwo == len(word2) {
			return len(word1) - indexOne
		}

		// main recursive logic
		minDeletions := math.MaxInt32

		if word1[indexOne] == word2[indexTwo] {
			minDeletions = recurse(indexOne+1, indexTwo+1)
		} else {
			minDeletions = 1 + min(recurse(indexOne+1, indexTwo), recurse(indexOne, indexTwo+1), minDeletions)
		}

		memo[key] = minDeletions
		return minDeletions
	}

	return recurse(0, 0)
}

// getting the knight probability and figuring out the probability floating point value that the knight stays on board
func knightProbability(n int, k int, row int, column int) float64 {
	moveCache := make(map[string]float64)

	var recurse func(int, int, int) float64
	recurse = func(rowLocal, colLocal, remainingMoves int) float64 {
		// cached probability for the knights moves
		cacheKey := strconv.Itoa(rowLocal) + "-" + strconv.Itoa(colLocal) + "-" + strconv.Itoa(remainingMoves)
		if cachedVal, found := moveCache[cacheKey]; found {
			return float64(cachedVal)
		}
		// out of bounds base case
		if rowLocal < 0 || colLocal < 0 || rowLocal >= n || colLocal >= n {
			return 0.0
		}
		// main base cases
		if remainingMoves == 0 {
			return 1.0
		}

		probability := 0.0
		// 8 directions that the knight can move into
		directions := [][]int{
			{2, 1}, {2, -1}, {-2, 1}, {-2, -1},
			{1, 2}, {1, -2}, {-1, 2}, {-1, -2},
		}
		// exploring all possible 8 directions for the knight
		for _, direction := range directions {
			newRow := direction[0] + rowLocal
			newCol := direction[1] + colLocal
			if remainingMoves > 0 {
				probability = probability + recurse(newRow, newCol, remainingMoves-1)
			}
		}
		// updating the probability and returning the final probability from each recursive chain
		finalProbabilityPerMove := probability / 8
		moveCache[cacheKey] = finalProbabilityPerMove
		return finalProbabilityPerMove
	}

	return recurse(row, column, k)
}

// word search problem using dfs to solve the path and to see whether the letters make up the word or not
func exist(board [][]byte, word string) bool {
	rowLen := len(board)
	colLen := len(board[0])
	exist := false
	filler := '#'

	var recurse func(int, int, int, [][]byte) bool
	recurse = func(row, col, index int, boardRef [][]byte) bool {
		if index == len(word) {
			return true
		}
		// base case
		if row < 0 || col < 0 || row >= rowLen || col >= colLen || boardRef[row][col] == byte(filler) {
			return false
		}

		if boardRef[row][col] != word[index] {
			return false
		}
		// visited
		tempCell := boardRef[row][col]
		boardRef[row][col] = byte(filler)
		validPath := false

		directions := [][]int{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}

		for _, direction := range directions {
			newRow := direction[0] + row
			newCol := direction[1] + col
			validPath = recurse(newRow, newCol, index+1, boardRef)
			if validPath {
				return validPath
			}
		}

		boardRef[row][col] = tempCell // for backtracking
		return validPath
	}

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[i]); j++ {
			if board[i][j] == word[0] {
				exist = recurse(i, j, 0, board)
				if exist {
					return exist
				}
			}

		}
	}
	return exist
}

// deleting strings and letters from its prefix so a prefix copy can also be deleted in addition
func deleteString(s string) int {
	minDeletions := 0
	memo := make(map[int]int)

	var recurse func(int) int
	recurse = func(start int) int {
		// cached max key value
		key := start
		if keyVal, found := memo[key]; found {
			return keyVal
		}
		if start == len(s) {
			return 0
		}

		minLocalDel := 1
		// for loop keeps in check and limit within the half string length
		for index := 1; index <= (len(s)-start)/2; index++ {
			// checking the equal length sub in the next section while staying in len bounds
			if s[start:start+index] == s[start+index:start+2*index] {
				minLocalDel = max(minLocalDel, 1+recurse(start+index))
			}
		}

		memo[key] = minLocalDel
		return minLocalDel
	}

	minDeletions = recurse(0)

	return minDeletions
}

// checking to see whether all the flower garden ranges can be watered or not but using the minimum number of taps
func minTaps(n int, ranges []int) int {
	memo := make(map[int]int)
	intervals := make([]int, 0)
	minTaps := math.MaxInt32

	// will distribute intervals where the left most index will contain the limit of the right most
	for index, value := range ranges {
		leftLimit := max(0, index-value)
		rightLimit := min(n, index+value)
		if rightLimit > leftLimit {
			intervals[leftLimit] = rightLimit // adding the left index limit to the right limit
		}

	}
	// primary recursive function to check for the widest ranges
	var recurse func(int) int
	recurse = func(position int) int {
		cacheKey := position
		if val, found := memo[cacheKey]; found {
			return val
		}
		// main base case that will return 0 if it reaches the target
		if position >= n {
			return 0
		}
		minLocalTaps := math.MaxInt32
		for index := 0; index <= position; index++ {
			currRightRange := intervals[index]
			if currRightRange > position {
				localResult := recurse(currRightRange)
				if localResult != math.MaxInt32 {
					minLocalTaps = min(minLocalTaps, localResult+1)
				}
			}
		}
		memo[cacheKey] = minLocalTaps
		return minLocalTaps
	}
	minTaps = recurse(0)
	if minTaps == math.MaxInt32 {
		return -1
	}

	return minTaps
}

// problem of video stitching using dfs memo
// we primarily care about the end values only
func videoStitching(clips [][]int, time int) int {
	memo := make(map[int]int)
	maxReach := make([]int, time+1) // by default the intiail values are 0
	minClips := math.MaxInt32

	// populate max reach array and its a single dimension linear array
	for index := 0; index < len(clips); index++ {
		clipStart := clips[index][0]
		clipEnd := clips[index][1]
		if clipStart <= time {
			maxReach[clipStart] = max(maxReach[clipStart], clipEnd)
		}

	}

	var recurse func(int) int
	recurse = func(currPos int) int {
		// min clip count
		key := currPos
		if val, found := memo[key]; found {
			return val
		}
		if currPos >= time {
			return 0
		}
		minClipCount := math.MaxInt32
		furthest := min(maxReach[currPos], time)
		// recursing through max reach and checking for the right most limit in order to collect the final count
		for index := currPos + 1; index <= furthest; index++ {
			localRecurseResult := recurse(index)
			if localRecurseResult != math.MaxInt32 {
				minClipCount = min(minClipCount, 1+localRecurseResult)
			}

		}
		memo[key] = minClipCount
		return minClipCount
	}

	minClips = recurse(0)
	if minClips == math.MaxInt32 {
		return -1
	}
	return minClips
}

// longest string chain using dfs recursive method and using the cache approach to store the current max length of the chain
func longestStrChain(words []string) int {
	maxStringChain := 0
	wordSet := make(map[string]bool)
	for _, word := range words {
		wordSet[word] = true
	}
	memo := make(map[string]int)

	var recurse func(string) int
	recurse = func(currWord string) int {
		// cached chain value
		cacheKey := currWord
		if val, found := memo[cacheKey]; found {
			return val
		}
		localMaxLength := 1 // will contain the local max length of the current series of chain

		for index := 0; index < len(currWord); index++ {
			currSlice := currWord[:index] + currWord[index+1:]
			if wordSet[currSlice] == true {
				currMaxChain := recurse(currSlice)
				// updating the localMaxLength with +1 to get the current chain value
				localMaxLength = max(localMaxLength, 1+currMaxChain)
			}
		}

		memo[cacheKey] = localMaxLength
		return localMaxLength
	}

	for _, word := range words {
		maxStringChain = max(maxStringChain, recurse(word))
	}

	return maxStringChain
}

// simple one step solution
func maxBottlesDrunk(numBottles int, numExchange int) int {
	cache := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(empty, rate int) int {

		cacheKey := fmt.Sprintf("%d-%d", empty, rate)
		if val, found := cache[cacheKey]; found {
			return val
		}

		if empty < rate {
			return 0
		}

		return 1 + recurse(empty-rate+1, rate+1)

	}

	return numBottles + recurse(numBottles, numExchange)
}

// recursion hard -> skip and count recursion logic cannot exceed minProfit account and n number of groups of people
func profitableSchemes(n int, minProfit int, group []int, profit []int) int {
	memo := make(map[string]int)
	const MOD = 1000000007
	// main recursive to check membersUsed , currprofit and things like that
	var recurse func(index, membersUsed, currProfit int) int
	recurse = func(index, membersUsed, currProfit int) int {
		// cache key -> returning cached max
		cacheKey := strconv.Itoa(index) + "-" + strconv.Itoa(membersUsed) + "-" + strconv.Itoa(currProfit)
		if cachedSchemes, found := memo[cacheKey]; found {
			return cachedSchemes
		}
		// main base case and to check currProfit is bigger than minProfit then return 1
		if index >= len(group) {
			if currProfit >= minProfit {
				return 1
			} else {
				return 0
			}
		}
		// included ways
		includedSchemeCount := 0
		// skipped path for min scheme count -> recursive index iteration despite whichever index ur in
		skippedSchemeCount := recurse(index+1, membersUsed, currProfit)
		// use the current path
		if membersUsed+group[index] <= n { // to check membersUsed leakage
			currMinProfit := min(minProfit, currProfit+profit[index]) // optimization approach
			includedSchemeCount = recurse(index+1, membersUsed+group[index], currMinProfit)
		}
		totalSchemes := (includedSchemeCount + skippedSchemeCount) % MOD
		memo[cacheKey] = totalSchemes
		return totalSchemes
	}

	return recurse(0, 0, 0)
}

// HARD level string dfs memo recursive based question
func minStickers(stickers []string, target string) int {
	stickerCounts := make([][26]int, len(stickers))
	stickerCache := make(map[string]int)
	countWays := math.MaxInt32
	const MOD = 1000000007
	// counting letters and returning a sub 26 letter freq matrix
	countLetters := func(word string) [26]int {
		var counts [26]int
		for _, character := range word {
			counts[character-'a']++
		}
		return counts
	}
	// constructing stick count matrix to store frequencies of chars in stickers
	for i, sticker := range stickers {
		stickerCounts[i] = countLetters(sticker)
	}
	// subtracting letters from the count matrix
	substractLetters := func(stickerCountMatrix [26]int, currTarget string) string {
		var count [26]int
		for _, ch := range currTarget {
			count[ch-'a']++
		}
		// subtracting the letter
		for i := 0; i < 26; i++ {
			count[i] -= stickerCountMatrix[i]
			if count[i] < 0 { // offsetting to 0
				count[i] = 0
			}
		}
		var newTarget string
		// populating the new target
		for i := 0; i < 26; i++ { // current letter index
			for j := 0; j < count[i]; j++ { // checking how many times the current letter is present
				newTarget += string('a' + i)
			}
		}
		return newTarget
	}

	// main recursive function to check very sub stickers
	var recurse func(string) int
	recurse = func(currTarget string) int {
		stickerKey := currTarget
		if val, found := stickerCache[stickerKey]; found {
			return val
		}

		// main base case
		if currTarget == "" {
			return 0
		}
		// first letter of currtarget
		currFirstIndex := currTarget[0] - 'a'
		minLocalWays := math.MaxInt32

		for _, stickerCountMatrix := range stickerCounts {
			if stickerCountMatrix[currFirstIndex] == 0 {
				continue
			}
			newTarget := substractLetters(stickerCountMatrix, currTarget)
			minRes := 1 + recurse(newTarget)
			minLocalWays = min(minLocalWays, minRes)
		}

		stickerCache[stickerKey] = minLocalWays
		return minLocalWays
	}
	countWays = recurse(target)
	if countWays == math.MaxInt32 {
		return -1
	}
	return countWays
}

func minPathSum(grid [][]int) int {
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		if row < 0 || col < 0 || row >= len(grid) || col >= len(grid[0]) {
			return math.MaxInt32
		}
		cacheKey := strconv.Itoa(row) + "-" + strconv.Itoa(col)
		if val, found := memo[cacheKey]; found {
			return val
		}
		// bottom right is the final destination
		if row == len(grid)-1 && col == len(grid[0])-1 {
			return grid[row][col]
		}
		minLocalWays := min(recurse(row+1, col), recurse(row, col+1)) + grid[row][col]
		memo[cacheKey] = minLocalWays
		return minLocalWays

	}

	return recurse(0, 0)
}

// main recursive approach would be going down then down right then memoization the minimum path
func minimumTotal(triangle [][]int) int {
	minPathSum := math.MaxInt32
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "-" + strconv.Itoa(col)
		if val, found := memo[cacheKey]; found {
			return val
		}
		// base case
		if row < 0 || col < 0 || row >= len(triangle) || col >= len(triangle[row]) {
			return 0
		}
		// main base case to return the final element
		if row == len(triangle)-1 && col == len(triangle[row])-1 {
			return triangle[row][col]
		}
		localMin := math.MaxInt32
		// we can only more down or down - right
		localMin = min(recurse(row+1, col+1), recurse(row+1, col)) + triangle[row][col]
		memo[cacheKey] = localMin
		return localMin
	}

	minPathSum = recurse(0, 0)
	if minPathSum == math.MaxInt32 {
		return -1
	}
	return minPathSum
}

// dfs recursive approach -> bidirectional graph approach
func findTheCity(n int, edges [][]int, distanceThreshold int) int {
	memo := make(map[string]int)
	citiesCovered := math.MaxInt32
	bestCityIndex := -1
	// bidirectional graph
	graph := make(map[int][][2]int) // map will have a matrix has from, to and weightage
	for _, edge := range edges {
		from, to, weight := edge[0], edge[1], edge[2]
		// adding from and to as key pairs for bidirectional approach
		graph[from] = append(graph[from], [2]int{to, weight})
		graph[to] = append(graph[to], [2]int{from, weight})
	}
	// main recursive function
	var recurse func(int, int, map[int]bool) int
	recurse = func(currentCity, remainingVisits int, visitedSet map[int]bool) int {
		// visited cache key
		visitedKey := strconv.Itoa(currentCity) + "-" + strconv.Itoa(remainingVisits)
		if val, found := memo[visitedKey]; found {
			return val
		}
		// remaining visits below 0 means there is city can be visited
		if remainingVisits < 0 {
			return 0
		}
		visitedSet[currentCity] = true
		// main recursive calculation
		visitedCities := 0
		for _, curr := range graph[currentCity] {
			destination, weight := curr[0], curr[1]
			if !visitedSet[destination] && remainingVisits >= weight {
				if remainingVisits > 0 {
					localVisits := recurse(destination, remainingVisits-weight, visitedSet)
					visitedCities += 1 + localVisits
				}
			}
		}
		visitedSet[currentCity] = false
		memo[visitedKey] = visitedCities
		return visitedCities
	}
	// should be starting from each city
	for cityIndex := 0; cityIndex < n; cityIndex++ {
		result := recurse(cityIndex, distanceThreshold, make(map[int]bool))
		// pruning based on cities covered and city index threshold reached
		if result < citiesCovered || (result == citiesCovered && cityIndex > bestCityIndex) {
			citiesCovered = result
			bestCityIndex = cityIndex
		}
	}
	return bestCityIndex
}

// straight forward dp problem
func minSwap(nums1 []int, nums2 []int) int {
	memo := make(map[string]int)
	minOperations := math.MaxInt32
	minOperationsSwap := math.MaxInt32

	// main recursive function
	var recurse func(int, bool) int
	recurse = func(index int, isPrevSwapped bool) int {
		// returning the cached min operation count
		key := fmt.Sprintf("%d-%t", index, isPrevSwapped)
		if val, found := memo[key]; found {
			return val
		}
		// main base case
		if index >= len(nums1) {
			return 0
		}
		minCountOperations := math.MaxInt32
		prev1 := nums1[index-1]
		prev2 := nums2[index-1]
		// swapping logic comes after determining the state whether its acceptable or not to swap
		if isPrevSwapped {
			prev1, prev2 = prev2, prev1
		}
		// dont swap
		if nums1[index] > prev1 && nums2[index] > prev2 {
			minCountOperations = min(minCountOperations, recurse(index+1, false))
		}
		// swap
		if nums1[index] > prev2 && nums2[index] > prev1 {
			minCountOperations = min(minCountOperations, 1+recurse(index+1, true))
		}

		memo[key] = minCountOperations
		return minCountOperations
	}

	minOperations = recurse(1, false)
	minOperationsSwap = 1 + recurse(1, true)
	return min(minOperations, minOperationsSwap)
}

// HARD Problem := finding out the optimal length of hte string
func getLengthOfOptimalCompression(s string, k int) int {
	memo := make(map[string]int)
	n := len(s)
	// function to count same value
	countSame := func(count int) int {
		if count == 1 {
			return 1
		}
		if count < 10 {
			return 2
		}
		// so if its double digits
		if count < 100 {
			return 3
		}
		return 4
	}

	var recurse func(int, int) int
	recurse = func(index, kCount int) int {
		// base cases
		cacheKey := strconv.Itoa(index) + "-" + strconv.Itoa(kCount)
		if val, found := memo[cacheKey]; found {
			return val
		}
		if index == n {
			return 0
		}
		// if more deletion values remain than the final characters remaining then delete all
		if n-index <= kCount {
			return 0
		}
		result := math.MaxInt32

		// initial deletion step
		if kCount > 0 {
			result = min(result, recurse(index+1, kCount-1))
		}

		// if not deleted then checking similar count
		sameCount := 0
		deleteCount := 0
		for subIndex := index; subIndex < n; subIndex++ {
			if s[subIndex] == s[index] {
				sameCount++
			} else {
				deleteCount++
			}
			// if too many deleted then get out of the loop
			if deleteCount > kCount {
				break
			}
			// getting the minimum of result recursed value and taking countSame length along with it and seeing how many can be deleted
			result = min(result, countSame(sameCount)+recurse(subIndex+1, kCount-deleteCount))
		}

		memo[cacheKey] = result
		return result
	}

	return recurse(0, k)
}

// using greedy
func compress(chars []byte) int {
	readIndex := 0
	writeIndex := 0

	for readIndex < len(chars) {
		currChar := chars[readIndex]
		sameCount := 0

		// adding the same chars as a count
		for readIndex < len(chars) && chars[readIndex] == currChar {
			sameCount++
			readIndex++
		}

		// updating the first letter of hte changed write index
		chars[writeIndex] = currChar
		writeIndex++

		if sameCount > 1 {
			for _, char := range []byte(fmt.Sprintf("%d", sameCount)) {
				chars[writeIndex] = char
				writeIndex++
			}
		}
	}

	return writeIndex
}

// length of LIS is a tricky one
func lengthOfLIS(nums []int) int {
	n := len(nums)

	// dp[i][j] represents LIS starting at index i, with previous index j-1
	// (we shift prevIndex by +1 so that -1 → 0)
	dp := make([][]int, n)
	for i := range dp {
		dp[i] = make([]int, n+1)
		for j := range dp[i] {
			dp[i][j] = -1
		}
	}

	var recurse func(int, int) int
	recurse = func(currIndex, prevIndex int) int {
		// base case
		if currIndex == n {
			return 0
		}
		// check memo
		if dp[currIndex][prevIndex+1] != -1 {
			return dp[currIndex][prevIndex+1]
		}
		// option 1: skip current element
		skip := recurse(currIndex+1, prevIndex)

		// option 2: take current element (only if valid)
		include := 0
		if prevIndex == -1 || nums[prevIndex] < nums[currIndex] {
			include = 1 + recurse(currIndex+1, currIndex)
		}

		// store and return
		dp[currIndex][prevIndex+1] = max(include, skip)
		return dp[currIndex][prevIndex+1]
	}

	return recurse(0, -1)
}

// creating maxDotProduct sequence
func maxDotProduct(nums1 []int, nums2 []int) int {
	memo := make(map[string]int)
	maxProd := math.MinInt32

	var recurse func(int, int) int
	recurse = func(indexOne, indexTwo int) int {
		cachkey := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)
		if val, found := memo[cachkey]; found {
			return val
		}
		// if it exceeds either of the 0 then there is no element to multiply with
		if indexOne >= len(nums1) || indexTwo >= len(nums2) {
			return math.MinInt32
		}

		maxDotProduct := 0

		// starting new for multiplying indexOne with indexTwo
		startNew := nums1[indexOne] * nums2[indexTwo]
		// taking both and starting the path
		takeBoth := nums1[indexOne]*nums2[indexTwo] + recurse(indexOne+1, indexTwo+1)

		// skips
		skipFirst := recurse(indexOne+1, indexTwo)
		skipSecond := recurse(indexOne, indexTwo+1)

		maxDotProduct = max(skipFirst, skipSecond, takeBoth, startNew)
		memo[cachkey] = maxDotProduct
		return maxDotProduct
	}

	maxProd = recurse(0, 0)
	if maxProd == math.MinInt32 {
		return -1
	}
	return maxProd
}

// min Distance
func maxUncrossedLines(nums1 []int, nums2 []int) int {
	// caching
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(indexOne, indexTwo int) int {
		// cached max crossed lines
		cacheKey := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)
		if val, found := memo[cacheKey]; found {
			return val
		}
		// base case for crossing boundaries
		if indexOne >= len(nums1) || indexTwo >= len(nums2) {
			return 0
		}
		// include the current straight lines if values are equal
		totalUncrossedLines := 0
		if nums1[indexOne] == nums2[indexTwo] {
			totalUncrossedLines = max(totalUncrossedLines, 1+recurse(indexOne+1, indexTwo+1))
		} else if nums1[indexOne] != nums2[indexTwo] {
			// return max skipped step
			totalUncrossedLines = max(recurse(indexOne+1, indexTwo), recurse(indexOne, indexTwo+1))
		}
		// adding together the total skipped and straight line network combinations
		memo[cacheKey] = totalUncrossedLines
		return totalUncrossedLines
	}

	return recurse(0, 0)
}

// getting the key pressed distance
func minimumDistance(word string) int {
	keyPos := make(map[rune][2]int)
	cache := make(map[string]int)
	// key press look up table
	for i := 0; i < 26; i++ {
		letter := rune('A' + i) // rune gets the unicode value of a certain char
		row := i / 6
		col := i % 6
		keyPos[letter] = [2]int{row, col}
	}

	var recurse func(int, int, int) int
	recurse = func(index, pos1, pos2 int) int {
		// caches minimum value
		key := strconv.Itoa(index) + "-" + strconv.Itoa(pos1) + "-" + strconv.Itoa(pos2)
		if val, found := cache[key]; found {
			return val
		}
		// all explored letters in the word
		if index >= len(word) {
			return 0
		}
		currChar := rune(word[index])
		minKeyPressDistance := math.MaxInt32
		currDistance := math.MaxInt32
		// main dfs logic to calculate the distance where two distances need to be calculated in order to check
		if pos1 == -1 {
			currDistance = 0
		} else {
			currDistance = int(math.Abs((float64(keyPos[currChar][0]) - float64(keyPos[rune(word[pos1])][0]))) +
				(math.Abs((float64(keyPos[currChar][1]) - float64(keyPos[rune(word[pos1])][1])))))
		}
		minKeyPressDistance = min(minKeyPressDistance, currDistance+recurse(index+1, index, pos2))

		if pos2 == -1 {
			currDistance = 0
		} else {
			currDistance = int(math.Abs((float64(keyPos[currChar][0]) - float64(keyPos[rune(word[pos2])][0]))) +
				(math.Abs((float64(keyPos[currChar][1]) - float64(keyPos[rune(word[pos2])][1])))))
		}
		minKeyPressDistance = min(minKeyPressDistance, currDistance+recurse(index+1, pos1, index))
		cache[key] = minKeyPressDistance
		return minKeyPressDistance
	}

	minDistance := recurse(0, -1, -1)
	if minDistance == math.MaxInt32 {
		return -1
	}
	return minDistance
}

// max profit by buying and short selling stocks => main approach would be a balance between holding and selling stocks
func maximumProfit(prices []int, k int) int64 {
	memo := make(map[string]int64) // storing max profit per index count
	const NEG_INF int64 = -9_000_000_000_000_000
	// main dfs recursive function
	var recurse func(int, int, int) int64
	recurse = func(index, transactionsLeft, isHoldingPos int) int64 {
		// cached stock value
		key := strconv.Itoa(index) + "-" + strconv.Itoa(transactionsLeft) + "-" + strconv.Itoa(isHoldingPos)
		if val, found := memo[key]; found {
			return val
		}
		// main base case for max profit once end is reached
		if index >= len(prices) {
			if isHoldingPos != 0 {
				return NEG_INF // invalid: open position left
			}
			return 0 // valid: finished all days with no open trade
		}
		// no transactions can be traded
		if transactionsLeft == 0 {
			if isHoldingPos != 0 {
				return NEG_INF
			}
			return 0
		}

		// local recursive maxProfit value
		var maxProfit int64
		maxProfit = NEG_INF
		// skipped index
		skippedCurrentStock := recurse(index+1, transactionsLeft, isHoldingPos)
		// three cases where to hold or short sell
		switch isHoldingPos {
		// no stock so can buy or short sell
		case 0:
			// bought so only can sell and get the profit of that day
			buyCurrent := -int64(prices[index]) + recurse(index+1, transactionsLeft, 1)
			shortSellCurrent := int64(prices[index]) + recurse(index+1, transactionsLeft, -1)
			maxProfit = max(buyCurrent, shortSellCurrent, maxProfit)
		// sell current and get money
		case 1:
			sellCurrent := int64(prices[index]) + recurse(index+1, transactionsLeft-1, 0)
			maxProfit = max(sellCurrent, maxProfit)
		// buy back the short sell amount meaning u lose money
		case -1:
			buyBackShortSell := -int64(prices[index]) + recurse(index+1, transactionsLeft-1, 0)
			maxProfit = max(maxProfit, buyBackShortSell)
		}

		maxProfit = max(maxProfit, skippedCurrentStock) // final comparison between skipped and and current maxProfit
		memo[key] = maxProfit
		return maxProfit
	}

	return recurse(0, k, 0)
}

// minimum letter insertions to make a string palindrom and getting the minimumcount of insertions
func minInsertions(s string) int {
	memo := make(map[string]int)
	// primary recursive function to calculate the minimum count
	var recurse func(int, int) int
	recurse = func(leftIndex, rightIndex int) int {
		// cached min count value for minimum insertions required
		cacheKey := strconv.Itoa(leftIndex) + "-" + strconv.Itoa(rightIndex)
		if val, found := memo[cacheKey]; found {
			return val
		}
		// main base case to control the index traversal from either sides of the string for palindrome check
		if leftIndex >= rightIndex || rightIndex <= leftIndex {
			return 0
		}
		minInsertionCount := math.MaxInt32
		// main recursive logic
		if s[leftIndex] == s[rightIndex] {
			minInsertionCount = min(minInsertionCount, recurse(leftIndex+1, rightIndex-1))
		} else {
			// insert before
			insertBefore := recurse(leftIndex+1, rightIndex)
			insertAfter := recurse(leftIndex, rightIndex-1)
			// updating the minimum insertion counnt
			minInsertionCount = min(insertAfter, insertBefore) + 1

		}
		memo[cacheKey] = minInsertionCount
		return minInsertionCount
	}

	return recurse(0, len(s)-1)
}

// organizing events and the maximum value that can be attained from them
func maxValue(events [][]int, k int) int {
	sort.Slice(events, func(i, j int) bool {
		return events[i][0] < events[j][0]
	})

	// caching
	dp := make([][]int, len(events)+1)
	for i := range dp {
		dp[i] = make([]int, k+1)
		for j := range dp[i] {
			dp[i][j] = -1
		}
	}

	var recurse func(int, int) int
	recurse = func(currIndex, kRemaining int) int {
		// cached mininmum events
		if dp[currIndex][kRemaining] != -1 {
			return dp[currIndex][kRemaining]
		}
		// base case
		if kRemaining == 0 || currIndex >= len(events) {
			return 0
		}
		maximumSumEvent := 0
		// skip event
		skipEvent := recurse(currIndex+1, kRemaining)
		// include event needs to include based on the next index
		includeEvent := 0
		// adding the next best index
		nextBestIndex := sort.Search(len(events), func(i int) bool { // go inbuilt sort.search does binary search
			return events[i][0] > events[currIndex][1]
		})
		if kRemaining > 0 {
			includeEvent = events[currIndex][2] + recurse(nextBestIndex, kRemaining-1)
		}
		maximumSumEvent = max(includeEvent, skipEvent)
		dp[currIndex][kRemaining] = maximumSumEvent
		return maximumSumEvent
	}

	return recurse(0, k)
}

// job scheduling
func jobScheduling(startTime []int, endTime []int, profit []int) int {
	// caching
	dpCache := make([]int, len(startTime))
	for i := range dpCache {
		dpCache[i] = -1
	}
	// making events and sorting it out
	events := make([][]int, len(startTime))
	for i := range startTime {
		events[i] = []int{startTime[i], endTime[i], profit[i]}
	}
	// sorting the events based on start time
	sort.Slice(events, func(i, j int) bool {
		return events[i][0] < events[j][0]
	})
	// main recursive logic
	var recurse func(int) int
	recurse = func(currIndex int) int {
		// base case
		if currIndex >= len(events) {
			return 0
		}
		// cached result
		if dpCache[currIndex] != -1 {
			return dpCache[currIndex]
		}

		maxProfit := 0
		skipCurrent := recurse(currIndex + 1)
		includeCurrent := 0
		// using binary search
		nextBestIndex := sort.Search(len(events), func(i int) bool {
			return events[i][0] >= events[currIndex][1]
		})
		includeCurrent = events[currIndex][2] + recurse(nextBestIndex)

		maxProfit = max(skipCurrent, includeCurrent)
		dpCache[currIndex] = maxProfit
		return maxProfit
	}

	return recurse(0)
}

// dfs recursion approach
func maxScore(n int, k int, stayScore [][]int, travelScore [][]int) int {
	// using dynamic dp
	cache := make([][]int, k+1)
	for i := range cache {
		cache[i] = make([]int, n)
		for j := range cache[i] {
			cache[i][j] = -1 // -1 means not visited
		}
	}
	maxPoints := 0
	var recurse func(int, int) int
	recurse = func(currDay, currCity int) int {
		// main base case
		if currDay == k {
			return 0
		}
		// maxed cached value
		if cache[currDay][currCity] != -1 {
			return cache[currDay][currCity]
		}

		// stay in current city and get the score
		stayCurrent := stayScore[currDay][currCity] + recurse(currDay+1, currCity)
		// skip destinations
		nextDestination := 0
		maxPoints := stayCurrent

		// traverse across current city ... sub recursion chain
		for city := 0; city < n; city++ {
			if city == currCity { // skipping current city
				continue
			}
			nextDestination = travelScore[currCity][city] + recurse(currDay+1, city) // updating the recursion to look for the next day
			maxPoints = max(nextDestination, stayCurrent, maxPoints)
		}
		cache[currDay][currCity] = maxPoints // updated cached max points
		return maxPoints
	}
	// starting from all potential cities
	for startingCity := range n {
		// day is stable
		maxPoints = max(maxPoints, recurse(0, startingCity))
	}
	return maxPoints
}

// solving for maximum points with skipped based on brainPower needed
func mostPoints(questions [][]int) int64 {
	cache := make(map[int]int64)
	var recurse func(int) int64
	recurse = func(currIndex int) int64 {
		key := currIndex
		if val, found := cache[key]; found {
			return val
		}
		if currIndex >= len(questions) {
			return 0
		}
		maxScore := 0.0
		points, brainPower := questions[currIndex][0], questions[currIndex][1]
		// skip logic simply skips the index
		skipCurrent := recurse(currIndex + 1)
		// include logic
		includeCurrent := int64(points) + int64(recurse(currIndex+brainPower+1))
		if skipCurrent > includeCurrent {
			maxScore = float64(skipCurrent)
		} else {
			maxScore = float64(includeCurrent)
		}
		cache[key] = int64(maxScore)
		return int64(maxScore)
	}

	return recurse(0)
}

// maximum dishes cooked with the given satisfaction range
func maxSatisfaction(satisfaction []int) int {
	// sorting to make sure its all in descending order
	sort.Slice(satisfaction, func(i, j int) bool {
		return satisfaction[i] < satisfaction[j]
	})

	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(currIndex, time int) int {
		// cached max
		cacheKey := strconv.Itoa(currIndex) + "-" + strconv.Itoa(time)
		if val, found := memo[cacheKey]; found {
			return val
		}

		if currIndex >= len(satisfaction) {
			return 0
		}

		maxSatisfactionScore := 0

		skipCurrent := recurse(currIndex+1, time)
		includeCurrent := satisfaction[currIndex]*time + recurse(currIndex+1, time+1)

		maxSatisfactionScore = max(skipCurrent, includeCurrent)
		memo[cacheKey] = maxSatisfactionScore
		return maxSatisfactionScore
	}

	maxDishScore := recurse(0, 1)
	if maxDishScore < 0 {
		return 0
	}
	return maxDishScore
}

// getting minimum distance based on median value between houses...
func minDistance(houses []int, k int) int {
	// sorting the houses
	sort.Slice(houses, func(i, j int) bool {
		return houses[i] < houses[j]
	})
	// cost matrix
	costs := make([][]int, len(houses))
	cache := make(map[string]int)
	// abs function for absolute substraction
	abs := func(x int) int {
		if x < 0 {
			return -x
		}
		return x
	}
	// populating cost matrix with the appropriate median index value
	for index := 0; index < len(houses); index++ {
		costs[index] = make([]int, len(houses))
		for subIndex := range costs[index] {
			costs[index][subIndex] = 0
		}
	}
	//calculating costs median value and populating the sub matrix array
	for i := 0; i < len(houses); i++ {
		for j := i; j < len(houses); j++ {
			currMedianIndex := (i + j) / 2
			currMedianValue := houses[currMedianIndex]

			totalCostPerRange := 0

			// calculating the median value for every subrange
			for rangeIndex := i; rangeIndex <= j; rangeIndex++ {
				currentRangeDistance := abs(houses[rangeIndex] - currMedianValue)
				totalCostPerRange += currentRangeDistance
			}
			// updating median cost value per sub range
			costs[i][j] = totalCostPerRange
		}
	}

	var recurse func(int, int) int
	recurse = func(currIndex, kRemaining int) int {
		// cached value for minimum cost
		cacheKey := strconv.Itoa(currIndex) + "-" + strconv.Itoa(kRemaining)
		if val, found := cache[cacheKey]; found {
			return val
		}
		// primary base conditions for checking for the subcost calculation
		if currIndex >= len(houses) {
			return 0
		}
		if kRemaining <= 0 {
			return math.MaxInt32 / 2
		}
		// when one mail box remaining it can serve all the houses... tricky
		if kRemaining == 1 {
			return costs[currIndex][len(houses)-1] // ✅ All remaining houses
		}
		minCostCalculation := math.MaxInt32 / 2
		// main dfs recursive logic to testing between every sub cost value
		for index := currIndex; index < len(houses); index++ {
			currCost := costs[currIndex][index]
			totalPreviousCost := recurse(index+1, kRemaining-1)
			if totalPreviousCost < math.MaxInt32/2 { // only can calculate the total cost if its below the desired base case limit
				minCostCalculation = min(minCostCalculation, currCost+totalPreviousCost)
			}
		}
		cache[cacheKey] = minCostCalculation
		return minCostCalculation
	}

	return recurse(0, k)
}

// getting max two events
func maxTwoEvents(events [][]int) int {
	// sorting the events based on starting time
	sort.Slice(events, func(i, j int) bool {
		return events[i][0] < events[j][0]
	})
	memo := make(map[string]int)
	// base recursion
	var recurse func(int, int) int
	recurse = func(currIndex, count int) int {
		// cached max sum event value
		cacheKey := strconv.Itoa(currIndex) + "-" + strconv.Itoa(count)
		if val, found := memo[cacheKey]; found {
			return val
		}
		// main base case for finishing the array or reaching the limit
		if count == 2 || currIndex >= len(events) {
			return 0
		}
		maxSum := 0
		// skip current event
		skipCurrent := recurse(currIndex+1, count)
		// include current
		includeCurrent := 0
		// using binary search to shorten the logic and optimizing it and returning the final index
		nextBestIndex := sort.Search(len(events), func(i int) bool {
			return events[i][0] > events[currIndex][1]
		})
		includeCurrent = recurse(nextBestIndex, count+1) + events[currIndex][2]
		maxSum = max(includeCurrent, skipCurrent)
		memo[cacheKey] = maxSum
		return maxSum
	}

	return recurse(0, 0)
}

// using dfs memo to find the maximum earnings for the taxi distance covered
func maxTaxiEarnings(n int, rides [][]int) int64 {
	// sorting the array based on starting time
	sort.Slice(rides, func(i, j int) bool {
		return rides[i][0] < rides[j][0]
	})
	memo := make(map[int]int64)

	var recurse func(int) int64
	recurse = func(currIndex int) int64 {
		// maxed cache path
		key := currIndex
		if val, found := memo[key]; found {
			return val
		}
		if currIndex >= len(rides) {
			return 0
		}
		maxRidePoints := int64((rides[currIndex][1] - rides[currIndex][0]) + rides[currIndex][2])
		// skip
		skipCurrent := recurse(currIndex + 1)
		// include current ride ... dont forget to offset the index
		offSet := currIndex + 1
		nextBestIndex := sort.Search(len(rides)-offSet, func(i int) bool {
			actualIndex := i + offSet // forcing the index to appear after the current one so there are no overlaps
			return rides[actualIndex][0] >= rides[currIndex][1]
		})
		actualIndex := offSet + nextBestIndex // need to return the next best offset index
		includeCurrent := recurse(actualIndex) + int64((rides[currIndex][1]-rides[currIndex][0])+rides[currIndex][2])

		maxRidePoints = max(skipCurrent, includeCurrent)
		memo[key] = maxRidePoints
		return maxRidePoints
	}

	return recurse(0)

}

// max operations to check three ways of pair sum whether its possible or not
// remmeber its about deleting elements
func maxOperations(nums []int) int {
	memo := make(map[string]int)
	n := len(nums)
	maxOperations := 0
	var recurse func(int, int, int) int
	recurse = func(left, right, targetScore int) int {
		cacheKey := strconv.Itoa(left) + "-" + strconv.Itoa(right) + "-" + strconv.Itoa(targetScore)
		if val, found := memo[cacheKey]; found {
			return val
		}
		if left >= right {
			return 0
		}
		maxLocalOperation := 0
		// calls
		recurseOne := 0
		recurseTwo := 0
		recurseThree := 0
		// first two elements
		if nums[left]+nums[left+1] == targetScore {
			recurseOne = 1 + recurse(left+2, right, targetScore)
		}
		if nums[left]+nums[right] == targetScore {
			recurseTwo = 1 + recurse(left+1, right-1, targetScore)
		}
		if nums[right]+nums[right-1] == targetScore {
			recurseThree = 1 + recurse(left, right-2, targetScore)
		}
		maxLocalOperation = max(recurseOne, recurseTwo, recurseThree)
		memo[cacheKey] = maxLocalOperation
		return maxLocalOperation
	}

	// first two elements skip
	targetScore := nums[0] + nums[1]
	maxOperationsOne := 1 + recurse(2, n-1, targetScore)
	// first and last element skip
	targetScore = nums[0] + nums[n-1]
	maxOperationsTwo := 1 + recurse(1, n-2, targetScore)
	// last two elements skip
	targetScore = nums[n-1] + nums[n-2]
	maxOperationsThree := 1 + recurse(0, n-3, targetScore)

	maxOperations = max(maxOperationsOne, maxOperationsTwo, maxOperationsThree)
	return maxOperations
}

// longest palindromic subsequence -> primary goal is to return the length of the sequence
func longestPalindromicSubsequence(s string, k int) int {
	n := len(s)
	memo := make([][][]int, n)
	for i := range memo {
		memo[i] = make([][]int, n)
		for j := range memo[i] {
			memo[i][j] = make([]int, k+1)
			for kk := range memo[i][j] {
				memo[i][j][kk] = -1 // -1 means not computed
			}
		}
	}

	abs := func(x int) int {
		if x < 0 {
			return -x // to counter act the negative into positive
		}
		return x
	}
	// calculates distance between characters
	minDist := func(c1, c2 byte) int {
		diff := abs(int(c1) - int(c2))
		return min(diff, 26-diff)
	}
	// main dfs function
	var recurse func(int, int, int) int
	recurse = func(left, right, operationRemaining int) int {
		// returning local max subsequence length
		if memo[left][right][operationRemaining] != -1 {
			return memo[left][right][operationRemaining]
		}
		// base cases
		if left > right {
			return 0
		}
		if left == right {
			return 1
		}
		// declaring the local maximum to check for max subsequence that is a palindrome
		localMaxLength := 0
		// main recursive calls when on the current character
		equalCharRecurse := 0
		nonEqualCharRecurse := 0

		if s[left] == s[right] {
			equalCharRecurse = 2 + recurse(left+1, right-1, operationRemaining)
		} else {
			minCostDistance := minDist(s[left], s[right])
			if minCostDistance <= operationRemaining { // only check the recursion if it falls under the condition
				nonEqualCharRecurse = 2 + recurse(left+1, right-1, operationRemaining-minCostDistance)
			}
		}
		// skip logics -> either skip from the front or the end
		skipLeftChar := recurse(left+1, right, operationRemaining)
		skipRightChar := recurse(left, right-1, operationRemaining)

		localMaxLength = max(skipLeftChar, skipRightChar, equalCharRecurse, nonEqualCharRecurse)
		memo[left][right][operationRemaining] = localMaxLength
		return localMaxLength
	}

	return recurse(0, n-1, k)
}

// getting overtyped solution
func possibleStringCount(word string, k int) int {
	groups := []int{}
	memo := make(map[[2]int]int)
	MOD := int(1e9 + 7) // to offset the exceeding lengths and values

	// populating groups with occurence
	i := 0
	for i < len(word) {
		j := i
		for j < len(word) && word[i] == word[j] {
			j += 1
		}
		groups = append(groups, j-i)
		i = j
	}

	var recurse func(int, int) int
	recurse = func(groupIndex, currLen int) int {
		// main base case
		if groupIndex >= len(groups) {
			if currLen >= k {
				return 1
			}
			return 0
		}

		// main optimization
		if currLen >= k {
			result := 1
			for keep := groupIndex; keep < len(groups); keep++ {
				result = (result * groups[keep]) % MOD
			}
			return result
		}
		// cached value
		key := [2]int{groupIndex, currLen}
		if val, found := memo[key]; found {
			return val
		}

		ways := 0
		// main dfs recursive function call to calculate all the ways
		groupSize := groups[groupIndex]
		for keep := 1; keep <= groupSize; keep += 1 {
			ways = (ways + recurse(groupIndex+1, currLen+keep)) % MOD
		}

		memo[key] = ways
		return ways
	}

	return recurse(0, 0)
}

// tallest billboard
func tallestBillboard(rods []int) int {
	memo := make(map[[2]int]int)

	var recurse func(int, int) int
	recurse = func(currIndex, currDiff int) int {
		// main base case
		if currIndex == len(rods) {
			if currDiff == 0 {
				return 0
			}
			return -(1 << 30)
		}
		// cached max
		key := [2]int{currIndex, currDiff}
		if val, found := memo[key]; found {
			return val
		}

		currMaxDim := 0
		skipCurrent := recurse(currIndex+1, currDiff)
		// add for left
		includeLeft := rods[currIndex] + recurse(currIndex+1, currDiff+rods[currIndex])
		includeRight := recurse(currIndex+1, currDiff-rods[currIndex]) // no need to add because ur reducing it here
		currMaxDim = max(includeLeft, includeRight, skipCurrent)
		memo[key] = currMaxDim
		return currMaxDim
	}

	return recurse(0, 0)
}

func predictTheWinner(nums []int) bool {
	cache := make(map[[2]int]int)
	var dfs func(int, int) int
	dfs = func(left, right int) int {
		// main base case since if its equal it will return only one of the numbers
		if left >= right {
			return nums[left]
		}
		// max cache
		key := [2]int{left, right}
		if val, found := cache[key]; found {
			return val
		}

		pickLeft := nums[left] - dfs(left+1, right)
		pickRight := nums[right] - dfs(left, right-1)

		return max(pickLeft, pickRight)
	}
	currWinner := dfs(0, len(nums)-1)
	// for winner checking
	if currWinner >= 0 {
		return true
	} else {
		return false
	}
}

// stone game
func stoneGameII(piles []int) int {
	cache := make(map[[2]int]int)

	var recurse func(int, int) int
	recurse = func(currIndex, currM int) int {
		// cached value
		key := [2]int{currIndex, currM}
		if val, found := cache[key]; found {
			return val
		}

		if currIndex >= len(piles) {
			return 0 // nothing else can be taken
		}

		maxPileSum := 0

		for take := 1; take <= 2*currM; take++ {
			totalRemaining := 0
			// accesing all the piles within the range index
			for i := currIndex; i < len(piles); i++ {
				totalRemaining += piles[i]
			}
			opponentsTotal := recurse(take+currIndex, max(currM, take))
			myTotal := totalRemaining - opponentsTotal

			maxPileSum = max(maxPileSum, myTotal)
		}
		cache[key] = maxPileSum
		return maxPileSum
	}

	return recurse(0, 1) // M should be atleast one for 2*M value
}

// count all the routes
func countRoutes(locations []int, start int, finish int, fuel int) int {
	MOD := int(1e9 + 7)
	memo := make(map[[2]int]int)

	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}

	var recurse func(int, int) int
	recurse = func(currCity, remainingFuel int) int {
		// returning caches max
		key := [2]int{currCity, remainingFuel}
		if keyVal, found := memo[key]; found {
			return keyVal
		}
		currentCount := 0

		if currCity >= len(locations) {
			return 0
		}
		if currCity == finish {
			currentCount = 1
		}

		// going through all the cities
		for nextCity := 0; nextCity < len(locations); nextCity++ {
			if currCity == nextCity { // cannot be at the same city
				continue
			}
			nextCityVal := locations[nextCity]
			fuelCost := abs(nextCityVal - locations[currCity])

			if fuelCost <= remainingFuel {
				// index should be passed here
				currentCount = (currentCount + recurse(nextCity, remainingFuel-fuelCost)) % MOD
			}
		}

		memo[key] = currentCount
		return currentCount
	}

	return recurse(start, fuel)
}

// out of boundary
func findPaths(m int, n int, maxMove int, startRow int, startColumn int) int {
	memo := make(map[[3]int]int)
	MOD := int(1e9 + 7)
	var recurse func(int, int, int) int
	recurse = func(currRow, currCol, movesLeft int) int {
		// max cached value based on currRow and currCol... default matrix structure caching
		key := [3]int{currRow, currCol, movesLeft}
		if val, found := memo[key]; found {
			return val
		}
		// base case -> valid out of bounds return 1
		if currRow >= m || currCol >= n || currCol < 0 || currRow < 0 {
			return 1
		}
		if movesLeft == 0 { // no more moves left to reduce and continue the recursive chain
			return 0
		}

		currCount := 0
		directions := [][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}

		// traversing in all four directions
		for _, dir := range directions {
			newRow, newCol := dir[0], dir[1]
			currCount = (currCount + recurse(currRow+newRow, currCol+newCol, movesLeft-1)) % MOD
		}

		memo[key] = currCount
		return currCount
	}

	return recurse(startRow, startColumn, maxMove) // main base start of the dfs structure
}

// finding the maximum spells that can be added
func maximumTotalDamage(power []int) int64 {
	freqMap := make(map[int]int)
	for _, currPower := range power {
		if val, found := freqMap[currPower]; found {
			freqMap[currPower] = val + 1
		} else {
			freqMap[currPower] = 1
		}
	}
	// for sorting it should use the current damage value
	uniqueDamageArray := []int{}
	for power, _ := range freqMap {
		uniqueDamageArray = append(uniqueDamageArray, power)
	}
	sort.Slice(uniqueDamageArray, func(i, j int) bool {
		return uniqueDamageArray[i] < uniqueDamageArray[j]
	})
	// sorting powers
	memo := make(map[int]int64) // key value

	var recurse func(int) int64
	recurse = func(currIndex int) int64 {

		key := currIndex
		if val, found := memo[key]; found {
			return val
		}
		// main base case to stop it
		if currIndex >= len(uniqueDamageArray) {
			return 0
		}
		currDamageRate := uniqueDamageArray[currIndex]
		totalDamageRate := uniqueDamageArray[currIndex] * freqMap[uniqueDamageArray[currIndex]] // gets the frequency

		skipCurrentSpell := recurse(currIndex + 1)

		nextBestIndex := sort.Search(len(uniqueDamageArray), func(i int) bool { // automatically will speak repeated numbers
			return uniqueDamageArray[i] > currDamageRate+2
		})

		takeCurrentSpell := totalDamageRate + int(recurse(nextBestIndex))
		maxDamage := max(int64(takeCurrentSpell), skipCurrentSpell)
		memo[key] = maxDamage
		return maxDamage
	}

	return recurse(0)
}

// goal is to reach the last index.. positions initially from 0 hence need to have a inside for loop to check all combination from start
func maximumJumps(nums []int, target int) int {
	memo := make(map[int]int)
	n := len(nums) - 1

	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}
	var recurse func(int) int
	recurse = func(currIndex int) int {
		key := currIndex
		if val, found := memo[key]; found {
			return val
		}
		// primary base case
		if currIndex == n {
			return 0
		}
		// if the value crosses the boundary of the total
		if currIndex >= n {
			return math.MinInt32
		}
		maxWays := -1
		currValue := nums[currIndex]
		// traversing all paths from 0
		for index := currIndex + 1; index <= n; index++ {
			nextValue := nums[index]
			if abs(nextValue-currValue) <= target {
				currRecursiveResult := recurse(index)
				if currRecursiveResult != -1 { // check is needed since -1 will lead to nowhere
					maxWays = max(maxWays, currRecursiveResult+1)
				}
			}
		}
		memo[key] = maxWays
		return maxWays
	}
	maxWays := recurse(0)
	if maxWays == 0 {
		return -1
	}
	return maxWays
}

// using dfs memo to find all the proper slots and minimize the distance
func minimumTotalDistance(robot []int, factory [][]int) int64 {
	memo := make(map[[2]int]int)
	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}
	// sorting and populating based on positions of factories and robots
	sort.Slice(robot, func(i, j int) bool {
		return robot[i] < robot[j]
	})
	slots := []int{}
	// populating slots for insertion
	for _, factoryVal := range factory {
		position, limit := factoryVal[0], factoryVal[1]
		for index := 0; index < limit; index++ {
			slots = append(slots, position)
		}
	}
	sort.Slice(slots, func(i, j int) bool {
		return slots[i] < slots[j]
	})
	// main recursive function to check for factory positioning
	var recurse func(int, int) int
	recurse = func(robotIndex, slotIndex int) int {
		// memoized min value for positions
		key := [2]int{robotIndex, slotIndex}
		if val, found := memo[key]; found {
			return val
		}
		if robotIndex >= len(robot) {
			return 0
		}
		if slotIndex >= len(slots) {
			return math.MaxInt64
		}
		minPlacements := math.MaxInt64

		skipCurrent := recurse(robotIndex, slotIndex+1) // keep the robot but dont place it in the current slot index

		currDistanceFromSlot := abs(robot[robotIndex] - slots[slotIndex])
		includeCurrent := recurse(robotIndex+1, slotIndex+1)
		if includeCurrent != math.MaxInt64 {
			includeCurrent += currDistanceFromSlot
		}

		minPlacements = min(skipCurrent, includeCurrent)
		memo[key] = minPlacements
		return minPlacements
	}

	return int64(recurse(0, 0))
}

// 0,1 Knapsack style problem to reach the target
func waysToReachTarget(target int, types [][]int) int {
	cache := make(map[[2]int]int)
	MOD := int(1e9 + 7)
	var recurse func(int, int) int
	recurse = func(currIndex, remainingPoints int) int {
		// getting the total number of cached ways of reaching the target
		key := [2]int{currIndex, remainingPoints}
		if val, found := cache[key]; found {
			return val
		}
		// main base conditions
		if currIndex >= len(types) {
			if remainingPoints == 0 {
				return 1
			}
			return 0
		}
		if remainingPoints == 0 {
			return 1
		}
		if remainingPoints < 0 {
			return 0
		}

		totalWays := 0
		currPerScore := types[currIndex][1]
		targetCount := types[currIndex][0]
		// checking for every count
		for count := 0; count <= targetCount; count++ {
			currScore := currPerScore * count
			// does not have enough points to
			if currScore <= remainingPoints {
				totalWays = (totalWays + recurse(currIndex+1, remainingPoints-currScore)) % MOD
			}
		}

		cache[key] = totalWays
		return totalWays
	}

	return recurse(0, target) // if the remaining points hit 0 that is a valid path
}

// finding the best ways to divide the corridors accurately -- FLAWD logic
func numberOfWaysCorridor(corridor string) int {
	memo := make(map[[2]int]int)
	MOD := int(1e9 + 7)
	currAvailableSeats := 0

	for _, val := range corridor {
		if val == 'S' {
			currAvailableSeats++
		}
	}
	if currAvailableSeats == 0 || currAvailableSeats%2 != 0 {
		return 0
	}
	// will return the total number of ways
	var recurse func(int, int) int
	recurse = func(currIndex, seatsInSection int) int {
		// cached ways for seats in section
		key := [2]int{currIndex, seatsInSection}
		if val, found := memo[key]; found {
			return val
		}
		// main base case
		if currIndex >= len(corridor) {
			if seatsInSection == 0 || seatsInSection == 2 { // check for 2 because if its the last row and 0 for reset
				return 1
			}
			return 0
		}

		totalDividingWays := 0
		currPosition := corridor[currIndex]
		applyDivide := 0
		keepSeats := 0

		if currPosition == 'S' {
			if seatsInSection+1 > 2 {
				return 0
			}
			totalDividingWays = recurse(currIndex+1, seatsInSection+1) % MOD // since its a seat it will increment the seat
		}
		// skip or current approach will be applied
		if currPosition == 'P' {
			if seatsInSection == 2 {
				// only two conditions that are part of the seat count
				applyDivide = recurse(currIndex+1, 0) % MOD
				keepSeats = recurse(currIndex+1, seatsInSection) % MOD
				totalDividingWays = (applyDivide + keepSeats) % MOD
			} else {
				totalDividingWays = recurse(currIndex+1, seatsInSection) % MOD // skip this plant and apply divide later
			}
		}
		memo[key] = totalDividingWays
		return totalDividingWays

	}

	return recurse(0, 0)
}

// getting the minimum number of ways to match target
func minValidStrings(words []string, target string) int {
	memo := make(map[int]int)
	// main recursive function to check through each word
	var recurse func(int) int
	recurse = func(currIndex int) int {
		key := currIndex
		if val, found := memo[key]; found {
			return val
		}
		if currIndex >= len(target) { // no more ways found since min ways reached
			return 0
		}
		// each valid prefix that matches the target word is a valid path
		minWays := math.MaxInt32
		// checking for all possible prefixes
		for _, word := range words {
			for length := 1; length <= min(len(word), len(target)-currIndex); length++ {
				// if the prefix matches the target prefix then we continue from there
				checkValid := len(target) >= currIndex+length
				if checkValid {
					checkMatch := word[0:length] == target[currIndex:currIndex+length]
					if !checkMatch {
						break
					}
					if checkMatch {
						minWays = min(minWays, 1+recurse(currIndex+length))
					}
				}

			}
		}
		memo[key] = minWays
		return minWays
	}

	minWays := recurse(0)
	if minWays == math.MaxInt32 {
		return -1
	}
	return minWays
}

// climbing stairs II
func climbStairs(n int, costs []int) int {
	memo := make(map[int]int)

	var dfs func(int) int
	dfs = func(index int) int {
		key := index
		if val, found := memo[key]; found {
			return val
		}
		if index == n {
			return 0
		}
		minTotalCost := math.MaxInt32

		// jump indices
		for jump := 1; jump < 4; jump++ {
			next_step := index + jump
			if next_step <= n {
				minTotalCost = min(minTotalCost, costs[next_step-1]+jump*jump+dfs(next_step))
			}
		}
		memo[key] = minTotalCost
		return minTotalCost
	}

	return dfs(0)
}

// longest ideal string subsequence
func longestIdealString(s string, k int) int {
	memo := make(map[[2]int]int)

	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}

	var recurse func(int, byte) int
	recurse = func(currIndex int, last_char byte) int {

		key := [2]int{currIndex, int(last_char)}
		if val, found := memo[key]; found {
			return val
		}

		if currIndex >= len(s) {
			return 0
		}
		maxLen := 0

		skipCurrent := recurse(currIndex+1, last_char)
		includeCurrent := 0
		diff := abs(int(s[currIndex]) - int(last_char))

		if last_char == 0 || diff <= k {
			includeCurrent = 1 + recurse(currIndex+1, s[currIndex])
		}

		maxLen = max(skipCurrent, includeCurrent)
		memo[key] = maxLen
		return maxLen
	}

	return recurse(0, 0)
}

// HARD - getting strictly increasing arrays after replacing elements from arr2 to arr1
func makeArrayIncreasing(arr1 []int, arr2 []int) int {
	memo := make(map[[2]int]int)
	// sorting the arr 2 for easier extraction
	sort.Slice(arr2, func(i, j int) bool {
		return arr2[i] < arr2[j]
	})
	// removing duplicates from arr2
	unique := []int{}
	for index, value := range arr2 {
		if index == 0 || arr2[index-1] != value {
			unique = append(unique, arr2[index])
		}
	}
	arr2 = unique

	var recurse func(int, int) int
	recurse = func(index, prevValue int) int {
		// cached max value ¥
		key := [2]int{index, prevValue}
		if val, found := memo[key]; found {
			return val
		}
		// base case
		if index >= len(arr1) {
			return 0
		}
		increaseSequence := math.MaxInt32
		notIncreasingSequence := math.MaxInt32

		if arr1[index] > prevValue {
			increaseSequence = recurse(index+1, arr1[index]) // updating the previous value if the sequence is correct
		}
		// replacement should be allowed for both conditions because this might lead to better sequences
		closestValueIndex := sort.Search(len(arr2), func(i int) bool {
			return arr2[i] > prevValue
		})
		if closestValueIndex < len(arr2) {
			notIncreasingSequence = 1 + recurse(index+1, arr2[closestValueIndex])
		}

		minWays := min(notIncreasingSequence, increaseSequence)
		memo[key] = minWays
		return minWays
	}

	minWays := recurse(0, math.MinInt32)
	if minWays == math.MaxInt32 {
		return -1
	}
	return minWays
}

// rotating for minimum distance
func findRotateSteps(ring string, key string) int {
	// populating the index array with character
	ringMap := make(map[byte][]int)
	for index, character := range ring {
		ringMap[byte(character)] = append(ringMap[byte(character)], index)
	}
	memo := make(map[[2]int]int) // caching for memo
	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}
	var recurse func(int, int) int
	recurse = func(ringIndex, keyIndex int) int {
		// key index for storing
		keyMemo := [2]int{ringIndex, keyIndex}
		if val, found := memo[keyMemo]; found {
			return val
		}
		// base case
		if len(key) == keyIndex {
			return 0
		}
		minCost := math.MaxInt32
		for _, targetIndex := range ringMap[byte(key[keyIndex])] {
			clockwiseDistance := abs(ringIndex - targetIndex)
			antiClockwise := len(ring) - clockwiseDistance
			localCost := 1 + recurse(targetIndex, keyIndex+1) + min(antiClockwise, clockwiseDistance)
			minCost = min(localCost, minCost)
		}
		memo[keyMemo] = minCost
		return minCost

	}

	return recurse(0, 0) // returning the minimum path
}

func minHeightShelves(books [][]int, shelfWidth int) int {
	memo := make(map[int]int)

	var recurse func(int) int
	recurse = func(currIndex int) int {

		key := currIndex
		if val, found := memo[key]; found {
			return val
		}
		if currIndex >= len(books) {
			return 0
		}
		minShelfHeight := math.MaxInt32
		currShelfWidth := 0
		currShelfHeight := 0

		// recursing through all the shelf for maximum width and height efficiency
		for index := currIndex; index < len(books); index++ {
			currThickNess := books[index][0]
			currHeight := books[index][1]
			currShelfWidth += currThickNess
			// break the recursive chain if the current shelf width increases more the count shelf width
			if currShelfWidth > shelfWidth {
				break
			}
			currShelfHeight = max(currShelfHeight, currHeight)
			recursedHeightVal := recurse(index+1) + currShelfHeight
			minShelfHeight = min(recursedHeightVal, minShelfHeight)

		}
		memo[key] = minShelfHeight
		return minShelfHeight
	}

	return recurse(0)
}

// minimum grid moves
func minimumMovesGrid(grid [][]int) int {
	memo := make(map[string]int)
	// abs value
	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}
	// distance calculation for empty and excess array matrix for index traversal
	distanceCalc := func(distanceOne []int, distanceTwo []int) int {
		x1, y1 := distanceOne[0], distanceOne[1]
		x2, y2 := distanceTwo[0], distanceTwo[1]
		return abs(x1-x2) + abs(y1-y2)
	}
	// populate exceed and 0 locations
	excess := [][]int{}
	empty := [][]int{}
	used := []bool{}

	// excess = means more than 1 and empty which is 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			currVal := grid[i][j]
			if currVal > 1 {
				// 1 remains in the cell
				for sub := 0; sub < currVal-1; sub++ {
					excess = append(excess, []int{i, j})
				}
			}
			if currVal == 0 {
				empty = append(empty, []int{i, j})
			}
		}
	}
	// After the loops
	used = make([]bool, len(excess))

	// main recursive functionality
	var recurse func(int, []bool) int
	recurse = func(emptyIndex int, used []bool) int {
		// memoized min
		key := fmt.Sprintf("%d-%v", emptyIndex, used)
		if val, found := memo[key]; found {
			return val
		}
		if emptyIndex >= len(empty) {
			return 0
		}

		minMoves := math.MaxInt32
		currentEmptyIndexVal := empty[emptyIndex]
		// for loop recursion step ... every step for index recursion needs to be checked
		for index, val := range excess {
			currDistance := distanceCalc(val, currentEmptyIndexVal)
			if !used[index] { // if its not in the used set then get the new value
				used[index] = true
				newValue := currDistance + recurse(emptyIndex+1, used)
				minMoves = min(newValue, minMoves)
				used[index] = false // for backtracking
			}

		}
		memo[key] = minMoves
		return minMoves
	}

	return recurse(0, used)
}

// finding the most quiet among the pool of rich people
// wrong
func loudAndRich(richer [][]int, quiet []int) []int {
	n := len(quiet)
	quietest := make([]int, n)
	for index := 0; index < len(quietest); index++ {
		quietest[index] = -1
	}
	graph := make(map[int][]int)
	// making the graph to show a has more money than b
	for _, pair := range richer {
		a, b := pair[0], pair[1]
		graph[b] = append(graph[b], a)
	}
	// main recursive function to calculate
	var recurse func(int) int
	recurse = func(personId int) int {
		// if there is no value then return the quietest right then and there
		if quietest[personId] != -1 {
			return quietest[personId]
		}
		quietPerson := personId
		currQuitestPersonRicher := graph[quietPerson]
		currQuitestPersonValue := quiet[quietPerson]

		// looping through to check and update the quietest person
		for _, currRicherPerson := range currQuitestPersonRicher {
			currQuietestPerson := recurse(currRicherPerson)
			currRicherQuietValue := quiet[currRicherPerson] // currRichPerson quiet value
			if quiet[currQuietestPerson] > currQuitestPersonValue {
				currQuitestPersonValue = currRicherQuietValue
			}
		}
		quietest[personId] = currQuitestPersonValue
		return currQuitestPersonValue
	}

	// getting the quietest person from each start
	for i := 0; i < n; i++ {
		recurse(i)
	}
	return quietest
}

// getting the strangePrinter to print letters and names
func strangePrinter(s string) int {
	n := len(s)
	memo := make(map[[2]int]int)

	var recurse func(int, int) int
	recurse = func(leftIndex, rightIndex int) int {
		// returning the memo for fixed minWays
		key := [2]int{leftIndex, rightIndex}
		if val, found := memo[key]; found {
			return val
		}
		// valid counting point when both the index are equal
		if leftIndex == rightIndex {
			return 1
		}
		if leftIndex > rightIndex {
			return 0
		}
		minWays := math.MaxInt32

		if s[leftIndex] == s[rightIndex] {
			minWays = min(minWays, recurse(leftIndex, rightIndex-1)) // one side iteration is fine because it will be the same result
		} else {
			currMinWays := math.MaxInt32
			for k := leftIndex; k < rightIndex; k++ {
				currMinWays = min(currMinWays, recurse(leftIndex, k)+recurse(k+1, rightIndex)) // for both sides substring
			}
			minWays = min(minWays, currMinWays)
		}

		memo[key] = minWays
		return minWays
	}

	minTurns := recurse(0, n-1)
	return minTurns
}

// buy and sell stock while the transaction is k remaining
func maxProfit(k int, prices []int) int {
	memo := make(map[[3]int]int) // holding will be used as 0 and 1

	var recurse func(int, int, int) int
	recurse = func(day, k, holding int) int {
		// memoized max profit
		key := [3]int{day, k, holding}
		if val, found := memo[key]; found {
			return val
		}
		// no more calculations if day is completed and k == 0
		if day >= len(prices) || k == 0 {
			return 0
		}
		maxProfit := 0

		if holding == 0 {
			// choose to buy or not to buy
			buy := -prices[day] + recurse(day+1, k, 1)
			dontBuy := recurse(day+1, k, holding) // keep the same
			maxProfit = max(buy, dontBuy)

		} else if holding == 1 {
			sell := prices[day] + recurse(day+1, k-1, 0)
			dontSell := recurse(day+1, k, holding)
			maxProfit = max(sell, dontSell)
		}
		memo[key] = maxProfit
		return maxProfit
	}

	return recurse(0, k, 0)
}

// deleting rows while keeping the overall string sorted
func minDeletionSize(strs []string) int {
	memo := make(map[[2]int]int)
	colLen := len(strs[0]) // length of a single string
	rowLen := len(strs)

	// checking whether current vertical rows are correct or not as soon as a single
	checkRowValid := func(curr_col, prev_col int) bool {
		state := true
		for i := 0; i < rowLen; i++ {
			if strs[i][curr_col] < strs[i][prev_col] { // if one condition fails then return entirely
				return false
			}
		}
		return state
	}

	var recurse func(int, int) int
	recurse = func(curr_col, prev_col int) int {
		// memoized cached items for min ways
		key := [2]int{curr_col, prev_col}
		if val, found := memo[key]; found {
			return val
		}
		// main base case
		if curr_col >= colLen {
			return 0
		}

		skipCurrCol := recurse(curr_col+1, prev_col) // keeping the prev col intact
		includeCurrCol := 0

		// include either because it will add the next sequence if the col ascending order is confirmed
		if prev_col == -1 || checkRowValid(curr_col, prev_col) {
			includeCurrCol = 1 + recurse(curr_col+1, curr_col)
		}

		maxKept := max(skipCurrCol, includeCurrCol)
		memo[key] = maxKept
		return maxKept
	}

	maxKept := recurse(0, -1)
	return colLen - maxKept
}

// minimum difficult of the splits that can happen
func minDifficulty(jobDifficulty []int, d int) int {
	memo := make(map[[2]int]int)

	var recurse func(int, int) int
	recurse = func(currIndex, days_remaining int) int {
		// cached min ways
		key := [2]int{currIndex, days_remaining}
		if val, found := memo[key]; found {
			return val
		}
		minDifficulty := math.MaxInt32
		// if the days == 1 then return the remaining numbers
		if days_remaining == 1 {
			maxDiff := 0
			for index := currIndex; index < len(jobDifficulty); index++ {
				maxDiff = max(maxDiff, jobDifficulty[index])
			}
			return maxDiff
		}
		// no more jobs remaining
		if currIndex >= len(jobDifficulty) {
			return 0
		}
		// looping through splits while keeping valid splits
		maxCurrDiffSoFar := 0
		for end := currIndex; end <= len(jobDifficulty)-days_remaining; end++ {
			maxCurrDiffSoFar = max(maxCurrDiffSoFar, jobDifficulty[end]) // incremental updates
			currRecurseRes := recurse(end+1, days_remaining-1)           // returning the max ways from other paths
			totalWays := maxCurrDiffSoFar + currRecurseRes
			minDifficulty = min(totalWays, minDifficulty)
		}
		memo[key] = minDifficulty
		return minDifficulty
	}

	minWays := recurse(0, d)
	// max check
	if minWays == math.MaxInt32 {
		return -1
	}
	return minWays
}

// minimum time required for all assigned workers
func minimumTimeRequired(jobs []int, k int) int {
	workers := make([]int, k) // assigned workers based on the rate of k with default 0

	var backtrack func(int, int) int
	backtrack = func(jobIndex int, currBest int) int {
		// main base case to return the current maxLoad borne by one of the workers
		if jobIndex >= len(jobs) {
			maxLoad := 0
			for _, load := range workers {
				maxLoad = max(maxLoad, load)
			}
			return maxLoad
		}

		minWays := math.MaxInt32 // current min ways

		// checking for every worker combinations to populate the workers
		for index := 0; index < k; index++ {
			if workers[index] >= minWays {
				continue
			}
			workers[index] += jobs[jobIndex]

			currMaxLoad := 0
			for _, load := range workers {
				currMaxLoad = max(load, currMaxLoad)
			}

			if currMaxLoad < minWays {
				maxBacktrackedLoad := backtrack(jobIndex+1, minWays)
				minWays = min(minWays, maxBacktrackedLoad)
			}
			// backtracking step
			workers[index] -= jobs[jobIndex]
			if workers[index] == 0 {
				break
			}
		}
		return minWays
	}

	return backtrack(0, math.MaxInt32)
}

// buying and selling stocks with cool down period right after selling
func maxProfitCooldown(prices []int) int {
	memo := make(map[[2]int]int)

	var recurse func(int, int) int
	recurse = func(currIndex, isHolding int) int {
		// max cached profit
		key := [2]int{currIndex, isHolding}
		if val, found := memo[key]; found {
			return val
		}
		// stock becomes worthless either way since no more days to sell
		if currIndex >= len(prices) {
			return 0
		}
		maxProfit := 0

		// there is option of holding stock and do nothing and cool down period only activates when selling
		if isHolding == 0 {
			buy := -prices[currIndex] + recurse(currIndex+1, 1)
			holdBuy := recurse(currIndex+1, isHolding)
			maxProfit = max(holdBuy, buy)
		} else {
			sell := +prices[currIndex] + recurse(currIndex+2, 0)
			holdSell := recurse(currIndex+1, isHolding)
			maxProfit = max(holdSell, sell)
		}

		memo[key] = maxProfit
		return maxProfit
	}

	// stating isHolding as 0 and 1 for true and false
	return recurse(0, 0)
}

func splitArray(nums []int, k int) int {
	memo := make(map[[2]int]int)

	var recurse func(int, int) int
	recurse = func(array_index, remaining int) int {
		// memoized min max value
		key := [2]int{array_index, remaining}
		if val, found := memo[key]; found {
			return val
		}
		if remaining == 1 { // return the final sum of the partition of the remaining  value is 1 since its last partition
			currSum := 0
			for index := array_index; index < len(nums); index++ {
				currSum += nums[index]
			}
			return currSum
		}
		minLargestSum := math.MaxInt32

		currSum := 0 // local accumulated sum
		for index := array_index; index <= len(nums)-remaining; index++ {
			currSum += nums[index]
			if currSum >= minLargestSum {
				break
			}
			recursedSum := recurse(index+1, remaining-1)
			currMaxSum := max(recursedSum, currSum)
			minLargestSum = min(minLargestSum, currMaxSum)
		}
		memo[key] = minLargestSum
		return minLargestSum
	}

	minLarge := recurse(0, k)
	return minLarge
}

// backtracking question
func getWordsInLongestSubsequence(words []string, groups []int) []string {
	// Memoization: key is [currIndex, lastIndex+1], value is best subsequence
	memo := make(map[[2]int][]string)

	canSelect := func(index int, lastIndex int) bool {
		if lastIndex == -1 {
			return true
		}

		if groups[index] == groups[lastIndex] {
			return false
		}

		word := words[index]
		checkWord := words[lastIndex]

		if len(word) != len(checkWord) {
			return false
		}

		hammingDistance := 0
		for i := 0; i < len(word); i++ {
			if word[i] != checkWord[i] {
				hammingDistance++
				if hammingDistance > 1 {
					return false
				}
			}
		}
		return hammingDistance == 1
	}

	var recurse func(int, int) []string
	recurse = func(currIndex, lastIndex int) []string {
		// Base case
		if currIndex >= len(words) {
			return []string{}
		}

		// Check memo
		key := [2]int{currIndex, lastIndex + 1} // +1 to handle -1
		if val, exists := memo[key]; exists {   // memoization removes the bottle neck of redundant searches
			return val
		}

		// Skip current word
		skip := recurse(currIndex+1, lastIndex)

		// Include current word if valid
		include := []string{}
		if canSelect(currIndex, lastIndex) {
			next := recurse(currIndex+1, currIndex)
			include = append([]string{words[currIndex]}, next...)
		}

		// Choose the longer one
		result := skip
		if len(include) > len(skip) {
			result = include
		}

		memo[key] = result
		return result
	}

	return recurse(0, -1)
}

// cutting wood vertically and horizontally.. goal is to return the maximum cost possible
func sellingWood(m int, n int, prices [][]int) int64 {
	memo := make(map[[2]int]int)
	priceMap := make(map[[2]int]int)

	// populating the price map to return the cost of present cuts directly
	for _, val := range prices {
		height, width, price := val[0], val[1], val[2]
		priceMap[[2]int{height, width}] = price
	}
	// main recursive function to cutting and checking through all the maximum pieces through the recursive chain
	var recurse func(int, int) int
	recurse = func(height, width int) int {
		// main base case.. if any of the dims hit 0 then there is no cost incurred
		if height == 0 || width == 0 {
			return 0
		}
		// memoized maxvalue
		key := [2]int{height, width}
		if val, found := memo[key]; found {
			return val
		}
		// will store the current maxPriceValue that will be current if any or 0 if no value present for the current cuts
		maxPriceValue := priceMap[[2]int{height, width}]
		// cutting horizontally means cutting the height
		for i := 1; i < height; i++ {
			topPiece := recurse(i, width)
			bottomPiece := recurse(height-i, width)
			maxPriceValue = max(maxPriceValue, topPiece+bottomPiece)
		}
		// cutting vertically means cutting the width
		for i := 1; i < width; i++ {
			leftPiece := recurse(height, i)
			rightPiece := recurse(height, width-i)
			maxPriceValue = max(maxPriceValue, leftPiece+rightPiece)
		}

		memo[key] = maxPriceValue // storing the memoized max price value within the memo
		return maxPriceValue
	}

	return int64(recurse(m, n)) // passing the original sizes
}

// minimum coins
func minimumCoins(prices []int) int {
	memo := make(map[[2]int]int)
	// main recursive function
	var recurse func(int, int) int
	recurse = func(currIndex, freeLimit int) int {
		if currIndex >= len(prices) {
			return 0
		}
		// memoized index
		key := [2]int{currIndex, freeLimit}
		if val, found := memo[key]; found {
			return val
		}
		minCoins := math.MaxInt32

		if currIndex > freeLimit {
			minCoins = recurse(currIndex+1, currIndex+currIndex+1) + prices[currIndex]
		} else {
			skipCurrent := recurse(currIndex+1, freeLimit)
			includeCurrent := recurse(currIndex+1, max(freeLimit, currIndex+currIndex+1)) + prices[currIndex]
			minCoins = min(skipCurrent, includeCurrent)
		}
		memo[key] = minCoins
		return minCoins
	}

	return recurse(0, -1)
}

// miniimize the difference of matrix -> row wise movement
func minimizeTheDifference(mat [][]int, target int) int {
	memo := make(map[[2]int]int)
	rowLen := len(mat)

	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}

	var recurse func(int, int) int
	recurse = func(rowIndex, currSum int) int {
		// main base to return the smallest difference
		if rowIndex == rowLen {
			return abs(currSum - target)
		}
		if currSum > target+5000 {
			return abs(currSum - target)
		}
		// memoized min value
		key := [2]int{rowIndex, currSum}
		if val, found := memo[key]; found {
			return val
		}
		currRow := mat[rowIndex]
		minimumVal := math.MaxInt32
		// only loop for each horizontal row choices
		for _, choice := range currRow {
			currRes := recurse(rowIndex+1, currSum+choice)
			minimumVal = min(currRes, minimumVal)
		}
		memo[key] = minimumVal
		return minimumVal
	}

	return recurse(0, 0)
}

// matrix sum
func maxMatrixSum(matrix [][]int) int64 {

	abs := func(num int) int {
		if num < 0 {
			return -num
		}
		return num
	}
	negativeCount := 0
	smallestElement := math.MaxInt32
	totalSum := 0
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[i]); j++ {
			element := matrix[i][j]
			if element < 0 {
				negativeCount++
			}
			totalSum += abs(element)
			smallestElement = min(smallestElement, abs(element))
		}
	}

	oddCheck := false
	if negativeCount%2 == 1 {
		oddCheck = true
	}

	if oddCheck {
		totalSum = totalSum - (2 * smallestElement)
		return int64(totalSum)
	}
	return int64(totalSum)
}

// minimum cost for splitting array for trimmed importance value
func minCostSplitImportance(nums []int, k int) int {
	memo := make(map[int]int)

	var recurse func(int) int
	recurse = func(currStart int) int {
		key := currStart
		if val, found := memo[key]; found {
			return val
		}
		// main base case
		if currStart >= len(nums) {
			return 0
		}
		minCost := math.MaxInt32
		freqMap := make(map[int]int)
		trimLen := 0

		for index := currStart; index < len(nums); index++ {
			currNum := nums[index]
			freqMap[currNum]++
			newOccurence := freqMap[currNum]

			if newOccurence == 2 {
				trimLen += 2
			} else if newOccurence > 2 {
				trimLen++
			}

			importanceValue := trimLen + k
			existingRes := recurse(index + 1)
			minCost = min(minCost, importanceValue+existingRes)
		}
		memo[key] = minCost
		return minCost
	}

	minCost := recurse(0)
	return minCost
}

func countStableSubsequences(nums []int) int {
	const MOD = 1000000007
	memo := make(map[[3]int]int)
	n := len(nums)

	var recurse func(int, int, int) int
	recurse = func(currIndex, lastParity, consequtiveCount int) int {
		// main base case completed subsequence
		if currIndex >= n {
			return 1
		}
		currRes := 0
		// memoized value
		key := [3]int{currIndex, lastParity, consequtiveCount}
		if val, found := memo[key]; found {
			return val
		}

		currNum := nums[currIndex]
		canTake := false
		currParity := currNum % 2
		newParity := currParity
		newCount := 0

		// skip current but add to final result
		currRes += recurse(currIndex+1, lastParity, consequtiveCount)

		if lastParity == -1 {
			canTake = true
			newCount = 1
		} else if currParity != lastParity {
			canTake = true
			newCount = 1
		} else if consequtiveCount < 2 { // means its one
			canTake = true
			newCount = 2
		}

		if canTake {
			currRes += recurse(currIndex+1, newParity, newCount)
		}
		currRes %= MOD
		memo[key] = currRes
		return currRes
	}

	return recurse(0, -1, 0) - 1
}

// medium level for dfs memoization exploration
func maxBalancedShipments(weight []int) int {
	memo := make(map[int]int)

	var recurse func(int) int
	recurse = func(currIndex int) int {
		//maximum memoized weight partition count
		key := currIndex
		if val, found := memo[key]; found {
			return val
		}
		if currIndex >= len(weight) {
			return 0
		}
		currMaxWeight := weight[currIndex]
		maxPartitions := recurse(currIndex + 1)

		// including current and getting all partitions
		for index := currIndex; index < len(weight); index++ {
			currWeight := weight[index]
			currMaxWeight = max(currWeight, currMaxWeight)

			if currWeight < currMaxWeight {
				includeCurrent := 1 + recurse(index+1)
				maxPartitions = max(includeCurrent, maxPartitions)
			}
		}

		memo[key] = maxPartitions
		return maxPartitions
	}

	return recurse(0)
}

// coloring border connected components with dfs
func colorBorder(grid [][]int, row int, col int, color int) [][]int {
	visitedCoords := make(map[[2]int]bool)
	borderCoords := [][2]int{} // will contain the border coords
	rowLen := len(grid)
	colLen := len(grid[0])
	originalColor := grid[row][col]

	var recurse func(int, int)
	recurse = func(row, col int) {
		// boundary check
		if row >= rowLen || col >= colLen || row < 0 || col < 0 || grid[row][col] != originalColor {
			return
		}
		if visitedCoords[[2]int{row, col}] {
			return
		}
		// adding to visited coors
		visitedCoords[[2]int{row, col}] = true
		// checking for borders
		if row == 0 || col == 0 || row == rowLen-1 || col == colLen-1 {
			borderCoords = append(borderCoords, [2]int{row, col})
		} else {
			// checking for atleast one different neighbor
			currCell := grid[row][col]
			directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
			for _, dir := range directions {
				newCol := col + dir[1]
				newRow := row + dir[0]
				if currCell != grid[newRow][newCol] {
					borderCoords = append(borderCoords, [2]int{row, col})
					break
				}
			}
		}

		// four directional checks
		recurse(row+1, col)
		recurse(row, col+1)
		recurse(row-1, col)
		recurse(row, col-1)

	}
	recurse(row, col)

	// coloring the borders
	for index := 0; index < len(borderCoords); index++ {
		row, col := borderCoords[index][0], borderCoords[index][1]
		grid[row][col] = color
	}

	return grid
}

// using minimum jumps to reach the target with constraints
func minimumJumps1654(forbidden []int, a int, b int, x int) int {
	memo := make(map[[2]int]int)
	result := 0
	boolToInt := func(val bool) int {
		if val {
			return 1
		}
		return 0
	}
	//assigning forbidden map
	maxForbidden := 0
	forbiddenMap := make(map[int]bool)
	for _, forbiddenVal := range forbidden {
		forbiddenMap[forbiddenVal] = true
		if forbiddenVal > maxForbidden {
			maxForbidden = forbiddenVal
		}
	}
	upperLimit := 2000 + a + b

	// main recursive function
	var recurse func(int, bool) int
	recurse = func(currPosition int, wasBackward bool) int {
		if currPosition == x { // target has been reached
			return 0
		}
		// main base case ot limit current position
		if currPosition > upperLimit || forbiddenMap[currPosition] || currPosition < 0 {
			return math.MaxInt32
		}
		// memoized base case for memoized minimum jumps
		key := [2]int{currPosition, boolToInt(wasBackward)}
		if val, found := memo[key]; found {
			return val
		}
		// base case for forbidden check
		minJumps := math.MaxInt32

		// always can move forward
		forward := recurse(currPosition+a, false)
		if forward < math.MaxInt32 {
			totalJumps := forward + 1
			minJumps = min(minJumps, totalJumps)
		}

		// only calls if wasBackward is false
		if !wasBackward {
			backward := recurse(currPosition-b, true)
			if backward < math.MaxInt32 {
				totalJumps := backward + 1
				minJumps = min(minJumps, totalJumps)
			}
		}

		memo[key] = minJumps
		return minJumps
	}

	result = recurse(0, false)

	if result == math.MaxInt32 {
		return -1
	}
	return result
}

// checking number of arithmetic slices and returning all of the subsequence
func numberOfArithmeticSlices(nums []int) int {
	memo := make(map[string]int) // memoization for storing maximum subsequence length from starting point
	result := 0

	// main recursive function -- its already pruned hence I wont need to calculate length
	var recurse func(int, int, int) int
	recurse = func(currIndex, lastValue, lastDiff int) int {
		// main base case to return the length
		if currIndex == len(nums) {
			return 0
		}
		// memoized return value
		key := fmt.Sprintf("%d,%d,%d", currIndex, lastValue, lastDiff)
		if val, found := memo[key]; found {
			return val
		}

		// skip current number
		count := recurse(currIndex+1, lastValue, lastDiff)

		// include current but need to check whether its valid or not based on the currDiff check
		currDiffRecursiveTree := nums[currIndex] - lastValue
		if currDiffRecursiveTree == lastDiff {
			count += 1
			count += recurse(currIndex+1, nums[currIndex], currDiffRecursiveTree)
		}

		memo[key] = count
		return count

	}

	// getting the starting points using a nested loops
	for i := 0; i < len(nums); i++ {
		for j := i + 1; j < len(nums); j++ {
			currDiff := nums[j] - nums[i]
			result += recurse(j+1, nums[j], currDiff)
		}
	}

	return result
}

// calculating where the ball will fall after its being dropped from the top row
func findBall(grid [][]int) []int {
	rowLen := len(grid)
	colLen := len(grid[0])
	dropLocation := make([]int, colLen)

	// main dfs function to check all the possibble path when the ball is dropped
	var recurse func(int, int) int
	recurse = func(row, col int) int {
		// if it reaches the last row that means we have an exit point
		if row == rowLen {
			return col
		}
		nextRow := row + 1 // row will always increment
		direction := grid[row][col]
		nextCol := col + direction

		// ball gets stuck
		if nextCol < 0 || nextCol >= colLen {
			return -1
		}
		nextDirection := grid[row][nextCol] // initialize next direction after checking
		// vtrap pattern check
		if direction != nextDirection {
			return -1
		}
		// since the direction decides the nextRow and nextCol we can simply return here
		return recurse(nextRow, nextCol)
	}

	// will populate the drop location once the ball drops all the way down
	for colIndex := 0; colIndex < colLen; colIndex++ {
		dropLocation[colIndex] = recurse(0, colIndex)
	}

	return dropLocation
}
