package main

import (
	"fmt"
	"math"
	"math/big"
	"encoding/hex"
	"github.com/obscuren/sha3"
	"testing"
 	"strconv"
)

//For use in benchmarking
const tree_depth = 5
const tape_width = 32 //int(math.Pow(2,tree_depth))  < would be nice, but go won't recognize as const
const tape_depth = 100

//Number of operations that are drawn from to form the tape and the tree
const num_ops = 9
//This is the number of times the PoW algorithm is run
const sample_size = 100

func Bytes2Hex(d []byte) string {
        return hex.EncodeToString(d)
}

func Hex2Bytes(str string) []byte {
        h, _ := hex.DecodeString(str)
        return h
}

func plus(z *big.Int, x *big.Int, y *big.Int) *big.Int {
	var lim big.Int
	lim.Exp(big.NewInt(2),big.NewInt(256), big.NewInt(0))
	z.Add(x,y)
	return z.Mod(z,&lim)
	
}

func times(z *big.Int, x *big.Int, y *big.Int) *big.Int {
        var lim, x1, y1 big.Int
        lim.Exp(big.NewInt(2),big.NewInt(256), big.NewInt(0))
	x1.Set(x)
	y1.Set(y)
        z.Mul(x,y)
        z.Mod(z,&lim)
	return z
}

func mod(z *big.Int, x *big.Int, y *big.Int) *big.Int {

        if (x.Cmp(big.NewInt(0)) == 0 || y.Cmp(big.NewInt(0)) == 0) {
		return big.NewInt(0)
	}	
	if x.Cmp(y) == -1 { //if x < y
		z.Mod(y,x)
	} else if x.Cmp(y) == 1 { //if x > y
		z.Mod(x,y)
	}
	return z
}

func xor(z *big.Int, x *big.Int, y *big.Int) *big.Int {return z.Xor(x,y)}

func nxor(z *big.Int, x *big.Int, y *big.Int) *big.Int {
	z.Xor(x,y)
	return z.Not(z)
}

func and(z *big.Int, x *big.Int, y *big.Int) *big.Int {return z.And(x,y)}

func not(z *big.Int, x *big.Int, y *big.Int) *big.Int {
	_ = y
	return z.Not(x)
}

func or(z *big.Int, x *big.Int, y *big.Int) *big.Int {return z.Or(x,y)}

func rshift(z *big.Int, x *big.Int, y *big.Int) *big.Int {
       	var lim big.Int
        lim.Exp(big.NewInt(2),big.NewInt(256), big.NewInt(0))
	z.Rsh(x,7)
	return z.Mod(z,&lim)
}


func GetOp(i int) func(z *big.Int, x *big.Int, y *big.Int) *big.Int{
	switch i {
	case 0:
		return plus
	case 1:
		return xor
	case 2:
		return not
	case 3:
		return times
	case 4:
		return mod
	case 5:
		return rshift
	case 6:
		return or
	case 7:
		return and
	case 8:
		return nxor
	}
	return plus
}

//the tapelink is used to t 
type Tapelink struct {
        I int64
        J int64
        op func(z *big.Int, x *big.Int, y *big.Int) *big.Int
}

func Sha3Bin(data []byte) []byte {
        d := sha3.NewKeccak256()
        d.Write(data)
        return d.Sum(nil)
}

func BenchmarkSha3Bin(b *testing.B){
        for i:= 0; i < b.N; i++ {
                Sha3Bin([]byte(string(i)))
        }       
} 

//generates a tape of operations that is w*d long
func gen_tape(seed int64, w int, d int) []Tapelink {
	var X = big.NewInt(0)
	var Y = big.NewInt(0)
	Y.Exp(big.NewInt(2),big.NewInt(80), nil)
	var M = big.NewInt(0)
	var T []Tapelink
	for i := 0; i < w*d; i++{
		// add empty link to tape
		T = append(T, *new(Tapelink))

		// generate new entropy as needed
		if (int64(X.Cmp(Y)) == -1) {X.SetBytes(Sha3Bin([]byte(strconv.FormatInt(seed + int64(i), 10))))}

		// Pick random index I
		T[i].I = M.Mod(X,big.NewInt(int64(w))).Int64()
		M = big.NewInt(int64(w))
		X = X.Div(X,M)

		// Pick random index J
		mm := big.NewInt(M.Mod(X,big.NewInt(int64(w-1))).Int64() + int64(1) + T[i].I)
		T[i].J = M.Mod(mm, big.NewInt(int64(w))).Int64()
		M = big.NewInt(int64(w-1))
		X = X.Div(X,M)

		// Pick random operation
		T[i].op = GetOp(int(M.Mod(X, big.NewInt(int64(num_ops))).Int64()))
		M = big.NewInt(int64(num_ops))
		X = X.Div(X,M)	
	}
	return T
}

func BenchmarkGen_tape(b *testing.B){
        var X big.Int
        var s int64 
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
                b.StopTimer()
                X.SetBytes(Sha3Bin([]byte(string(i))))
                s = X.Int64()
                b.StartTimer()
		gen_tape(s, tape_width, tape_depth) 
	}

}

func gen_inputs(seed int64, w int) []big.Int {
	var A []big.Int
	for i := 0; i < w; i++ {
		A = append(A, *new(big.Int))
		if (i % 256 == 0) {
			A[i].SetBytes(Sha3Bin([]byte(strconv.FormatInt(seed + int64(i), 10))))
		} else {
			A[i].Lsh(&A[i-1], 1)
		}
	}
	return A
}

func BenchmarkGen_inputs(b *testing.B){
        var X big.Int
        var s int64
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
                b.StopTimer()
                X.SetBytes(Sha3Bin([]byte(string(i))))
                s = X.Int64()
                b.StartTimer()
		gen_inputs(s, tape_width)
	}
}

//this changes the inputs as it goes through a tape with d links
func run_tape(tape []Tapelink, inputs []big.Int, d int) {
	var X *big.Int
	X = big.NewInt(0)
	for i := 0; i < d; i++ {
		X = tape[i].op(X, &inputs[tape[i].I], &inputs[tape[i].J])
		inputs[tape[i].I].Set(X) 
	}
}
 
func BenchmarkRun_tape(b *testing.B){
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
	        T := gen_tape(int64(i), tape_width, tape_depth)
	        I := gen_inputs(int64(i), tape_width)
		b.StartTimer()
		run_tape(T,I,tape_width*tape_depth)
	}
}


//returns a 2^d - 1 length tape of operations (no I or J's in the Tapelinks)
func gen_tree(seed int64, d int) []Tapelink {
        M := big.NewInt(0) 			// dummy variable
	X := big.NewInt(0) 			// entropy variable
        Y := big.NewInt(0) 			// entropy buffer size 
	Y.Exp(big.NewInt(2),big.NewInt(80), nil)
        var T []Tapelink 			// the tree will be stored here

        for i := 0; i < int(math.Pow(2, float64(d))) - 1; i++{

                T = append(T, *new(Tapelink))

		//giving it more entropy, if X < 2^32
		if (X.Cmp(Y) == -1) {X.SetBytes(Sha3Bin([]byte(strconv.FormatInt(seed + int64(i),10))))}
              
		//filling the tape with random ops
		T[i].op = GetOp(int(M.Mod(X, big.NewInt(num_ops)).Int64()))
                M = big.NewInt(num_ops)
                X = X.Div(X,M)
        }
        return T
}

func BenchmarkGen_tree(b *testing.B) {
	var X big.Int
	var s int64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		X.SetBytes(Sha3Bin([]byte(string(i))))
		s = X.Int64()
		b.StartTimer()
		gen_tree(s, tree_depth)
	}
}

//there should be 2^d inputs and 2^d - 1 links in the tape. not complying is an unhandled exception
func run_tree(inputs []big.Int, tree []Tapelink, d int) *big.Int {
	X := big.NewInt(0)
	counter := 0
	for j := 0; j < d; j++ {
		for i := 0; i < int(math.Pow(2,float64(d - j - 1))); i++ {
			X = tree[counter].op(X, &inputs[2*i], &inputs[2*i + 1])		
			inputs[i].Set(X)
			counter += 1
		}
	}
	
	return &inputs[0]
}

func BenchmarkRun_tree(b *testing.B) {
	var X big.Int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
	        var inputs []big.Int
		for j := 0; j < tape_width; j++ {
        	        X.SetBytes(Sha3Bin([]byte(string(j + i*tape_width))))
                	inputs = append(inputs, X)
	        }
		tree := gen_tree(X.Int64(), tree_depth)
		b.StartTimer()
		run_tree(inputs, tree, tree_depth)
	}
}

func sha_cap(s []string, n int) []byte{
        var feed string
        feed = ""
        for i := 0; i < n; i++ {
                feed += s[i]
        }
        return Sha3Bin([]byte(feed))
}


func BenchmarkSha_cap(b *testing.B){
        var X big.Int
	var s []string
	b.ResetTimer()
       	for i := 0; i < b.N; i++ {
		b.StopTimer()
		X.SetBytes(Sha3Bin([]byte(string(i))))
		s = append(s, X.String())
		b.StartTimer()
                sha_cap(s,1)
        }
}

func main(){
	var seed int64
	var sample []big.Int


	seed = int64(13300331)

	for i := 0; i < sample_size; i++ {

       	 	Tape := gen_tape(seed, tape_width, tape_depth)
	        Tree := gen_tree(seed, tree_depth)
		
		seed += int64(i*i)
		if i%10 == 0 {
			fmt.Printf("i: %d\n",i)
		}
		I := gen_inputs(seed, tape_width)
		run_tape(Tape, I, tape_depth*tape_width)
		
		
		output := *run_tree(I, Tree, tree_depth)
		if output.Cmp(big.NewInt(0)) == 0 {
			fmt.Printf("We have a zero from the tree, form operation: \n")
			fmt.Println(Tree[len(Tree)-1])
			
		}
		
	
		var blockhashes []string
		blockhashes = append(blockhashes, output.String())
		
		sample = append(sample, output)
		H := sha_cap(blockhashes, 1)
	
		output.SetBytes(H)
		//sample = append(sample, output)
	}

	var collisions int = 0
	for i := 0; i < sample_size; i++ {
		//fmt.Println(sample[i])
		for j:=0; j < i; j++ {
			//if sample[i] == sample[j] { 
			if sample[i].Cmp(&sample[j]) == 0 {
				collisions += 1
				//fmt.Printf("collision on i, j: %d, %d", i, j)
				//fmt.Println(sample[i])
			}		
		}
	}
	fmt.Printf("number of outputs with same value: %d out of %d\n", (1+int(math.Pow(float64(1+8*collisions),0.5)))/2, sample_size)	
}
