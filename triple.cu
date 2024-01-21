#include <iostream>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/partition.h>

using namespace std;

// 2^64-1 - if you want to fix the overflow issues you'll have to somehow cap values that overflow this.
#define absmax 18446744073709551615

// This is generally a mouthful to write.
#define pr thrust::pair<uint32_t, uint32_t>


/* ------------------------------------------------------------------------------------------------------------------
 * 
 * TRIPLE.CU version 1.0 written 2024-01-20 =========================================================================
 * 
 * This is a program to find triples of k-gonal numbers that have the property D(n).
 * Authored by Christian Zhou-Zheng, with help from Sounak Bagchi.
 * Created for number theory research at Euler Circle under Simon Rubenstein-Salzedo.
 * 
 * USER GUIDE -------------------------------------------------------------------------------------------------------
 * 
 * Everything that should be altered during use is in the "PARAMETERS" section below.
 * Change the parameters k and n accordingly.
 * Change the parameters abound, bbound, and cbound to change the search space.
 *   - Practical bounds on each (you run out of memory) are 10^5 for abound and bbound, and 10^10 for cbound.
 * You can also change the number of blocks and threads per block for your GPU.
 * When running this from the command line, on Linux (which you have to be on to run this as far as I'm aware), you
 *   can put "time" in front of it to benchmark runtime.
 * 
 * IMPLEMENTATION NOTES ---------------------------------------------------------------------------------------------
 * 
 * The program is split into two parts: checkPair and findc. checkPair takes an a and a b and checks if they satisfy
 *   D(n). If they do, it outputs the pair (a,b) to a thrust::device_vector. This section is known to work.
 * findc takes the output of checkPair and checks if the pair (a,b) satisfies D(n) with a c within cbound. If it does,
 *   it outputs c to a thrust::device_vector. This section is known to have issues.
 * There's a bit of intermediate processing in main() because due to how CUDA multithreading works, you can't have a
 *   dynamically resized output like a vector (even if thrust::device_vector masquerades as one, you can only modify)
 *   it from host code). So I have a full array of size abound*bbound (which is what causes the memory overflow that
 *   locks abound and bbound at 10^5) which has one position for each potential pair, each of which only gets written
 *   to if necessary. This results in a lot of waste that I can't figure out how to reduce. I do use 
 *   thrust::stable_partition to remove all the pairs with zeros, creating a smaller thrust::device_vector that I can 
 *   pass to findc, but I don't know how to reduce the size in checkPair.
 * 
 * KNOWN BUGS/NOTES -------------------------------------------------------------------------------------------------
 * 
 * I haven't actually tested the program for n != 1. It should work regardless.
 * Very large values of c/cbound will cause overflow issues with the eventual output. This makes no sense to me
 *   but it is what it is, and I think the <= in the final "print to stdout" section in main() fixes it.
 * Again, 10^5 is the practical limit for abound and bbound, due to the reasons detailed above. This can be fixed.
 * 
 * TODOS ------------------------------------------------------------------------------------------------------------
 * 
 * Ensure the program works for n != 1.
 * Fix the overflow issues with values of c.
 * Allow for a larger search space on abound and bbound by reducing the size of the array written to. Look into
 *   stream compaction. From Lemon#3040 on Discord:
 *   "Youll have to implement your own resizable vector
 *    The two choices is to either allocate what you are certain to be enough, then push the pairs there (if you dont
 *      care about ordering then you can simply increment an atomic, it wont hurt performance(at least if you 
 *      coalesce them by doing warp vote count, then have leader do the atomic)), and if it wasnt enough, you will 
 *      have to reallocate and run it again
 *    The problem is that you cant really do allocations from gpu
 *    You can preallocate a large buffer and then write a custom allocator, but it has its own problems
 *    So yea, resizable vectors are a lie
 *    Just get the upper bounds on output size, then use an atomic increment" 
 *   In particular, see https://stackoverflow.com/questions/34059753/cuda-stream-compaction-algorithm.
 * 
 * ------------------------------------------------------------------------------------------------------------------
 * Below here, change your parameters.
// PARAMETERS ---------------------------------------------------------------------------------------------------- */
#define k 7             // DEFAULT k=3 5 7 9
#define n 9             // DEFAULT n=1 or 9 for heptagonals
#define abound 1000     // DEFAULT 1000
#define bbound 1000     // DEFAULT 1000
#define cbound 10000000 // DEFAULT 10000000
#define boxsize 200     // DEFAULT 200 LIMIT 1024
#define threadsize 1000 // DEFAULT 1000 LIMIT 1024


// SQUARE-DETERMINING -----------------------------------------------------------------------------------------------
// Thanks to Norbert Juffa for this section, https://forums.developer.nvidia.com/t/integer-square-root/198642.
// This is imported code to quickly determine the floored integer square root of an integer.
/*
  Copyright (c) 2021, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright 
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Fast multiplication of two unsigned 32-bit integers with 64-bit result.
__device__ unsigned long long int umul_wide (unsigned int a, unsigned int b) {
    unsigned long long int r;
    asm ("mul.wide.u32 %0,%1,%2;\n\t" : "=l"(r) : "r"(a), "r"(b));
    return r;
}

// Compute the integer square root of a 64-bit unsigned integer by multiplying it with its inverse square root
// (which can be computed much faster), then applying one Newton-Raphson iteration and flooring.
__device__ uint64_t isqrtll (uint64_t a) {
    uint64_t rem, arg;
    uint32_t b, r, s, scal;

    arg = a;
    // Normalize argument
    scal = __clzll (a) & ~1;
    a = a << scal;
    b = a >> 32;
    // Approximate rsqrt accurately. Make sure it's an underestimate!
    float fb, fr;
    fb = (float)b;
    asm ("rsqrt.approx.ftz.f32 %0,%1; \n\t" : "=f"(fr) : "f"(fb));
    r = (uint32_t) fmaf (1.407374884e14f, fr, -438.0f);
    // Compute sqrt(a) as a * rsqrt(a).
    s = __umulhi (r, b);
    // NR iteration combined with back multiply.
    s = s * 2;
    rem = a - umul_wide (s, s);
    r = __umulhi ((uint32_t)(rem >> 32) + 1, r);
    s = s + r;
    // Denormalize result.
    s = s >> (scal >> 1);
    // Make sure we get the floor correct; can be off by one to either side.
    rem = arg - umul_wide (s, s);
    if ((int64_t)rem < 0) s--;
    else if (rem >= ((uint64_t)s * 2 + 1)) s++;
    return (arg == 0) ? 0 : s;
}


/* ------------------------------------------------------------------------------------------------------------------
 * My code starts here. Don't touch getKGonal or checkPair, but change findc and mess with types as needed (long long
 * vs. uint64_t in particular). Todo: fix the overflow issues.
 * --------------------------------------------------------------------------------------------------------------- */
// General formula for the k-gonal numbers. Do not touch the logic (unless I did something REALLY dumb), but change the type as needed.
__device__ long long int getKGonal(long long a) { return (k % 2 == 1) ? a*((k-2)*a+(4-k))/2 : a*((k/2-1)*a-((k/2-1)-1))/2; }

// Check if a is a perfect square. Potential issues here.
__device__ bool isPerfectSquare(uint64_t a) { 
    uint64_t s = isqrtll(a);
    // Check whether the square of its integer square root is itself.
    return s*s == a; 
}

// Each block takes an a-value, then the threads within that block split up b.
__global__ void checkPair(pr *out) {
    int aidx = blockIdx.x+1;
    int bidx = threadIdx.x;
    for(int a = aidx; a <= abound; a += boxsize) {
        if(a==0) continue;

        // This is not very efficient! You lose a couple of threads per block (depending on parameters and search bounds).
        for(int b = a+bidx; b <= bbound; b += threadsize) {
            long long int aprime = getKGonal(a);
            long long int bprime = getKGonal(b);
            // Write the pair to the output array if it satisfies D(n).
            if(isPerfectSquare(aprime * bprime + n)) out[(a-1)*bbound + b] = thrust::pair(a,b);
        }
    }
}

// We have a and b already, so check c-values.
// Each block takes a pair (a, b) in in, and each thread takes a subset of the search space for c.
__global__ void findc(thrust::device_vector<pr> in, long long int *out) {
    for(int q = blockIdx.x; q <= in.size(); q += boxsize) {
        // Grab the pair - for some reason you can't oneline this.
        pr p = in[q];
        int a = p.first;
        int b = p.second;

        // The "if(x==0) continue;" statements are here to avoid issues with values equalling 0.
        // I don't know whether I still need them but I worry the program will fall apart if I remove them.
        if(a==0 || b==0) continue;
        long long int aprime = getKGonal(a);
        long long int bprime = getKGonal(b);
        for(long long int c = b+threadIdx.x; c <= cbound; c += threadsize) {
            if(c==0) continue;

            // The overflow issues probably kick in around here. This is where you should test for the overflow issues.
            long long int cprime = getKGonal(c);
            if(isPerfectSquare(aprime * cprime + n) && isPerfectSquare(cprime * bprime + n)) out[q] = c;
        }
    }
}

// Predicate for thrust::stable_partition to remove all pairs with zeros below.
struct is_nonzero {
    __host__ __device__
    bool operator()(const pr x) { return x.first != 0 && x.second != 0; }
};


// MAIN -------------------------------------------------------------------------------------------------------------
int main()
{
    // I wish there was an easier way to do this... Standard memory allocation.
    // I don't actually know if I need to instantiate an array on the host side.
    // Check that out if you try to fix the memory overflow issues.
    pr *out, *d_out;
    out = (pr *)malloc(abound*bbound*sizeof(pr));
    cudaMalloc((void **)&d_out, abound*bbound*sizeof(pr));
    cudaMemcpy(d_out, out, abound*bbound*sizeof(pr), cudaMemcpyHostToDevice);

    // Run the first part of the program.
    checkPair<<<boxsize, threadsize>>>(d_out);

    // Absolutely need to wait for that to finish running before continuing.
    cudaDeviceSynchronize();


    // THE CODE IS DEFINITELY GOOD UP TO HERE. Past here, uh, no promises.


    // Grab the pointer from the output from the last part and use it to create a thrust::device_vector with the values.
    thrust::device_ptr<pr> d_ptr(d_out);
    thrust::device_vector<pr> d_vec(d_ptr, d_ptr + abound*bbound);
    // Sorts the vector putting all nonzero pairs first and nonzero pairs last, and returns an iterator to the first zero.
    thrust::device_vector<pr>::iterator new_end = thrust::stable_partition(d_vec.begin(), d_vec.end(), is_nonzero());
    // Create a new thrust::device_vector with exactly the nonzero pairs.
    thrust::device_vector<pr> d_vec2(d_vec.begin(), new_end);


    // THE CODE IS VERY NEARLY DEFINITELY GOOD UP TO HERE.


    // Second verse, same as the first. Standard memory allocation, but do I need to allocate memory on the host?
    long long int *out2, *d_out2;
    out2 = (long long int *)malloc(d_vec2.size()*sizeof(long long int));
    cudaMalloc((void **)&d_out2, d_vec2.size()*sizeof(long long int));
    cudaMemcpy(d_out2, out2, d_vec2.size()*sizeof(long long int), cudaMemcpyHostToDevice);

    // Run the second part of the program.
    findc<<<boxsize, threadsize>>>(d_vec2, d_out2);

    // Absolutely need to wait for that to finish running before continuing.
    cudaDeviceSynchronize();

    // Copy the output back to the host since we don't need to fanagle with intermediate processing again.
    cudaMemcpy(out2, d_out2, d_vec2.size()*sizeof(long long int), cudaMemcpyDeviceToHost);

    // Print final result to stdout.
    for(int i = 0; i < d_vec2.size(); i++) {
        // I'm PRETTY sure this line combats the overflow (the less-than in particular) but I'm not 100% sure.
        if(out2[i] <= 0) continue;
        pr p = d_vec2[i];
        cout << p.first << " " << p.second << " " << out2[i] << std::endl;
    }

    return 0;
}