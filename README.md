# triple
### Version 1.0 | Released 2024-01-20

This is a program to find triples of k-gonal numbers that have the property D(n).

Authored by Christian Zhou-Zheng, with help from Sounak Bagchi.

Created for number theory research at Euler Circle under Simon Rubenstein-Salzedo.

## User Guide

- Everything that should be altered during use is in the "PARAMETERS" section below.

- Change the parameters k and n accordingly.

- Change the parameters abound, bbound, and cbound to change the search space.

  - Practical bounds on each (you run out of memory) are 10^5 for abound and bbound, and 10^10 for cbound.

- You can also change the number of blocks and threads per block for your GPU.

- When running this from the command line, on Linux (which you have to be on to run this as far as I'm aware), you can put "time" in front of it to benchmark runtime.

## Implementation Notes

- The program is split into two parts: checkPair and findc. checkPair takes an a and a b and checks if they satisfy D(n). If they do, it outputs the pair (a,b) to a thrust::device_vector. This section is known to work.

- findc takes the output of checkPair and checks if the pair (a,b) satisfies D(n) with a c within cbound. If it does, it outputs c to a thrust::device_vector. This section is known to have issues.

- There's a bit of intermediate processing in main() because due to how CUDA multithreading works, you can't have a dynamically resized output like a vector (even if thrust::device_vector masquerades as one, you can only modify it from host code). So I have a full array of size abound*bbound (which is what causes the memory overflow that   locks abound and bbound at 10^5) which has one position for each potential pair, each of which only gets written to if necessary. This results in a lot of waste that I can't figure out how to reduce. I do use thrust::stable_partition to remove all the pairs with zeros, creating a smaller thrust::device_vector that I can pass to findc, but I don't know how to reduce the size in checkPair.

## Known Bugs/Notes

- I haven't actually tested the program for n != 1. It should work regardless.

- Very large values of c/cbound will cause overflow issues with the eventual output. This makes no sense to me but it is what it is, and I think the <= in the final "print to stdout" section in main() fixes it.

- Again, 10^5 is the practical limit for abound and bbound, due to the reasons detailed above. This can be fixed.

## Todos

- Ensure the program works for n != 1.

- Fix the overflow issues with values of c.

- Allow for a larger search space on abound and bbound by reducing the size of the array written to. Look into stream compaction. In particular, see https://stackoverflow.com/questions/34059753/cuda-stream-compaction-algorithm.