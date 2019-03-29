# wana
For EOS testing.

## Referance webcite
1. https://webassembly.org/
2. https://github.com/WebAssembly/
3. https://developer.mozilla.org/en-US/docs/WebAssembly

## Sections of WASM
1. user 0
2. type 1
3. functionDeclarations 3
4. table 4
5. memory 5
6. global 6
7. export 7
8. start 8
9. elem 9
10. functionDefinitions 10
11. data 11

## Types and its char in package 'struct'
1. @: native order, size & alignment (default)
2. =: native order, std. size & alignment
3. <: little-endian, std. size & alignment
4. \>: big-endian, std. size & alignment
5. !: same as >
6. x: pad byte
7. c: char
8. b: signed byte
9. B: unsigned byte
10. ?: boolean
11. h: short
12. H: unsigned short
13. i: int
14. I: unsigned int
15. l: long
16. L: unsigned long
17. f: float
18. d: double
19. s: string (array of char)
20. q: long long
21. Q: unsigned long long