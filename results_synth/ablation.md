| Sequence                                 |   MOTA |   IDF1 |   IDSW |   Frag |   FP |   FN |   MT |   ML |   Rcll |   Prcn |
|:-----------------------------------------|-------:|-------:|-------:|-------:|-----:|-----:|-----:|-----:|-------:|-------:|
| SORT (IoU + Hungarian, no memory)        |   66.1 |   56.2 |     10 |     12 |  217 |  780 |    6 |    1 |   73.8 |   91   |
| + 30-frame lost-track buffer             |   64   |   56.8 |     10 |     15 |  258 |  801 |    5 |    1 |   73   |   89.4 |
| + low-score det. association (BYTE)      |   62.6 |   69.2 |      7 |     10 |  326 |  778 |    5 |    1 |   73.8 |   87.1 |
| + camera-motion compensation             |   62.6 |   69.2 |      7 |     10 |  326 |  778 |    5 |    1 |   73.8 |   87.1 |
| + appearance re-ID of lost tracks (full) |   63.9 |   75.4 |      7 |     14 |  318 |  749 |    5 |    1 |   74.8 |   87.5 |
| + gap interpolation (offline)            |   63.2 |   75.3 |      3 |      9 |  363 |  729 |    5 |    1 |   75.5 |   86.1 |
