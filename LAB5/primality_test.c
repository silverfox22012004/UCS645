#include "primality_test.h"
#include <math.h>

int evaluate_prime(int num) {
    if (num < 2)  return -num;
    if (num == 2) return num;
    if (num % 2 == 0) return -num;

    int upper = (int)sqrt((double)num);
    for (int divisor = 3; divisor <= upper; divisor += 2) {
        if (num % divisor == 0)
            return -num;
    }
    return num;
}
