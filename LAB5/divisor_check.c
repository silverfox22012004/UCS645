#include "divisor_check.h"

int check_perfect_number(int val) {
    if (val < 2) return -val;

    int divisor_sum = 1;  /* 1 is always a divisor */
    for (int d = 2; d * d <= val; d++) {
        if (val % d == 0) {
            divisor_sum += d;
            if (d != val / d) {
                divisor_sum += val / d;
            }
        }
    }

    if (divisor_sum == val)
        return val;
    else
        return -val;
}
