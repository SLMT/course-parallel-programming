#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double RandFloat(double min, double max)
{
    double f = (double) rand() / RAND_MAX;
    return min + f * (max - min);
}

int main(int argc, char const *argv[]) {
        size_t i;

        size_t body_count = (size_t) atoi(argv[1]);
        FILE *out = fopen(argv[2], "w");

        double min = -1000.0, max = 1000.0;
        srand(time(NULL));

        fprintf(out, "%u\n", body_count);
        for (i = 0; i < body_count; i++)
                fprintf(out, "%lf %lf 0.0 0.0\n", RandFloat(min, max), RandFloat(min, max));

        return 0;
}
