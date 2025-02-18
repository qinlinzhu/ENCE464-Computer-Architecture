#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h> 
#include <unistd.h>
#include <sys/time.h>
#include <xmmintrin.h>  // SSE intrinsics
#include <emmintrin.h>  // SSE2 intrinsics
/**
* BUILDING :
* gcc -O3 -march=native -funroll-loops -fopenmp -o poisson poisson.c -pthread -g -pg
*
*/

#define NUM_THREADS 6
#define IND(x,y,z,n) ((((z) *n) + (y))*n + (x))

static bool debug = false;

pthread_mutex_t lock; 
typedef double MathFunc_t(double);

typedef struct
{
    int start;
    int end;
    float delta;
    double *source;
    double *curr;
    double *next;
    double n;
    int iterations;
	pthread_t thread;
} ThreadData;

typedef struct 
{
    double *curr;
    double *next;
} ArrayData;


pthread_barrier_t barrier;
// Function to perform integration using the trapezoid method in a thread


void *worker(void *args)
{

    ThreadData *data = (ThreadData *)args;
    double *next = data->next;
    double *curr = data->curr;
    int n = data->n;
    float delta = data->delta;
    double *source = data->source;

    for (int iter = 0; iter < data->iterations; iter++) {
        for (int z = data->start; z < data->end; z++) {
            // Handle boundaries (Neumann) for y = 0 and y = n-1, x = 0 and x = n-1
            // Boundary at y = 0 and x = 0

            next[IND(0,0,z,n)] = 
                1.0 / 6.0 * (2 * curr[IND(1,0,z,n)] // x+1 boundary (Neumann)
                + 2 * curr[IND(0,1,z,n)]            // y+1 boundary (Neumann)
                + curr[IND(0,0,z+1,n)]              // z+1 (interior point)
                + curr[IND(0,0,z-1,n)]              // z-1 (interior point)
                - delta * delta * source[IND(0,0,z,n)]);

            // Boundary at x = n-1, y = 0
            next[IND(n-1,0,z,n)] = 
                1.0 / 6.0 * (2 * curr[IND(n-2,0,z,n)]   // x-1 boundary (Neumann)
                + 2 * curr[IND(n-1,1,z,n)]             // y+1 boundary (Neumann)
                + curr[IND(n-1,0,z+1,n)]               // z+1 (interior point)
                + curr[IND(n-1,0,z-1,n)]               // z-1 (interior point)
                - delta * delta * source[IND(n-1,0,z,n)]);

            // Boundary at x = 0, y = n-1
            next[IND(0,n-1,z,n)] = 
                1.0 / 6.0 * (2 * curr[IND(1,n-1,z,n)]    // x+1 boundary (Neumann)
                + 2 * curr[IND(0,n-2,z,n)]              // y-1 boundary (Neumann)
                + curr[IND(0,n-1,z+1,n)]                // z+1 (interior point)
                + curr[IND(0,n-1,z-1,n)]                // z-1 (interior point)
                - delta * delta * source[IND(0,n-1,z,n)]);

            // Boundary at x = n-1, y = n-1
            next[IND(n-1,n-1,z,n)] = 
                1.0 / 6.0 * (2 * curr[IND(n-2,n-1,z,n)]  // x-1 boundary (Neumann)
                + 2 * curr[IND(n-1,n-2,z,n)]            // y-1 boundary (Neumann)
                + curr[IND(n-1,n-1,z+1,n)]              // z+1 (interior point)
                + curr[IND(n-1,n-1,z-1,n)]              // z-1 (interior point)
                - delta * delta * source[IND(n-1,n-1,z,n)]);


            // Internal points
            for (int x = 1; x < n-1; x++) {
                next[IND(x,0,z,n)] = 
                    1.0 / 6.0 * (curr[IND(x+1,0,z,n)]    // x+1
                    + curr[IND(x-1,0,z,n)]               // x-1
                    + 2 * curr[IND(x,1,z,n)]             // y+1 (Neumann)
                    + curr[IND(x,0,z+1,n)]               // z+1
                    + curr[IND(x,0,z-1,n)]               // z-1
                    - delta * delta * source[IND(x,0,z,n)]);

                next[IND(x,n-1,z,n)] = 
                    1.0 / 6.0 * (curr[IND(x+1,n-1,z,n)]  // x+1
                    + curr[IND(x-1,n-1,z,n)]             // x-1
                    + 2 * curr[IND(x,n-2,z,n)]           // y-1 (Neumann)
                    + curr[IND(x,n-1,z+1,n)]             // z+1
                    + curr[IND(x,n-1,z-1,n)]             // z-1
                    - delta * delta * source[IND(x,n-1,z,n)]);
            }

            for (int y = 1; y < n-1; y++) {
                next[IND(0,y,z,n)] = 
                    1.0 / 6.0 * (2 * curr[IND(1,y,z,n)]  // x+1 (Neumann)
                    + curr[IND(0,y+1,z,n)]               // y+1
                    + curr[IND(0,y-1,z,n)]               // y-1
                    + curr[IND(0,y,z+1,n)]               // z+1
                    + curr[IND(0,y,z-1,n)]               // z-1
                    - delta * delta * source[IND(0,y,z,n)]);

                next[IND(n-1,y,z,n)] = 
                    1.0 / 6.0 * (2 * curr[IND(n-2,y,z,n)]  // x-1 (Neumann)
                    + curr[IND(n-1,y+1,z,n)]              // y+1
                    + curr[IND(n-1,y-1,z,n)]              // y-1
                    + curr[IND(n-1,y,z+1,n)]              // z+1
                    + curr[IND(n-1,y,z-1,n)]              // z-1
                    - delta * delta * source[IND(n-1,y,z,n)]);
            

                for (int x = 1; x < n-1; x++) {
                    next[IND(x,y,z,n)] = 
                        1.0 / 6.0 * (curr[IND(x+1,y,z,n)]  // x+1
                        + curr[IND(x-1,y,z,n)]             // x-1
                        + curr[IND(x,y+1,z,n)]             // y+1
                        + curr[IND(x,y-1,z,n)]             // y-1
                        + curr[IND(x,y,z+1,n)]             // z+1
                        + curr[IND(x,y,z-1,n)]             // z-1
                        - delta * delta * source[IND(x,y,z,n)]);
                }
            }
        }
        
        // Swap the grids for the next iteration
        double *temp = curr;
        curr = next;
        next = temp;

        pthread_barrier_wait(&barrier);
    }
  
    return NULL;
}



int main(int argc, char **argv)
{
    
    int opt;
    int iterations = 10;
    int n = 5;
    float delta = 1;
    int x = -1;
    int y = -1;
    int z = -1;
    double amplitude = 1.0;
    int threads = NUM_THREADS;

    while  ((opt = getopt(argc, argv, "h:n:i:x:y:z:a:t:d:")) != -1) {
        switch(opt) {
            case 'h':
                printf("Usage: poisson [-n size] [-x source x-poisition] [-y source y-position] [-z source z-position] [-a source amplitude] [-i iterations] [-t threads] [-d] (for debug mode)\n");
                return EXIT_SUCCESS;
            case 'n':
                n = atoi(optarg);
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'x':
                x = atoi(optarg);
                break;
            case 'y':
                y = atoi(optarg);
                break;
            case 'z':
                z = atoi(optarg);
                break;
            case 'a':
                amplitude = atof(optarg);
                break;
            case 't':
                threads = atoi(optarg);
                break;
            case 'd':
                debug = true;
                break;
            default:
                fprintf(stderr, "Usage: poisson [-n size] [-x source x-poisition] [-y source y-position] [-z source z-position] [-a source amplitude]  [-i iterations] [-t threads] [-d] (for debug mode)\n");
                exit(EXIT_FAILURE);
        }
    }

    if (n % 2 == 0)
    {
        fprintf (stderr, "Error: n should be an odd number!\n");
        return EXIT_FAILURE;
    }
    // Coordinates
    if (x < 0 || x > n-1) x = n / 2;
    if (y < 0 || y > n-1) y = n / 2;
    if (z < 0 || z > n-1) z = n / 2;
    

    double *source = (double*)calloc(n * n * n, sizeof(double));
    if (source == NULL) {
        fprintf(stderr, "Error: failed to allocate source term (n=%i)\n", n);
        return EXIT_FAILURE;
    }

    double *curr = (double*)calloc(n * n * n, sizeof(double));
    double *next = (double*)calloc(n * n * n, sizeof(double));
    

// Pointers to the sections of the merged array

    if (curr == NULL || next == NULL) {
        fprintf(stderr, "Error: failed to allocate current/next state buffers\n");
        free(source);
        return EXIT_FAILURE;
    }
    ArrayData arr[2] = {curr, next};

    source[(z * n + y) * n + x] = amplitude;

    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            curr[IND(x,y,0,n)] = -1.0; // Bottom boundary
            next[IND(x,y,0,n)] = -1.0;
            curr[IND(x,y,n-1,n)] = 1.0; // Top boundary
            next[IND(x,y,n-1,n)] = 1.0;
        }
    }

    pthread_t threads_ids[threads];
    ThreadData threadData[threads];

   
    pthread_barrier_init(&barrier, NULL, threads);

    int range = n - 2; // The range to process (excluding boundaries)
    // struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < threads; i++) {
        threadData[i].start = 1 + (range * i) / threads;  // Start index for this thread
        threadData[i].end = 1 + (range * (i + 1)) / threads;    // End index for this thread
        threadData[i].delta = delta;
        threadData[i].source = source;
        threadData[i].curr = arr->curr;
        threadData[i].next = arr->next;
        threadData[i].n = n;
        threadData[i].iterations = iterations;  
        pthread_create(&threads_ids[i], NULL, &worker, &threadData[i]); 
    } 

    for (int i = 0; i < threads; i++) {
        pthread_join(threads_ids[i], NULL);
    }

    pthread_barrier_destroy(&barrier);
    

    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) { 
            printf("%0.5f ", curr[((n / 2) * n + y) * n + x]);
        }
        printf("\n");
    }

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    // printf("Time spent: %f seconds\n", time_spent);

    free(source);
    free(curr);
    free(next);
    return EXIT_SUCCESS;

}
