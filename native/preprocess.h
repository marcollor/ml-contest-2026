#ifndef TEMPORAL_CNN_PREPROCESS_H
#define TEMPORAL_CNN_PREPROCESS_H

#include <math.h>

enum { CHANNELS = 4, LENGTH = 256 };

typedef struct {
    int left;
    int right;
} PartitionBounds;

typedef PartitionBounds (*Partition)(const double *, double *, int, double);

static inline PartitionBounds partition_tail(const double *input, double *output,
                                            int start, int count, double pivot,
                                            PartitionBounds bounds)
{
    for (int i = start; i < count; ++i) {
        const double value = input[i];
        output[bounds.left] = value;
        output[bounds.right] = value;
        bounds.left += value < pivot;
        bounds.right -= value > pivot;
    }
    return bounds;
}

static PartitionBounds partition_scalar(const double *input, double *output,
                                        int count, double pivot)
{
    return partition_tail(input, output, 0, count, pivot, (PartitionBounds){0, count - 1});
}

static double median(double values[], int count, Partition partition)
{
    double scratch[LENGTH];
    double *input = values;
    double *output = scratch;
    int rank = count / 2;
    const int even = count % 2 == 0;
    double lower_bound = -INFINITY;

    for (;;) {
        const double pivot = input[count / 2];
        /* Keep smaller and larger values at opposite ends and discard ties. */
        const PartitionBounds bounds = partition(input, output, count, pivot);
        const int left = bounds.left;
        const int right = bounds.right;

        double *next = output;
        if (rank < left) {
            count = left;
        } else if (rank > right) {
            const int start = right + 1;
            lower_bound = pivot;
            next += start;
            count -= start;
            rank -= start;
        } else {
            if (!even || rank > left)
                return pivot;
            /* The lower middle value is immediately before the pivot's ties. */
            for (int i = 0; i < left; ++i)
                if (output[i] > lower_bound)
                    lower_bound = output[i];
            return (lower_bound + pivot) * 0.5;
        }
        output = input;
        input = next;
    }
}

static double clip(double value, double lower, double upper)
{
    if (value < lower)
        return lower;
    if (value > upper)
        return upper;
    return value;
}

/* Match src/normalize_data.py, including endpoint filling and float64 reductions. */
static void preprocess_channel(const float raw[LENGTH], float normalized[LENGTH],
                               Partition partition)
{
    const double epsilon = 1e-6;
    int positions[LENGTH];
    double values[LENGTH], scratch[LENGTH], imputed[LENGTH];
    int count = 0;

    int all_finite = 1;
    for (int i = 0; i < LENGTH; ++i)
        all_finite &= isfinite(raw[i]) != 0;
    if (all_finite) {
        for (int i = 0; i < LENGTH; ++i)
            values[i] = scratch[i] = raw[i];
        count = LENGTH;
    } else {
        for (int i = 0; i < LENGTH; ++i) {
            if (isfinite(raw[i])) {
                positions[count] = i;
                values[count] = scratch[count] = raw[i];
                ++count;
            }
        }
    }
    if (count == 0) {
        for (int i = 0; i < LENGTH; ++i)
            normalized[i] = 0.0f;
        return;
    }

    const double center = median(scratch, count, partition);
    for (int i = 0; i < count; ++i)
        scratch[i] = fabs(values[i] - center);
    const double bound = 6.0 * fmax(1.4826 * median(scratch, count, partition), sqrt(epsilon));
    if (all_finite) {
        for (int i = 0; i < LENGTH; ++i)
            imputed[i] = clip(values[i], center - bound, center + bound);
    } else {
        for (int i = 0; i < count; ++i)
            values[i] = clip(values[i], center - bound, center + bound);
        for (int i = 0; i <= positions[0]; ++i)
            imputed[i] = values[0];
        for (int j = 1; j < count; ++j) {
            const int left = positions[j - 1];
            const int right = positions[j];
            if (right - left > 1) {
                const double slope = (values[j] - values[j - 1]) / (right - left);
                for (int i = left + 1; i < right; ++i)
                    imputed[i] = values[j - 1] + slope * (i - left);
            }
            imputed[right] = values[j];
        }
        for (int i = positions[count - 1] + 1; i < LENGTH; ++i)
            imputed[i] = values[count - 1];
    }

    /* Independent accumulators hide add latency while retaining float64 precision. */
    double sums[4] = {0};
    for (int i = 0; i < LENGTH; i += 4)
        for (int lane = 0; lane < 4; ++lane)
            sums[lane] += imputed[i + lane];
    const double mean = ((sums[0] + sums[1]) + (sums[2] + sums[3])) / LENGTH;
    double squares[4] = {0};
    for (int i = 0; i < LENGTH; i += 4)
        for (int lane = 0; lane < 4; ++lane) {
            const double centered = imputed[i + lane] - mean;
            imputed[i + lane] = centered;
            squares[lane] += centered * centered;
        }
    const double sum_squares = (squares[0] + squares[1]) + (squares[2] + squares[3]);
    const double inverse_scale = 1.0 / sqrt(sum_squares / LENGTH + epsilon);
    for (int i = 0; i < LENGTH; ++i)
        normalized[i] = (float)clip(imputed[i] * inverse_scale, -10.0, 10.0);
}

#endif
