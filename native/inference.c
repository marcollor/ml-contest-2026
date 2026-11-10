/* Specialized CPU inference for model/architecture.py. */
#include "preprocess.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define X86_KERNELS 1
#else
#define X86_KERNELS 0
#endif

enum { FEATURES = 32, CLASSES = 8, WEIGHT_FLOATS = 7064 };

typedef struct {
    float stem[32][4][7];
    float stem_bias[32];
    float depthwise[4][32][5];
    float depthwise_bias[4][32];
    float pointwise[4][32 * 32];
    float pointwise_bias[4][32];
    float relations[30][16];
    float relations_bias[16];
    float classifier[80][8];
    float classifier_bias[8];
} Weights;

_Static_assert(sizeof(float) == 4, "32-bit floats required");
_Static_assert(sizeof(Weights) == WEIGHT_FLOATS * sizeof(float), "Unexpected weight layout");

typedef void (*Pointwise)(const float *, const float *, float *, const float *,
                          const float *, int);

typedef struct Engine {
    Weights weights;
    Pointwise pointwise;
    Partition partition;
    int vectorized;
} Engine;

static float relu(float value)
{
    return value > 0.0f ? value : 0.0f;
}

static void pointwise_scalar(const float *restrict input, const float *restrict residual,
                             float *restrict output, const float *restrict weights,
                             const float *restrict bias, int length)
{
    for (int channel = 0; channel < FEATURES; ++channel) {
        float *row = output + channel * length;
        memset(row, 0, length * sizeof(float));
        for (int source = 0; source < FEATURES; ++source) {
            const int index = ((channel / 4) * FEATURES + source) * 4 + channel % 4;
            const float weight = weights[index];
            for (int t = 0; t < length; ++t)
                row[t] += input[source * length + t] * weight;
        }
        for (int t = 0; t < length; ++t)
            row[t] = relu((row[t] + bias[channel]) + residual[channel * length + t]);
    }
}

#if X86_KERNELS
/* Pack selected double lanes into the low end, in their original order. */
static const struct {
    int32_t indices[8];
    int count;
} partition_lanes[16] = {
    {{0, 1, 0, 1, 0, 1, 0, 1}, 0},
    {{0, 1, 0, 1, 0, 1, 0, 1}, 1},
    {{2, 3, 0, 1, 0, 1, 0, 1}, 1},
    {{0, 1, 2, 3, 0, 1, 0, 1}, 2},
    {{4, 5, 0, 1, 0, 1, 0, 1}, 1},
    {{0, 1, 4, 5, 0, 1, 0, 1}, 2},
    {{2, 3, 4, 5, 0, 1, 0, 1}, 2},
    {{0, 1, 2, 3, 4, 5, 0, 1}, 3},
    {{6, 7, 0, 1, 0, 1, 0, 1}, 1},
    {{0, 1, 6, 7, 0, 1, 0, 1}, 2},
    {{2, 3, 6, 7, 0, 1, 0, 1}, 2},
    {{0, 1, 2, 3, 6, 7, 0, 1}, 3},
    {{4, 5, 6, 7, 0, 1, 0, 1}, 2},
    {{0, 1, 4, 5, 6, 7, 0, 1}, 3},
    {{2, 3, 4, 5, 6, 7, 0, 1}, 3},
    {{0, 1, 2, 3, 4, 5, 6, 7}, 4},
};

__attribute__((target("avx2")))
static PartitionBounds partition_avx2(const double *input, double *output,
                                      int count, double pivot)
{
    const __m256d pivot_vector = _mm256_set1_pd(pivot);
    PartitionBounds bounds = {0, count - 1};
    int i = 0;
    /* Two full stores fit in the unused interval while eight items remain. */
    for (; i + 8 <= count; i += 4) {
        const __m256d values = _mm256_loadu_pd(input + i);
        const int lower = _mm256_movemask_pd(_mm256_cmp_pd(values, pivot_vector, _CMP_LT_OQ));
        const int upper = _mm256_movemask_pd(_mm256_cmp_pd(values, pivot_vector, _CMP_GT_OQ));
        const __m256i bits = _mm256_castpd_si256(values);
        const __m256i lower_order = _mm256_loadu_si256((const __m256i *)partition_lanes[lower].indices);
        const __m256i upper_order = _mm256_loadu_si256((const __m256i *)partition_lanes[upper].indices);
        const __m256i packed_lower = _mm256_permutevar8x32_epi32(bits, lower_order);
        const __m256i packed_upper = _mm256_permutevar8x32_epi32(bits, upper_order);
        _mm256_storeu_pd(output + bounds.left, _mm256_castsi256_pd(packed_lower));
        _mm256_storeu_pd(output + bounds.right - 3,
                         _mm256_permute4x64_pd(_mm256_castsi256_pd(packed_upper), _MM_SHUFFLE(0, 1, 2, 3)));
        bounds.left += partition_lanes[lower].count;
        bounds.right -= partition_lanes[upper].count;
    }
    return partition_tail(input, output, i, count, pivot, bounds);
}

__attribute__((target("avx2,fma")))
static void pointwise_avx2(const float *input, const float *residual, float *output,
                           const float *weights, const float *bias, int length)
{
    for (int channel = 0; channel < FEATURES; channel += 4) {
        for (int t = 0; t < length; t += 16) {
            __m256 a = _mm256_setzero_ps();
            __m256 b = a, c = a, d = a;
            __m256 e = a, f = a, g = a, h = a;
            for (int source = 0; source < FEATURES; ++source) {
                const __m256 x = _mm256_loadu_ps(input + source * length + t);
                const __m256 y = _mm256_loadu_ps(input + source * length + t + 8);
                const float *w = weights + ((channel / 4) * FEATURES + source) * 4;
                const __m256 w0 = _mm256_broadcast_ss(w);
                const __m256 w1 = _mm256_broadcast_ss(w + 1);
                const __m256 w2 = _mm256_broadcast_ss(w + 2);
                const __m256 w3 = _mm256_broadcast_ss(w + 3);
                a = _mm256_fmadd_ps(x, w0, a);
                b = _mm256_fmadd_ps(x, w1, b);
                c = _mm256_fmadd_ps(x, w2, c);
                d = _mm256_fmadd_ps(x, w3, d);
                e = _mm256_fmadd_ps(y, w0, e);
                f = _mm256_fmadd_ps(y, w1, f);
                g = _mm256_fmadd_ps(y, w2, g);
                h = _mm256_fmadd_ps(y, w3, h);
            }
            const __m256 first[4] = {a, b, c, d};
            const __m256 second[4] = {e, f, g, h};
            for (int j = 0; j < 4; ++j) {
                const int index = (channel + j) * length + t;
                const __m256 offset = _mm256_broadcast_ss(bias + channel + j);
                const __m256 x = _mm256_add_ps(_mm256_add_ps(first[j], offset),
                                               _mm256_loadu_ps(residual + index));
                const __m256 y = _mm256_add_ps(_mm256_add_ps(second[j], offset),
                                               _mm256_loadu_ps(residual + index + 8));
                _mm256_storeu_ps(output + index, _mm256_max_ps(x, _mm256_setzero_ps()));
                _mm256_storeu_ps(output + index + 8, _mm256_max_ps(y, _mm256_setzero_ps()));
            }
        }
    }
}
#endif

int engine_has_avx2(void)
{
#if X86_KERNELS
    __builtin_cpu_init();
    return !!(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"));
#else
    return 0;
#endif
}

/* mode is -1 for automatic dispatch, 0 for scalar, or 1 for AVX2/FMA. */
Engine *engine_create(const float *weights, size_t count, int mode)
{
    if (!weights || count != WEIGHT_FLOATS || mode < -1 || mode > 1)
        return NULL;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(weights[i]))
            return NULL;
    const int supported = engine_has_avx2();
    if (mode == 1 && !supported)
        return NULL;
    Engine *engine = malloc(sizeof(*engine));
    if (!engine)
        return NULL;
    memcpy(&engine->weights, weights, sizeof(engine->weights));
    engine->pointwise = pointwise_scalar;
    engine->partition = partition_scalar;
    engine->vectorized = 0;
#if X86_KERNELS
    if (supported && mode != 0) {
        engine->pointwise = pointwise_avx2;
        engine->partition = partition_avx2;
        engine->vectorized = 1;
    }
#endif
    return engine;
}

void engine_destroy(Engine *engine)
{
    free(engine);
}

const char *engine_mode(const Engine *engine)
{
    return engine->vectorized ? "avx2_fma" : "scalar";
}

static void preprocess(const float *raw, float *output, Partition partition)
{
    for (int channel = 0; channel < CHANNELS; ++channel)
        preprocess_channel(raw + channel * LENGTH, output + channel * LENGTH, partition);
}

int engine_preprocess(const float *raw, float *output)
{
    if (!raw || !output)
        return -1;
    Partition partition = partition_scalar;
#if X86_KERNELS
    if (engine_has_avx2())
        partition = partition_avx2;
#endif
    preprocess(raw, output, partition);
    return 0;
}

static void stem(const float *input, float *restrict output, const Weights *restrict weights)
{
    float padded[4][262] = {{0}};
    for (int channel = 0; channel < CHANNELS; ++channel)
        memcpy(padded[channel] + 3, input + channel * LENGTH, LENGTH * sizeof(float));
    for (int channel = 0; channel < FEATURES; ++channel) {
        float *row = output + channel * LENGTH;
        memset(row, 0, LENGTH * sizeof(float));
        for (int source = 0; source < CHANNELS; ++source)
            for (int k = 0; k < 7; ++k) {
                const float weight = weights->stem[channel][source][k];
                for (int t = 0; t < LENGTH; ++t)
                    row[t] += padded[source][t + k] * weight;
            }
        for (int t = 0; t < LENGTH; ++t)
            row[t] = relu(row[t] + weights->stem_bias[channel]);
    }
}

#if X86_KERNELS
__attribute__((target("avx2,fma")))
static void stem_avx2(const float *input, float *output, const Weights *weights)
{
    float padded[CHANNELS][LENGTH + 6] = {{0}};
    for (int channel = 0; channel < CHANNELS; ++channel)
        memcpy(padded[channel] + 3, input + channel * LENGTH, LENGTH * sizeof(float));

    const __m256 zero = _mm256_setzero_ps();
    for (int channel = 0; channel < FEATURES; channel += 2) {
        const __m256 first_bias = _mm256_set1_ps(weights->stem_bias[channel]);
        const __m256 second_bias = _mm256_set1_ps(weights->stem_bias[channel + 1]);
        float *first_row = output + channel * LENGTH;
        float *second_row = first_row + LENGTH;
        for (int t = 0; t < LENGTH; t += 16) {
            __m256 first_low = zero, first_high = zero;
            __m256 second_low = zero, second_high = zero;
            for (int source = 0; source < CHANNELS; ++source) {
                for (int k = 0; k < 7; ++k) {
                    const __m256 first_weight = _mm256_set1_ps(weights->stem[channel][source][k]);
                    const __m256 second_weight = _mm256_set1_ps(weights->stem[channel + 1][source][k]);
                    const float *samples = padded[source] + t + k;
                    const __m256 low = _mm256_loadu_ps(samples);
                    const __m256 high = _mm256_loadu_ps(samples + 8);
                    first_low = _mm256_fmadd_ps(low, first_weight, first_low);
                    first_high = _mm256_fmadd_ps(high, first_weight, first_high);
                    second_low = _mm256_fmadd_ps(low, second_weight, second_low);
                    second_high = _mm256_fmadd_ps(high, second_weight, second_high);
                }
            }
            _mm256_storeu_ps(first_row + t, _mm256_max_ps(_mm256_add_ps(first_low, first_bias), zero));
            _mm256_storeu_ps(first_row + t + 8, _mm256_max_ps(_mm256_add_ps(first_high, first_bias), zero));
            _mm256_storeu_ps(second_row + t, _mm256_max_ps(_mm256_add_ps(second_low, second_bias), zero));
            _mm256_storeu_ps(second_row + t + 8, _mm256_max_ps(_mm256_add_ps(second_high, second_bias), zero));
        }
    }
}

__attribute__((target("avx2,fma")))
static void depthwise_avx2(const float *input, float *output, const Weights *weights,
                           int length, int block, int dilation)
{
    const __m256 zero = _mm256_setzero_ps();
    for (int channel = 0; channel < FEATURES; ++channel) {
        float padded[160] = {0};
        memcpy(padded + 2 * dilation, input + channel * length, length * sizeof(float));
        const float *weight = weights->depthwise[block][channel];
        __m256 taps[5];
        for (int k = 0; k < 5; ++k)
            taps[k] = _mm256_set1_ps(weight[k]);
        const __m256 bias = _mm256_set1_ps(weights->depthwise_bias[block][channel]);
        float *row = output + channel * length;
        for (int t = 0; t < length; t += 32) {
            /* Interleave four FMA chains without changing the tap order. */
            __m256 sums[4];
            for (int lane = 0; lane < 4; ++lane)
                sums[lane] = _mm256_mul_ps(_mm256_loadu_ps(padded + t + 8 * lane), taps[0]);
            for (int k = 1; k < 5; ++k) {
                const float *samples = padded + t + k * dilation;
                for (int lane = 0; lane < 4; ++lane)
                    sums[lane] = _mm256_fmadd_ps(_mm256_loadu_ps(samples + 8 * lane),
                                                taps[k], sums[lane]);
            }
            for (int lane = 0; lane < 4; ++lane)
                _mm256_storeu_ps(row + t + 8 * lane,
                                 _mm256_max_ps(_mm256_add_ps(sums[lane], bias), zero));
        }
    }
}
#endif

static void downsample(const float *input, float *output, int length)
{
    for (int channel = 0; channel < FEATURES; ++channel) {
        const float *row = input + channel * length;
        float *result = output + channel * (length / 2);
        result[0] = (row[0] * 0.25f + row[0] * 0.5f) + row[1] * 0.25f;
        for (int t = 1; t < length / 2; ++t) {
            const float left = row[2 * t - 1];
            const float center = row[2 * t];
            const float right = row[2 * t + 1];
            result[t] = (left * 0.25f + center * 0.5f) + right * 0.25f;
        }
    }
}

static void depthwise_scalar(const float *input, float *output, const Weights *weights,
                             int length, int block, int dilation)
{
    for (int channel = 0; channel < FEATURES; ++channel) {
        float padded[160] = {0};
        float *row = output + channel * length;
        memcpy(padded + 2 * dilation, input + channel * length, length * sizeof(float));
        memset(row, 0, length * sizeof(float));
        for (int k = 0; k < 5; ++k) {
            const float weight = weights->depthwise[block][channel][k];
            for (int t = 0; t < length; ++t)
                row[t] += padded[t + k * dilation] * weight;
        }
        for (int t = 0; t < length; ++t)
            row[t] = relu(row[t] + weights->depthwise_bias[block][channel]);
    }
}

static void residual(const Engine *engine, const float *input, float *output,
                     int length, int block, int dilation)
{
    float filtered[32 * 128];
#if X86_KERNELS
    if (engine->vectorized)
        depthwise_avx2(input, filtered, &engine->weights, length, block, dilation);
    else
#endif
        depthwise_scalar(input, filtered, &engine->weights, length, block, dilation);
    engine->pointwise(filtered, input, output, engine->weights.pointwise[block],
                      engine->weights.pointwise_bias[block], length);
}

static float relation(const float *left, const float *right, int length)
{
    float partial[8] = {0};
    int t = 0;
    for (; t + 8 <= length; t += 8)
        for (int lane = 0; lane < 8; ++lane)
            partial[lane] += left[t + lane] * right[t + lane];
    float sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane)
        sum += partial[lane];
    for (; t < length; ++t)
        sum += left[t] * right[t];
    return sum / length;
}

int engine_logits(const Engine *engine, const float *raw, float logits[CLASSES])
{
    if (!engine || !raw || !logits)
        return -1;
    float normalized[4 * 256];
    float a[32 * 256], b[32 * 128], pooled[80], relations[30];
    preprocess(raw, normalized, engine->partition);
#if X86_KERNELS
    if (engine->vectorized)
        stem_avx2(normalized, a, &engine->weights);
    else
#endif
        stem(normalized, a, &engine->weights);
    downsample(a, b, 256);
    residual(engine, b, a, 128, 0, 1);
    downsample(a, b, 128);
    residual(engine, b, a, 64, 1, 2);
    downsample(a, b, 64);
    residual(engine, b, a, 32, 2, 4);
    residual(engine, a, b, 32, 3, 8);

    for (int channel = 0; channel < FEATURES; ++channel) {
        const float *row = b + channel * 32;
        float sum = 0.0f;
        for (int t = 0; t < 32; ++t)
            sum += row[t];
        const float mean = sum / 32.0f;
        float variance = 0.0f;
        for (int t = 0; t < 32; ++t) {
            const float centered = row[t] - mean;
            variance += centered * centered;
        }
        pooled[channel] = mean;
        pooled[32 + channel] = sqrtf(variance / 32.0f + 1e-6f);
    }
    int index = 0;
    const int cross_lags[] = {0, 2, 8};
    for (int lag_index = 0; lag_index < 3; ++lag_index) {
        const int lag = cross_lags[lag_index];
        for (int left = 0; left < CHANNELS; ++left)
            for (int right = left + 1; right < CHANNELS; ++right)
                relations[index++] = relation(normalized + left * LENGTH,
                                               normalized + right * LENGTH + lag, LENGTH - lag);
    }
    const int self_lags[] = {1, 4, 16};
    for (int lag_index = 0; lag_index < 3; ++lag_index) {
        const int lag = self_lags[lag_index];
        for (int channel = 0; channel < CHANNELS; ++channel)
            relations[index++] = relation(normalized + channel * LENGTH,
                                           normalized + channel * LENGTH + lag, LENGTH - lag);
    }
    float relation_sums[16] = {0};
    for (int source = 0; source < 30; ++source)
        for (int output = 0; output < 16; ++output)
            relation_sums[output] += relations[source] * engine->weights.relations[source][output];
    for (int output = 0; output < 16; ++output)
        pooled[64 + output] = relu(relation_sums[output] + engine->weights.relations_bias[output]);

    float class_sums[CLASSES] = {0};
    for (int source = 0; source < 80; ++source)
        for (int output = 0; output < CLASSES; ++output)
            class_sums[output] += pooled[source] * engine->weights.classifier[source][output];
    for (int output = 0; output < CLASSES; ++output)
        logits[output] = class_sums[output] + engine->weights.classifier_bias[output];
    return 0;
}

int engine_predict(const Engine *engine, const float *raw)
{
    float logits[CLASSES];
    if (engine_logits(engine, raw, logits))
        return -1;
    int prediction = 0;
    for (int i = 1; i < CLASSES; ++i)
        if (logits[i] > logits[prediction])
            prediction = i;
    return prediction;
}
