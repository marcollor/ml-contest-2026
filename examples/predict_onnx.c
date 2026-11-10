/* Linux/WSL example. Run: ./predict_onnx model.onnx < sample.txt
 * Input: 4 channels of 256 whitespace-separated floats, channel by channel.
 * Use nan for missing values. Output: one class ID (0 through 7).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <onnxruntime_c_api.h>

enum { CHANNELS = 4, LENGTH = 256, CLASSES = 8 };

static int compare_double(const void *a, const void *b)
{
    const double left = *(const double *)a;
    const double right = *(const double *)b;
    return (left > right) - (left < right);
}

static double median(double values[], int count)
{
    qsort(values, count, sizeof(values[0]), compare_double);
    const int middle = count / 2;
    return count % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

/* Match the preprocessing in src/normalize_data.py. */
static void preprocess_channel(float channel[LENGTH])
{
    const double epsilon = 1e-6;
    int positions[LENGTH];
    double values[LENGTH], scratch[LENGTH], imputed[LENGTH];
    int count = 0;

    for (int i = 0; i < LENGTH; ++i) {
        if (isfinite(channel[i])) {
            positions[count] = i;
            values[count] = scratch[count] = channel[i];
            ++count;
        }
    }
    if (count == 0) {
        for (int i = 0; i < LENGTH; ++i)
            channel[i] = 0.0f;
        return;
    }

    const double center = median(scratch, count);
    for (int i = 0; i < count; ++i)
        scratch[i] = fabs(values[i] - center);
    const double bound = 6.0 * fmax(1.4826 * median(scratch, count), sqrt(epsilon));
    for (int i = 0; i < count; ++i)
        values[i] = fmin(fmax(values[i], center - bound), center + bound);

    for (int i = 0; i <= positions[0]; ++i)
        imputed[i] = values[0];
    for (int j = 1; j < count; ++j) {
        const int left = positions[j - 1];
        const int right = positions[j];
        const double slope = (values[j] - values[j - 1]) / (right - left);
        for (int i = left + 1; i < right; ++i)
            imputed[i] = values[j - 1] + slope * (i - left);
        imputed[right] = values[j];
    }
    for (int i = positions[count - 1] + 1; i < LENGTH; ++i)
        imputed[i] = values[count - 1];

    double sum = 0.0;
    for (int i = 0; i < LENGTH; ++i)
        sum += imputed[i];
    const double mean = sum / LENGTH;
    double sum_squares = 0.0;
    for (int i = 0; i < LENGTH; ++i) {
        imputed[i] -= mean;
        sum_squares += imputed[i] * imputed[i];
    }
    const double scale = sqrt(sum_squares / LENGTH + epsilon);
    for (int i = 0; i < LENGTH; ++i)
        channel[i] = (float)fmin(fmax(imputed[i] / scale, -10.0), 10.0);
}

#define CHECK_ORT(call) do { \
    status = (call); \
    if (status != NULL) \
        goto cleanup; \
} while (0)

static int predict(const char *model_path, float series[CHANNELS][LENGTH])
{
    const OrtApi *ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (ort == NULL) {
        fputs("The ONNX Runtime library does not support this header's API version.\n", stderr);
        return -1;
    }

    OrtStatus *status = NULL;
    OrtEnv *env = NULL;
    OrtSessionOptions *options = NULL;
    OrtSession *session = NULL;
    OrtMemoryInfo *memory = NULL;
    OrtValue *input = NULL;
    OrtValue *outputs[] = {NULL};
    const char *input_names[] = {"series"};
    const char *output_names[] = {"logits"};
    const int64_t input_shape[] = {1, CHANNELS, LENGTH};
    const int64_t output_shape[] = {1, CLASSES};
    float logits[CLASSES];
    int prediction = -1;

    CHECK_ORT(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "predict", &env));
    CHECK_ORT(ort->CreateSessionOptions(&options));
    CHECK_ORT(ort->SetIntraOpNumThreads(options, 1));
    CHECK_ORT(ort->SetInterOpNumThreads(options, 1));
    CHECK_ORT(ort->CreateSession(env, model_path, options, &session));
    CHECK_ORT(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory));
    CHECK_ORT(ort->CreateTensorWithDataAsOrtValue(
        memory, series, sizeof(float[CHANNELS][LENGTH]), input_shape, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));
    CHECK_ORT(ort->CreateTensorWithDataAsOrtValue(
        memory, logits, sizeof(logits), output_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &outputs[0]));

    const OrtValue *inputs[] = {input};
    CHECK_ORT(ort->Run(session, NULL, input_names, inputs, 1, output_names, 1, outputs));
    prediction = 0;
    for (int i = 1; i < CLASSES; ++i) {
        if (logits[i] > logits[prediction])
            prediction = i;
    }

cleanup:
    if (status != NULL) {
        fprintf(stderr, "ONNX Runtime: %s\n", ort->GetErrorMessage(status));
        ort->ReleaseStatus(status);
    }
    ort->ReleaseValue(outputs[0]);
    ort->ReleaseValue(input);
    ort->ReleaseMemoryInfo(memory);
    ort->ReleaseSession(session);
    ort->ReleaseSessionOptions(options);
    ort->ReleaseEnv(env);
    return prediction;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s model.onnx < sample.txt\n", argv[0]);
        return EXIT_FAILURE;
    }

    float series[CHANNELS][LENGTH];
    for (int channel = 0; channel < CHANNELS; ++channel) {
        for (int i = 0; i < LENGTH; ++i) {
            if (scanf("%f", &series[channel][i]) != 1) {
                fprintf(stderr, "Expected a float at channel %d, index %d.\n", channel, i);
                return EXIT_FAILURE;
            }
        }
        preprocess_channel(series[channel]);
    }

    const int prediction = predict(argv[1], series);
    if (prediction < 0)
        return EXIT_FAILURE;
    printf("%d\n", prediction);
    return EXIT_SUCCESS;
}
