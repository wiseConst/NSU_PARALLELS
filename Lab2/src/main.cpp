#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <omp.h>

#include "CoreUtils.h"

static constexpr uint32_t N = 50;
static constexpr uint32_t s_VEC_SIZE = N * N;
static constexpr uint32_t s_MAT_SIZE = s_VEC_SIZE * s_VEC_SIZE;

static uint32_t s_WORKER_THREADS_CURRENT = 1;
static constexpr uint32_t s_WORKER_THREADS = 16;

#define STEPS_COUNT 1
#define VAR_1 0
#define VAR_2 1
#define OUTPUT_INTO_FILE 0
#define SCHEDULING 0
#define SCHEDULE_TYPE dynamic // static, dynamic, guided
#define CHUNK_SIZE 8

float CalculateNorm(const std::vector<float>& vec)
{
    const float sumOfSquares = std::accumulate(vec.begin(), vec.end(), 0.F,
                                               [](const float acc, const float val)-> float
                                               {
                                                   return acc + val * val;
                                               });

    return std::sqrt(sumOfSquares);
}

std::vector<float> SolveLinearEquations(const std::vector<float>& mat, const std::vector<float>& vec,
                                        const float epsilon, const float tau)
{
    std::vector<float> output(vec.size(), 0);

    std::vector<float> Ax(vec.size(), 0);
    const float normB = CalculateNorm(vec);

    std::vector<float> tempAx(output.size(), 0);
    size_t iterationCount = 0;

    bool bLimitReached = false;
#if VAR_2
#pragma omp parallel num_threads(s_WORKER_THREADS_CURRENT)
#endif
    while (true)
    {
#if VAR_1
#if SCHEDULING
#pragma omp parallel for num_threads(s_WORKER_THREADS_CURRENT) schedule(SCHEDULE_TYPE, CHUNK_SIZE)
#else
#pragma omp parallel for num_threads(s_WORKER_THREADS_CURRENT)
#endif
#endif

#if VAR_2
#if SCHEDULING
#pragma omp for schedule(SCHEDULE_TYPE, CHUNK_SIZE)
#else
#pragma omp for
#endif
#endif
        for (size_t i = 0; i < output.size(); ++i)
        {
            for (size_t j = 0; j < output.size(); ++j)
            {
                Ax[i] += mat[i * output.size() + j] * output[j];

                tempAx[i] += mat[i * Ax.size() + j] * output[j];
            }

            tempAx[i] -= vec[i];
            output[i] -= tau * (Ax[i] - vec[i]);
        }

#pragma omp single
        {
            if (CalculateNorm(tempAx) / normB < epsilon) bLimitReached = true;

            Ax.assign(Ax.size(), 0);
            tempAx.assign(tempAx.size(), 0);
            ++iterationCount;
        }

        if (bLimitReached)break;
    }

  //  printf("Iterations: %zu\n", iterationCount);
    return output;
}

int main()
{
    static_assert((VAR_1 && VAR_2) == false);

    const auto matA = LoadData("../data/Test/matA.bin");
    if (matA.size() != s_MAT_SIZE)
    {
        printf("Failed to fullfill matA!\n");
        return 0;
    }

    const auto vecB = LoadData("../data/Test/vecB.bin");
    if (vecB.size() != s_VEC_SIZE)
    {
        printf("Failed to fullfill vecB!\n");
        return 0;
    }

#if OUTPUT_INTO_FILE
    std::string outputFile = VAR_1 ? "var_1_accel.txt" : VAR_2 ? "var_2_accel.txt" : "none.txt";
    std::ofstream foutAccel("../data/" + outputFile, std::ios::out | std::ios::trunc);
    if (!foutAccel.is_open())
    {
        printf("Failed to open file!\n");
        return 0;
    }

    outputFile = VAR_1 ? "var_1_efficiency.txt" : VAR_2 ? "var_2_efficiency.txt" : "none.txt";
    std::ofstream foutEffeciency("../data/" + outputFile, std::ios::out | std::ios::trunc);
    if (!foutEffeciency.is_open())
    {
        printf("Failed to open file!\n");
        return 0;
    }

    outputFile = VAR_1 ? "var_1_raw.txt" : VAR_2 ? "var_2_raw.txt" : "none.txt";
    std::ofstream foutRaw("../data/" + outputFile, std::ios::out | std::ios::trunc);
    if (!foutRaw.is_open())
    {
        printf("Failed to open file!\n");
        return 0;
    }
#endif

    constexpr float epsilon = 10e-3;
    constexpr float tau = -10e-2;

    double bestConsecutiveExecTime = MAXFLOAT;
    for (; s_WORKER_THREADS_CURRENT <= s_WORKER_THREADS; ++s_WORKER_THREADS_CURRENT)
    {
        double minTime = MAXFLOAT;
        double maxAccel = -MAXFLOAT;
        double maxEffi = -MAXFLOAT;

        for (uint32_t step{}; step < STEPS_COUNT; ++step)
        {
            Timer t = {};

            const auto vecX = SolveLinearEquations(matA, vecB, epsilon, tau);

            const auto elapsedSeconds = t.GetElapsedSeconds();
#if 0
            SaveData("../data/Test/vecX.bin", vecX.data(), vecX.size() * sizeof(vecX[0]));
#endif

            if (s_WORKER_THREADS_CURRENT == 1)
                bestConsecutiveExecTime = std::min(elapsedSeconds, bestConsecutiveExecTime);

            const auto acceleration = bestConsecutiveExecTime / elapsedSeconds;
            const auto efficiency = acceleration / s_WORKER_THREADS_CURRENT * 100.f;

            minTime = std::min(minTime, elapsedSeconds);
            maxAccel = std::max(maxAccel, acceleration);
            maxEffi = std::max(maxEffi, efficiency);
        }

#if OUTPUT_INTO_FILE
        foutRaw << s_WORKER_THREADS_CURRENT << "," << minTime << std::endl;
        foutAccel << s_WORKER_THREADS_CURRENT << "," << maxAccel << std::endl;
        foutEffeciency << s_WORKER_THREADS_CURRENT << "," << maxEffi << std::endl;
        printf("Done: %u\n", s_WORKER_THREADS_CURRENT);
#else
        printf("Threads: (%u), execution: (%0.4f) seconds. Accel: (%0.4f), Efficiency: (%0.2f)\n",
               s_WORKER_THREADS_CURRENT,
               minTime, maxAccel, maxEffi);
#endif
    }

#if OUTPUT_INTO_FILE
    foutAccel.close();
    foutEffeciency.close();
    foutRaw.close();
#endif


#if 0
    printf("vecX:\n");
    for (auto& v : vecX)
        printf("%f\n", v);
#endif

    return 0;
}
