#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <numeric>

#include "CoreUtils.h"

static constexpr uint32_t N          = 50;
static constexpr uint32_t s_VEC_SIZE = N * N;
static constexpr uint32_t s_MAT_SIZE = s_VEC_SIZE * s_VEC_SIZE;
static int32_t s_ProcNum             = 0;
static int32_t s_ClusterSize         = 0;

#define OUTPUT_INTO_FILE 1
#define LOG_ITERATIONS 0

float CalculateNormSquared(const std::vector<float>& vec)
{
    const float sumOfSquares = std::accumulate(vec.begin(), vec.end(), 0.F,
                                               [](const float acc, const float val)-> float
                                               {
                                                   return acc + val * val;
                                               });

    return sumOfSquares;
}

std::vector<float> SolveLinearEquations(const std::vector<int32_t>& adjustedVecOffsets, const std::vector<int32_t>& vecDisplacements,
                                        const std::vector<float>& partMatA, const std::vector<float>& vec,
                                        const float epsilon, const float tau)
{
    // global X solution.
    std::vector<float> output(vec.size(), 0);

    // eps^2 * || b ||
    const float epsilon2normB = epsilon * epsilon * std::sqrt(CalculateNormSquared(vec));

    std::vector<float> localAxb(adjustedVecOffsets[s_ProcNum], 0);
    std::vector<float> localOutput(adjustedVecOffsets[s_ProcNum], 0);

    size_t iterationCount = 0;
    bool bLimitReached    = false;
    while (!bLimitReached)
    {
        for (size_t i = 0; i < adjustedVecOffsets[s_ProcNum]; ++i)
        {
            // Calculate Ax-b.
            {
                localAxb[i] = -vec[vecDisplacements[s_ProcNum] + i];

                for (size_t j = 0; j < s_VEC_SIZE; ++j)
                {
                    localAxb[i] += partMatA[i * s_VEC_SIZE + j] * output[j];
                }
            }

            // Calculate next X.
            localOutput[i] -= tau * localAxb[i];
        }

        // Gather X and Bcast it.
        MPI_Allgatherv(localOutput.data(), adjustedVecOffsets[s_ProcNum],MPI_FLOAT, output.data(), adjustedVecOffsets.data(),
                       vecDisplacements.data(),MPI_FLOAT,
                       MPI_COMM_WORLD);

        // Part norm.
        const auto localEpsilon = CalculateNormSquared(localAxb);

        // Firstly sum up all local epsilons, then Bcast them.
        float globalEpsilon = 0.0f;
        MPI_Allreduce(&localEpsilon, &globalEpsilon, 1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);

        if (globalEpsilon < epsilon2normB) bLimitReached = true;

#if LOG_ITERATIONS
        if (s_ProcNum == 0)
        {
            printf("Iterations: %zu\n", iterationCount + 1);
            printf("eps: %f\n", std::sqrt(globalEpsilon));
        }
#endif

        ++iterationCount;
    }

    if (s_ProcNum == 0)
        printf("Calculation ended, iterations: %zu\n", iterationCount);

    return output;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &s_ClusterSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &s_ProcNum);

    char processorName[MPI_MAX_PROCESSOR_NAME] = {0};
    int32_t nameLength                         = 0;
    MPI_Get_processor_name(processorName, &nameLength);

    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processorName, s_ProcNum, s_ClusterSize);

    // Root owns matA.
    std::vector<float> matA;
    if (s_ProcNum == 0)
    {
        matA = LoadData("../data/matA.bin");

        if (matA.size() != s_MAT_SIZE)
        {
            printf("Failed to fullfill matA!\n");
            return 0;
        }
    }

    // Calculate offsets and displacements for scattering matA around other processes.
    // 1. Offsets.
    std::vector<int32_t> adjustedMatSizes(s_ClusterSize);
    std::vector<int32_t> adjustedVecOffsets(s_ClusterSize);
    const int32_t vecOffset = static_cast<int32_t>(s_VEC_SIZE) / s_ClusterSize;
    for (uint32_t i = 0; i < s_ClusterSize; ++i)
    {
        adjustedVecOffsets[i] = vecOffset;
        if (i < s_VEC_SIZE % s_ClusterSize)
            ++adjustedVecOffsets[i];

        adjustedMatSizes[i] = adjustedVecOffsets[i] * static_cast<int32_t>(s_VEC_SIZE); // How much each process takes of matA.
    }

    // 2. Displacements.
    std::vector<int32_t> matDisplacements(s_ClusterSize);
    std::vector<int32_t> vecDisplacements(s_ClusterSize);
    int32_t vecDisplacementOffset = 0;
    int32_t matDisplacementOffset = 0;
    for (uint32_t i = 0; i < s_ClusterSize; ++i)
    {
        matDisplacements[i] = matDisplacementOffset;
        vecDisplacements[i] = vecDisplacementOffset;

        vecDisplacementOffset += adjustedVecOffsets[i];
        matDisplacementOffset += adjustedMatSizes[i];
    }

    // 3. Scatter big matA around each process.
    std::vector<float> partMatA(adjustedMatSizes[s_ProcNum]);
    MPI_Scatterv(matA.data(), adjustedMatSizes.data(), matDisplacements.data(),MPI_FLOAT, partMatA.data(),
                 static_cast<int32_t>(partMatA.size()),MPI_FLOAT, 0,
                 MPI_COMM_WORLD);

    const auto vecB = LoadData("../data/vecB.bin");
    if (vecB.size() != s_VEC_SIZE)
    {
        printf("Failed to fullfill vecB!\n");
        return 0;
    }

    constexpr float epsilon = 10e-4;
    constexpr float tau     = -10e-2;

    const auto start = MPI_Wtime();
    const auto vecX  = SolveLinearEquations(adjustedVecOffsets, vecDisplacements, partMatA, vecB, epsilon, tau);

    const double time = MPI_Wtime() - start;
    double maxTime    = 0;
    MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (s_ProcNum == 0) printf("Time spent: %lf seconds.", maxTime);

#if OUTPUT_INTO_FILE
    if (s_ProcNum == 0)
        SaveData("../data/vecX.bin", vecX.data(), vecX.size() * sizeof(vecX[0]));
#endif

    MPI_Finalize();

    return 0;
}