#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstring>

#include <cassert>
#include <cmath>
#include <cfloat>

#include "CoreUtils.h"

static int32_t s_ProcNum     = 0;
static int32_t s_ClusterSize = 0;

static constexpr uint32_t N = 16;
constexpr float paramA      = 10e5;
constexpr float epsilon     = 10e-8;

#define GRID_INDEX(i, j, k) ((i) * N * N + (j) * N + (k))

constexpr int32_t Dx = 2, Dy = 2, Dz = 2;
constexpr int32_t X0 = -1, Y0 = -1, Z0 = -1;

// Precalculated constants.
constexpr float Hx = Dx / static_cast<float>(N - 1), Hy = Dy / static_cast<float>(N - 1), Hz = Dz / static_cast<float>(N - 1);
constexpr float invHx = 1.f / Hx, invHy = 1.f / Hy, invHz = 1.f / Hz;
constexpr float invHxSq = invHx * invHx, invHySq = invHy * invHy, invHzSq = invHz * invHz;
constexpr float coeff = 1.f / (2 * invHxSq + 2 * invHySq + 2 * invHzSq + paramA);

#define DUMP_LAYERS 0

FORCEINLINE float Phi(const float x, const float y, const float z)
{
    return x * x + y * y + z * z;
}

FORCEINLINE float Ro(const float x, const float y, const float z)
{
    return 6 - paramA * Phi(x, y, z);
}

FORCEINLINE float GetCoordX(const uint32_t i)
{
    return X0 + static_cast<float>(i) * Hx;
}

FORCEINLINE float GetCoordY(const uint32_t j)
{
    return Y0 + static_cast<float>(j) * Hy;
}

FORCEINLINE float GetCoordZ(const uint32_t k)
{
    return Z0 + static_cast<float>(k) * Hz;
}

#if DUMP_LAYERS
void DumpLayer(const std::vector<float>& grid, const uint32_t height, const std::string& filePath)
{
    std::ofstream out(filePath, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return;

    for (uint32_t x{}; x < N; ++x)
    {
        for (uint32_t y{}; y < N; ++y)
        {
            for (uint32_t z{}; z < height; ++z)
            {
                out << grid[GRID_INDEX(x, y, z)] << " ";
            }
            out << std::endl;
        }
        out << std::endl;
    }

    out.close();
}
#endif

FORCEINLINE void InitializeGrid(std::vector<float>& grid, const uint32_t layerSize, const int32_t layerCoord)
{
    for (uint32_t i{}; i < layerSize + 2; ++i)
    {
        const float z = GetCoordZ(layerCoord + i);
        for (uint32_t j{}; j < N; ++j)
        {
            const float x = GetCoordX(j);

            for (uint32_t k{}; k < N; ++k)
            {
                const float y = GetCoordY(k);

                if (k != 0 && k != N - 1 && j != 0 && j != N - 1 && z != Z0 && z != Z0 + Dz)
                {
                    grid[GRID_INDEX(i, j, k)] = 0;  // body
                }
                else
                {
                    grid[GRID_INDEX(i, j, k)] = Phi(x, y, z);  // edges
                }
            }
        }
    }

#if DUMP_LAYERS
    DumpLayer(grid, layerSize + 2, std::to_string(s_ProcNum));
#endif
}

FORCEINLINE float UpdateLayerAndGetDelta(const std::vector<float>& layer, std::vector<float>& layerBuffer, const uint32_t layerIdx,
                                         const int32_t layerCoord)
{
    const int32_t globalLayerCoord = layerCoord + static_cast<int32_t>(layerIdx);
    // std::cout << "ProcNum[" << s_ProcNum << "] got globalLayerCoord(inside whole omega): " << globalLayerCoord << std::endl;

    float maxDelta = FLT_MIN;

    if (globalLayerCoord == 0 || globalLayerCoord == N - 1)  // lower/upper border
    {
        const uint32_t layerDataOffset = layerIdx * N * N;
        memcpy(layerBuffer.data() + layerDataOffset, layer.data() + layerDataOffset, N * N * sizeof(layer[0]));
        return 0.f;
    }

    const float z = GetCoordZ(globalLayerCoord);
    for (uint32_t i{}; i < N; ++i)
    {
        const float x = GetCoordX(i);
        for (uint32_t j{}; j < N; ++j)
        {
            const float y = GetCoordY(j);

            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)  // side parts and front
            {
                layerBuffer[GRID_INDEX(layerIdx, i, j)] = layer[GRID_INDEX(layerIdx, i, j)];
                continue;
            }

            const float partX = invHxSq * (layer[GRID_INDEX(layerIdx, i + 1, j)] + layer[GRID_INDEX(layerIdx, i - 1, j)]);
            const float partY = invHySq * (layer[GRID_INDEX(layerIdx, i, j + 1)] + layer[GRID_INDEX(layerIdx, i, j - 1)]);
            const float partZ = invHzSq * (layer[GRID_INDEX(layerIdx + 1, i, j)] + layer[GRID_INDEX(layerIdx - 1, i, j)]);
            layerBuffer[GRID_INDEX(layerIdx, i, j)] = coeff * (partX + partY + partZ - Ro(x, y, z));

            maxDelta = std::max(maxDelta, std::abs(layerBuffer[GRID_INDEX(layerIdx, i, j)] - layer[GRID_INDEX(layerIdx, i, j)]));
        }
    }

    return maxDelta;
}

FORCEINLINE std::vector<float> SolveDE()
{
    const uint32_t layerSize = N / s_ClusterSize;  // height not including shadow borders
    if (s_ProcNum == 0) std::cout << "3D grid [ " << N << ", " << N << ", " << layerSize << "]\n";

    const int32_t layerCoord = s_ProcNum * static_cast<int32_t>(layerSize) - 1;
    //  std::cout << "ProcNum[" << s_ProcNum << "] got layerCoordZ(global offset along Z): " << layerCoord << std::endl;

    std::vector<float> layerBuffer(N * N * (layerSize + 2), 0);  // storing next state
    std::vector<float> layer(N * N * (layerSize + 2), 0);        // current
    InitializeGrid(layer, layerSize, layerCoord);

    MPI::Request lower[2]   = {};
    MPI::Request upper[2]   = {};
    float globalDelta       = FLT_MAX;
    uint32_t iterationCount = 0;
    while (globalDelta > epsilon)
    {
        ++iterationCount;

        float localDelta    = .0f;
        float localMaxDelta = FLT_MIN;

        // c++ memory layout left->right; top->bootom.
        if (s_ProcNum > 0)  // lower kuso4ek sender, upper receiver
        {
            lower[0] = MPI::COMM_WORLD.Isend(layer.data() + N * N, N * N, MPI::FLOAT, s_ProcNum - 1, 666);  // mark0
            lower[1] = MPI::COMM_WORLD.Irecv(layer.data(), N * N, MPI::FLOAT, s_ProcNum - 1, 666);          // mark1
        }

        if (s_ProcNum < s_ClusterSize - 1)  // upper kuso4ek sender, lower receiver
        {
            upper[0] = MPI::COMM_WORLD.Isend(layer.data() + N * N * layerSize, N * N, MPI::FLOAT, s_ProcNum + 1, 666);        // mark1
            upper[1] = MPI::COMM_WORLD.Irecv(layer.data() + N * N * (layerSize + 1), N * N, MPI::FLOAT, s_ProcNum + 1, 666);  // mark0
        }

        // update korobo4ka
        for (uint32_t layerIdx = 2; layerIdx < layerSize; ++layerIdx)
        {
            localDelta    = UpdateLayerAndGetDelta(layer, layerBuffer, layerIdx, layerCoord);
            localMaxDelta = std::max(localMaxDelta, localDelta);
        }

        //    while (lower[1].Test() || upper[1].Test()){   sleep/wait      }

        if (s_ProcNum > 0)
        {
            lower[1].Wait();  // Wait till receiving finished
        }

        localDelta    = UpdateLayerAndGetDelta(layer, layerBuffer, 1, layerCoord);  // upper kuso4ek
        localMaxDelta = std::max(localMaxDelta, localDelta);

        if (s_ProcNum < s_ClusterSize - 1)
        {
            upper[1].Wait();  // Wait till receiving finished
        }

        localDelta    = UpdateLayerAndGetDelta(layer, layerBuffer, layerSize, layerCoord);  //  lower kuso4ek
        localMaxDelta = std::max(localMaxDelta, localDelta);

        if (s_ProcNum > 0) lower[0].Wait();                  // Wait till sending finished
        if (s_ProcNum < s_ClusterSize - 1) upper[0].Wait();  // Wait till sending finished

        MPI::COMM_WORLD.Allreduce(&localMaxDelta, &globalDelta, 1, MPI::FLOAT, MPI::MAX);

        if (s_ProcNum == 0)
        {
            std::cout << "Allreduce delta: " << globalDelta << std::endl;
            std::cout << "Epsilon: " << epsilon << std::endl;
        }

        memcpy(layer.data(), layerBuffer.data(), layer.size() * sizeof(layer[0]));
    }

    if (s_ProcNum == 0)
    {
        std::cout << "Iterations: " << iterationCount << std::endl;
    }

    std::vector<float> result = {};
    if (s_ProcNum == 0)
    {
        result.resize(N * N * N, 0);
    }
    const auto recvCount = static_cast<int32_t>(layerSize * N * N);
    MPI::COMM_WORLD.Gather(layer.data() + N * N, recvCount, MPI::FLOAT, result.data(), recvCount, MPI::FLOAT, 0);

    return result;
}

FORCEINLINE float CalculateSolutionDelta(const std::vector<float>& grid)
{
    float maxDelta = FLT_MIN;

    for (uint32_t i{}; i < N; ++i)
    {
        const float x = GetCoordX(i);
        for (uint32_t j{}; j < N; ++j)
        {
            const float y = GetCoordY(j);
            for (uint32_t k{}; k < N; ++k)
            {
                const float z = GetCoordZ(k);

                const float currentDelta = std::abs(grid[GRID_INDEX(i, j, k)] - Phi(x, y, z));
                maxDelta                 = std::max(maxDelta, currentDelta);
            }
        }
    }

    return maxDelta;
}

#include <queue>
#include <atomic>

int main(int argc, char** argv)
{
    MPI::Init(argc, argv);

    s_ProcNum     = MPI::COMM_WORLD.Get_rank();
    s_ClusterSize = MPI::COMM_WORLD.Get_size();

    if (N % s_ClusterSize != 0)
    {
        if (s_ProcNum == 0) std::cout << "N % clusterSize != 0\n";
        return 0;
    }

    if (s_ProcNum == 0)
    {
        std::cout << "ClusterSize:" << s_ClusterSize << std::endl;
    }

    const double startTime            = MPI::Wtime();
    const std::vector<float> solution = SolveDE();
    const double diff                 = MPI::Wtime() - startTime;

    double maxTime = 0.0;
    MPI::COMM_WORLD.Allreduce(&diff, &maxTime, 1, MPI::DOUBLE, MPI::MAX);

    if (s_ProcNum == 0)
    {
        std::cout << "Max time: " << maxTime << " seconds.\n";
        std::cout << "Max Delta: " << CalculateSolutionDelta(solution) << std::endl;

#if DUMP_LAYERS
        DumpLayer(solution, N, "result.txt");
#endif
    }

    MPI::Finalize();

    return 0;
}