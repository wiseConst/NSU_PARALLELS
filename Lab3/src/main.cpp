#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <array>

#include "CoreUtils.h"

static int32_t s_ProcNum = 0;
static int32_t s_ClusterSize = 0;

static constexpr int32_t s_DimCount = 2;
static std::array<int32_t, s_DimCount> s_Dims = {0}; // Process count in each dimension.
static std::array<int32_t, s_DimCount> s_Coords = {0}; // Process coordinates inside grid communicator.

static MPI_Comm s_GridComm = {};
static MPI_Comm s_RowComm = {};
static MPI_Comm s_ColComm = {};

static constexpr uint32_t n1 = 1600;
static constexpr uint32_t n2 = 2500;
static constexpr uint32_t n3 = 2800;
static constexpr uint32_t p1 = 4;
static constexpr uint32_t p2 = 4;

#define OUTPUT_INTO_FILE 0
#define DUMP_MAT_A 1
#define DUMP_MAT_B 1
#define DUMP_MAT_C 1

struct Matrix {
    Matrix() = default;

    Matrix(const uint32_t rows, const uint32_t columns) : Rows(rows), Columns(columns) {
        assert(rows > 0 && columns > 0);
        Data.resize(rows * columns, 0);
    }

    ~Matrix() = default;

    void Print() const {
        for (uint32_t row{}; row < Rows; ++row) {
            for (uint32_t col{}; col < Columns; ++col) {
                std::cout << Data[row * Columns + col] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void Dump(const std::string &outputFilePath) const {
        std::ofstream out(outputFilePath, std::ios::out);
        if (!out.is_open()) return;
        for (uint32_t row{}; row < Rows; ++row) {
            for (uint32_t col{}; col < Columns; ++col) {
                out << Data[row * Columns + col] << " ";
            }
            out << std::endl;
        }
        out.close();
    }

    Matrix MultiplyMatrices(
            const Matrix &mat) const {
        assert(Rows > 0 && Columns > 0 && mat.Rows > 0 && mat.Columns > 0 && //
               Columns == mat.Rows);

        Matrix matC(Rows, mat.Columns);

        for (uint32_t rowA{}; rowA < Rows; ++rowA) {
            for (uint32_t colB{}; colB < mat.Columns; ++colB) {
                for (uint32_t colA{}; colA < Columns; ++colA) {
                    matC.Data[rowA * mat.Columns + colB] +=
                            Data[rowA * Columns + colA] * mat.Data[colA * mat.
                                    Columns + colB];
                }
            }
        }

        return matC;
    }

    static Matrix GetRandomMatrix(const uint32_t rows, const uint32_t cols) {
        Matrix matrix(rows, cols);
        for (uint32_t row{}; row < rows; ++row) {
            for (uint32_t col{}; col < cols; ++col)
                matrix.Data[row * cols + col] = s_FloatDistribution(s_Gen);
        }

        return matrix;
    }

    uint32_t Rows = 0;
    uint32_t Columns = 0;
    std::vector<float> Data = {};
};

static void InitCommunicators() {
    s_Dims = {p1, p2};

    if (s_ProcNum == 0) {
        std::cout << "Dimensions x:" << s_Dims[0] << ", y:" << s_Dims[1] << std::endl;
    }

    constexpr int32_t periods[s_DimCount] = {0};
    constexpr int32_t reorder = 1; // MPI API can reorder process pos in grid.

    // Creating global grid communicator.
    MPI_Cart_create(MPI_COMM_WORLD, s_DimCount, s_Dims.data(), periods, reorder, &s_GridComm);

    // Creating sub-communicator along X axis.
    constexpr int32_t remainDimsX[s_DimCount] = {0, 1};
    MPI_Cart_sub(s_GridComm, reinterpret_cast<const int *>(&remainDimsX), &s_RowComm);

    // Creating sub-communicator along Y axis.
    constexpr int32_t remainDimsY[s_DimCount] = {1, 0};
    MPI_Cart_sub(s_GridComm, reinterpret_cast<const int *>(&remainDimsY), &s_ColComm);
}

static void DestroyCommunicators() {
    MPI_Comm_free(&s_RowComm);
    MPI_Comm_free(&s_ColComm);
    MPI_Comm_free(&s_GridComm);
}

void ScatterMatA(const Matrix &matA, Matrix &partMatA, const std::vector<int32_t> &sendcounts,
                 const std::vector<int32_t> &displacements) {
    assert(partMatA.Data.data());

    const auto partMatASize = partMatA.Columns * partMatA.Rows;
    // Scatter rows along Y.(if process's x==0 and y=any)
    if (s_Coords[1] == 0) {
        //   printf("ProcNum along first column:%d\n", s_ProcNum);
        MPI_Scatterv(matA.Data.data(), sendcounts.data(), displacements.data(), MPI_FLOAT, partMatA.Data.data(),
                     partMatASize, MPI_FLOAT, 0, s_ColComm);
    }

    // BCast along X
    MPI_Bcast(partMatA.Data.data(), partMatASize, MPI_FLOAT, 0, s_RowComm);
#if DUMP_MAT_A
    partMatA.Dump("A" + std::to_string(s_ProcNum));
#endif
}

void ScatterMatB(const Matrix &matB, Matrix &partMatB) {
    assert(partMatB.Data.data());
    const auto partMatBSize = partMatB.Rows * partMatB.Columns;

    if (s_Coords[0] == 0) {
        MPI_Datatype columnType = {}, resizedColumnType = {};

        MPI_Type_vector(n2, partMatB.Columns, n3, MPI_FLOAT, &columnType);
        MPI_Type_commit(&columnType);

        MPI_Type_create_resized(columnType, 0,
                                partMatB.Columns * sizeof(float), // Used data size(bytes) without holes.
                                &resizedColumnType);
        MPI_Type_commit(&resizedColumnType);

        MPI_Scatter(matB.Data.data(), 1, resizedColumnType,
                    partMatB.Data.data(), partMatBSize, MPI_FLOAT,
                    0, s_RowComm);

        MPI_Type_free(&columnType);
        MPI_Type_free(&resizedColumnType);
    }

    // BCast along Y.
    MPI_Bcast(partMatB.Data.data(), partMatBSize, MPI_FLOAT, 0, s_ColComm);
#if DUMP_MAT_B
    partMatB.Dump("B" + std::to_string(s_ProcNum));
#endif
}

bool AreMatricesEqual(const Matrix &lhs, const Matrix &rhs) {
    if (lhs.Rows != rhs.Rows || lhs.Columns != rhs.Columns) return false;
    for (uint32_t row{}; row < lhs.Rows; ++row) {
        for (uint32_t col{}; col < lhs.Columns; ++col)
            if (lhs.Data[row * lhs.Columns + col] != rhs.Data[row * lhs.Columns + col])return false;

    }
    return true;
}

void GatherMatC(Matrix &matC, const Matrix &localMatC) {
    assert(localMatC.Data.data());

    const auto partMatCSize = localMatC.Rows * localMatC.Columns;
#if 0
    if (s_ProcNum == 0) {
        std::cout << "local mat C size = " << partMatCSize << ", n3 = " << n3 << std::endl;
    }
#endif

    MPI_Datatype blockType = {}, resizedBlockType = {};
    MPI_Type_vector(localMatC.Rows, localMatC.Columns, n3, MPI_FLOAT, &blockType);
    MPI_Type_commit(&blockType);

    MPI_Type_create_resized(blockType, 0,
                            localMatC.Columns * sizeof(float), // offset = chunkC.columns * sizeof(float)
                            &resizedBlockType);
    MPI_Type_commit(&resizedBlockType);

    std::vector<int32_t> displacements(s_Dims[0] * s_Dims[1], 0);
    for (int32_t row{}; row < s_Dims[0]; ++row) {
        for (int32_t col{}; col < s_Dims[1]; ++col) {
            displacements[row * s_Dims[1] + col] = row * s_Dims[1] * localMatC.Rows + col;

#if 1
            if (s_ProcNum == 0) {
                std::cout << "Displacements[" << row * s_Dims[1] + col << "]=" << col + row * s_Dims[1] * localMatC.Rows
                          << std::endl;
            }
#endif
        }
    }

    std::vector<int32_t> recvcounts(s_Dims[0] * s_Dims[1], 1);
    MPI_Gatherv(localMatC.Data.data(), partMatCSize, MPI_FLOAT,
                matC.Data.data(), recvcounts.data(), displacements.data(),
                resizedBlockType, 0, s_GridComm);

    MPI_Type_free(&blockType);
    MPI_Type_free(&resizedBlockType);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &s_ClusterSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &s_ProcNum);

    if (p1 * p2 != s_ClusterSize || n1 % p1 != 0 || n3 % p2 != 0) {
        printf("Invalid p1/p2 || n1/n2/n3 params!\n");
        MPI_Finalize();
        return 0;
    }

    char processorName[MPI_MAX_PROCESSOR_NAME] = {0};
    int32_t nameLength = 0;
    MPI_Get_processor_name(processorName, &nameLength);
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processorName, s_ProcNum, s_ClusterSize);

    InitCommunicators();
    MPI_Cart_coords(s_GridComm, s_ProcNum, s_DimCount, s_Coords.data());
    printf("Proc %d has pos: (%d, %d)\n", s_ProcNum, s_Coords[0], s_Coords[1]);

    Matrix matA = {}, matB = {};
    if (s_ProcNum == 0 || s_Coords[0] == 0 && s_Coords[1] == 0) // left upper cell (coordinate space: x right, y down)
    {
        matA = Matrix::GetRandomMatrix(n1, n2);
        matB = Matrix::GetRandomMatrix(n2, n3);

        std::cout << "A:(" << matA.Rows << ", " << matA.Columns << ")\n";
        //  matA.Print();

        std::cout << "B:(" << matB.Rows << ", " << matB.Columns << ")\n";
        // matB.Print();
    }

    // Dealing with A
    std::vector<int32_t> rowsPerProcess(s_Dims[0]);
    for (uint32_t i{}; i < rowsPerProcess.size(); ++i) {
        rowsPerProcess[i] = n1 / s_Dims[0];
        if (i < n1 % s_Dims[0])
            ++rowsPerProcess[i];
    }

    std::vector<int32_t> sendcountsA(s_Dims[0], 0);
    std::vector<int32_t> displacementsA(s_Dims[0], 0);
    int32_t matOffsetA = 0;
    for (uint32_t i{}; i < s_Dims[0]; ++i) {
        displacementsA[i] = matOffsetA;
        sendcountsA[i] = rowsPerProcess[i] * n2;
        matOffsetA += sendcountsA[i];
    }

    Matrix partMatA(rowsPerProcess[s_Coords[0]], n2);
    ScatterMatA(matA, partMatA, sendcountsA, displacementsA);

    // Dealing with B
    std::vector<int32_t> columnsPerProcess(s_Dims[1]);
    for (uint32_t i{}; i < columnsPerProcess.size(); ++i) {
        columnsPerProcess[i] = n3 / s_Dims[1];
        if (i < n3 % s_Dims[1])
            ++columnsPerProcess[i];

#if 0
        if (s_ProcNum == 0)
            std::cout << "Proc: " << i * s_Dims[1] << " has columns: " << columnsPerProcess[i] << std::endl;
#endif
    }

    Matrix partMatB(n2, columnsPerProcess[s_Coords[1]]);
    ScatterMatB(matB, partMatB);

    const auto start = MPI_Wtime();
    Matrix localMatC = partMatA.MultiplyMatrices(partMatB);
#if 1
    if (s_ProcNum == 0) {
        std::cout << "local mat C:(" << localMatC.Rows << ", " << localMatC.Columns << ")\n";
    }
#endif

#if DUMP_MAT_C
    localMatC.Dump("C" + std::to_string(s_ProcNum));
#endif

    Matrix gatheredMatC(n1, n3);
    GatherMatC(gatheredMatC, localMatC);

#if 1
    if (s_ProcNum == 0) {
        std::cout << "mat C:(" << gatheredMatC.Rows << ", " << gatheredMatC.Columns << ")\n";
    }
#endif

    const double time = MPI_Wtime() - start;
    double maxTime = 0.0;
    MPI_Reduce(&time, &maxTime,
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (s_ProcNum == 0) printf("Time spent: %lf\n", maxTime);

    if (s_ProcNum == 0) {
        const auto matC = matA.MultiplyMatrices(matB);
        if (!AreMatricesEqual(matC, gatheredMatC)) {
            printf("matC!=gatheredMatC!\n");
        }
    }

#if OUTPUT_INTO_FILE
    if (s_ProcNum == 0)
        SaveData("../data/matC.bin", matC.data(), matC.size() * sizeof(matC[0]));
#endif

    DestroyCommunicators();
    MPI_Finalize();

    return 0;
}
