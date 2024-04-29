#include <iostream>
#include <thread>
#include <mpi.h>
#include <cassert>

#include "CoreUtils.h"
#include "ConcurrentQueue.h"

static int32_t s_ProcNum     = 0;
static int32_t s_ClusterSize = 0;

static std::thread s_ReceiverThread = {};
static std::thread s_SolverThread   = {};
static bool s_bIsRunning            = false;
static auto s_TaskQueue             = ConcurrentQueue<uint32_t>();
static double s_DisbalanceSum       = 0.0;

#define ITERATIONS_TASK_COUNT 25u
#define BASE_TASK_COUNT 2000
#define TASKS_MULTIPLIER_MS 1

#define SOLVER_FINISHED (UINT32_MAX / 20)
#define NO_TASKS_TO_SHARE (SOLVER_FINISHED - 1)

#define OUTPUT_STATS 1
#define LOG_INFO 0
#define STEAL 1
#define RANDOMIZE 0  // thrash results

#if RANDOMIZE

#include <random>
#include <limits>

static std::random_device s_RandomDevice = {};
static std::mt19937 s_RandomEngine(s_RandomDevice());
static std::uniform_int_distribution<uint32_t> s_UInt32Distribution{0, std::numeric_limits<uint32_t>::max() - 2};

#endif

static void LaunchReceiverThread()
{
    s_ReceiverThread = std::thread(
        [&]
        {
            MPI::COMM_WORLD.Barrier();

            uint32_t pendingMessage         = 0;
            int32_t procNumReceiver         = 0;
            std::vector<uint32_t> taskArray = {};
            while (s_bIsRunning)
            {
                // NOTE: Using Irecv instead of Recv, in case we have only one process.
                auto request = MPI::COMM_WORLD.Irecv(&pendingMessage, 1, MPI::UNSIGNED, MPI::ANY_SOURCE, s_ClusterSize + 1);
                while (s_bIsRunning && !request.Test())
                {
                }

                if (!s_bIsRunning) return;

                if (pendingMessage == SOLVER_FINISHED)
                {
#if LOG_INFO
                    std::cout << ANSI_RED << "RECEIVER_THREAD_" << s_ProcNum << ": "
                              << "Proc[" << s_ProcNum << "] finished." << ANSI_RESET << std::endl;
#endif
                    s_bIsRunning = false;
                    return;
                }

                procNumReceiver = static_cast<int32_t>(pendingMessage);
#if LOG_INFO
                std::cout << ANSI_RED << "RECEIVER_THREAD_" << s_ProcNum << ": "
                          << "Proc[" << procNumReceiver << "] wants tasks from Proc[" << s_ProcNum << "]" << ANSI_RESET << std::endl;
#endif

                uint32_t taskCount = 0;
                if (s_TaskQueue.Size() > BASE_TASK_COUNT / s_ClusterSize)
                {
                    const uint32_t taskOutputCount = BASE_TASK_COUNT / s_ClusterSize;
                    taskArray.resize(taskOutputCount);

                    for (uint32_t i{}; i < taskOutputCount; ++i)
                    {
                        if (!s_TaskQueue.TryPop(taskArray[i])) break;

                        ++taskCount;
                    }
                }

#if LOG_INFO
                std::cout << ANSI_RED << "RECEIVER_THREAD_" << s_ProcNum << ": "
                          << "Proc[" << s_ProcNum << "] sends (" << taskCount << ") tasks to Proc[" << procNumReceiver << "]" << ANSI_RESET
                          << std::endl;
#endif

                taskCount = taskCount == 0 ? NO_TASKS_TO_SHARE : taskCount;
                MPI::COMM_WORLD.Send(&taskCount, 1, MPI::UNSIGNED, procNumReceiver, s_ProcNum);

                if (taskCount != NO_TASKS_TO_SHARE)
                    MPI::COMM_WORLD.Send(taskArray.data(), static_cast<int32_t>(taskCount), MPI::UNSIGNED, procNumReceiver, s_ProcNum);

                procNumReceiver = 0;
                taskArray.clear();
            }
        });
}

static void LaunchSolverThread()
{
    s_SolverThread = std::thread(
        [&]
        {
            constexpr auto executeTasks = []
            {
                while (!s_TaskQueue.IsEmpty())
                {
                    const uint32_t sleepTime = s_TaskQueue.Top();
                    s_TaskQueue.Pop();
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
                }
            };

            constexpr auto populateQueue = [&](const int32_t taskIteration)
            {
                for (uint32_t i{}; i < BASE_TASK_COUNT; ++i)
                    s_TaskQueue.Push((std::abs(s_ProcNum - (taskIteration % s_ClusterSize)) + 1) *
#if RANDOMIZE
                                     s_UInt32Distribution(s_RandomEngine) %
#endif
                                     TASKS_MULTIPLIER_MS);
            };

#if OUTPUT_STATS
            std::ofstream out("Log_" + std::to_string(s_ProcNum) + ".csv", std::ios::out | std::ios::trunc);
            if (!out.is_open()) exit(0);
            out << "iter,time" << std::endl;
#endif

            uint32_t taskCountResponse = 0;
            double iterationDuration = 0.0, shortestDuration = 0.0, longestDuration = 0.0;
            for (uint32_t taskIteration = 0; taskIteration < ITERATIONS_TASK_COUNT; ++taskIteration)
            {
                const double iterationStartTime = MPI::Wtime();
                MPI::COMM_WORLD.Barrier();

                populateQueue(static_cast<int32_t>(taskIteration));
                executeTasks();

#if STEAL
                for (int32_t procNum{}; procNum < s_ClusterSize; ++procNum)
                {
                    if (procNum == s_ProcNum) continue;
#if LOG_INFO
                    // Request for task count.
                    std::cout << ANSI_GREEN << "SOLVER_THREAD_" << s_ProcNum << ": "
                              << "Proc[" << s_ProcNum << "] asks Proc[" << procNum << "] for some tasks." << ANSI_RESET << std::endl;
#endif
                    MPI::COMM_WORLD.Send(&s_ProcNum, 1, MPI::UNSIGNED, procNum, s_ClusterSize + 1);

                    // Get task count.
                    MPI::COMM_WORLD.Recv(&taskCountResponse, 1, MPI::UNSIGNED, procNum, procNum);
#if LOG_INFO
                    std::cout << ANSI_GREEN << "SOLVER_THREAD_" << s_ProcNum << ": "
                              << "Proc[" << procNum << "] answered with (" << taskCountResponse << ") tasks." << ANSI_RESET << std::endl;
#endif
                    if (taskCountResponse != NO_TASKS_TO_SHARE)
                    {
                        const auto receivedTaskCount = static_cast<int32_t>(taskCountResponse);
                        std::vector<uint32_t> receiveTaskArray(receivedTaskCount);
                        MPI::COMM_WORLD.Recv(receiveTaskArray.data(), receivedTaskCount, MPI::UNSIGNED, procNum, procNum);
#if LOG_INFO
                        std::cout << ANSI_GREEN << "SOLVER_THREAD_" << s_ProcNum << ": "
                                  << "Proc[" << s_ProcNum << "] received " << receivedTaskCount << " tasks from Proc[" << procNum << "]"
                                  << ANSI_RESET << std::endl;
#endif

                        for (const auto& val : receiveTaskArray)
                            s_TaskQueue.Push(val);

                        executeTasks();
                    }
                }
#endif
                iterationDuration = MPI::Wtime() - iterationStartTime;

                MPI::COMM_WORLD.Allreduce(&iterationDuration, &longestDuration, 1, MPI::DOUBLE, MPI::MAX);
                MPI::COMM_WORLD.Allreduce(&iterationDuration, &shortestDuration, 1, MPI::DOUBLE, MPI::MIN);

                s_DisbalanceSum += (longestDuration - shortestDuration) / longestDuration;

                if (s_ProcNum == 0)
                {
                    std::cout << "Disbalance rate is " << ((longestDuration - shortestDuration) / longestDuration) * 100.f << "%"
                              << std::endl;
                    std::cout << "SOLVER_THREAD_" << s_ProcNum << ": "
                              << "Current iteration: " << taskIteration << std::endl;
                }

#if OUTPUT_STATS
                out << taskIteration << "," << iterationDuration << std::endl;
#endif
            }

            constexpr uint32_t executionFinishedMessage = SOLVER_FINISHED;
            for (int32_t procNum{}; procNum < s_ClusterSize; ++procNum)
            {
                if (procNum == s_ProcNum) continue;

                MPI::COMM_WORLD.Send(&executionFinishedMessage, 1, MPI::UNSIGNED, procNum, s_ClusterSize + 1);
            }

            if (s_ClusterSize == 1) s_bIsRunning = false;
        });
}

int main(int argc, char** argv)
{
    if (MPI::Init_thread(argc, argv, MPI_THREAD_MULTIPLE) != MPI_THREAD_MULTIPLE)
    {
        std::cout << "MPI_THREAD_MULTIPLE not supported!" << std::endl;
        MPI::Finalize();
        return 0;
    }

    s_ProcNum     = MPI::COMM_WORLD.Get_rank();
    s_ClusterSize = MPI::COMM_WORLD.Get_size();
    if (s_ProcNum == 0)
    {
        std::cout << "ClusterSize: " << s_ClusterSize << std::endl;
        std::cout << "Iterations: " << ITERATIONS_TASK_COUNT << std::endl;
        std::cout << "Tasks: " << BASE_TASK_COUNT << std::endl;
    }

    const double tasksBeginTime = MPI::Wtime();

    s_bIsRunning = true;

    LaunchReceiverThread();
    LaunchSolverThread();

    if (s_ReceiverThread.joinable()) s_ReceiverThread.join();
    if (s_SolverThread.joinable()) s_SolverThread.join();

    const double tasksEndTime = MPI::Wtime() - tasksBeginTime;
    double tasksMaxDuration   = 0.0;
    MPI::COMM_WORLD.Allreduce(&tasksEndTime, &tasksMaxDuration, 1, MPI::DOUBLE, MPI::MAX);

    if (s_ProcNum == 0)
    {
        std::cout << "Summary disbalance: " << s_DisbalanceSum / ITERATIONS_TASK_COUNT * 100.f << "%" << std::endl;
        std::cout << "Balanced task solving via parallelization completed in " << tasksMaxDuration << " seconds." << std::endl;
    }

    MPI::Finalize();
    return 0;
}