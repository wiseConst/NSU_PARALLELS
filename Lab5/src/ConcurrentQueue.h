#ifndef CONCURRENTQUEUE_H
#define CONCURRENTQUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T> class ConcurrentQueue
{
  public:
    explicit ConcurrentQueue() = default;
    ~ConcurrentQueue()         = default;

    [[nodiscard]] bool IsEmpty() const
    {
        std::scoped_lock lock(m_QueueMutex);
        return m_Queue.empty();
    }

    void Push(const T& value)
    {
        {
            std::unique_lock lock(m_QueueMutex);
            m_Queue.emplace(value);
        }

        m_CondVar.notify_one();
    }

    size_t Size() const
    {
        std::scoped_lock lock(m_QueueMutex);
        return m_Queue.size();
    }

    bool TryPop(T& outValue)
    {
        std::scoped_lock lock(m_QueueMutex);
        if (m_Queue.empty())
        {
            outValue = 0;
            return false;
        }

        outValue = m_Queue.back();
        m_Queue.pop();
        return true;
    }

    void Pop()
    {
        std::unique_lock lock(m_QueueMutex);
        m_CondVar.wait(lock, [&]{ return !m_Queue.empty(); });  // Get out of wait-loop when queue is not empty.

        m_Queue.pop();
    }

    const T& Top() const
    {
        std::unique_lock lock(m_QueueMutex);
        m_CondVar.wait(lock, [this]  { return !m_Queue.empty(); });  // Get out of wait-loop when queue is not empty.

        return m_Queue.back();
    }

  private:
    std::queue<T> m_Queue                     = {};
    mutable std::mutex m_QueueMutex           = {};
    mutable std::condition_variable m_CondVar = {};
};

#endif
