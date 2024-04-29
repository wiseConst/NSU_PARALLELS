#ifndef COREUTILS_H
#define COREUTILS_H

#include <chrono>

class Timer final
{
public:
    Timer() noexcept = default;
    ~Timer() = default;

    double GetElapsedSeconds() const { return GetElapsedMilliseconds() / 1000; }

    double GetElapsedMilliseconds() const
    {
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - m_StartTime);
        return elapsed.count();
    }

    static std::chrono::time_point<std::chrono::high_resolution_clock> Now()
    {
        return std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTime = Now();
};

static std::vector<float> LoadData(const std::string_view filePath)
{
    std::ifstream file(filePath.data(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        printf("Failed to open file \"%s\"!\n", filePath.data());
        return std::vector<float>{};
    }

    const auto fileSize = file.tellg();
    std::vector<float> buffer = {};
    buffer.resize(fileSize / sizeof(buffer[0]));

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

    file.close();
    return buffer;
}

static void SaveData(const std::string_view filePath, const void* data, const size_t dataSize)
{
    std::ofstream out(filePath.data(), std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        printf("Failed to open file \"%s\"!\n", filePath.data());
        return;
    }

    out.write(static_cast<const char*>(data), dataSize);
    out.close();
}


#endif //COREUTILS_H
