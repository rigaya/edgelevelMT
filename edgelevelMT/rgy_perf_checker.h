#pragma once

#include <array>
#include <string>
#include <chrono>
#include <cstdio>

template<bool enabled, size_t counterNum>
class RGYPerfChecker {
protected:
    std::array<std::string, counterNum> counterNames;
    std::array<std::chrono::high_resolution_clock::duration, counterNum> counters;
    std::array<std::chrono::high_resolution_clock::time_point, counterNum+1> timepoints;
    int64_t counts;
public:
    RGYPerfChecker(const std::array<std::string, counterNum>& names) : counterNames(names), counters(), timepoints(), counts(0) {
    };
    ~RGYPerfChecker() {};

    void settime(size_t tsIdx) {
        if (!enabled) return;
        timepoints[tsIdx] = std::chrono::high_resolution_clock::now();
    }
    void setcounter() {
        if (!enabled) return;
        for (size_t i = 1; i < timepoints.size(); i++) {
            counters[i-1] += (timepoints[i] - timepoints[i-1]);
        }
        counts++;
    }
    void print(const char *filename, int period = 1) {
        if (!enabled) return;
        if (period <= 0) period = 1;
        if (counts == 0 || ((counts % period) != 0)) return;
        FILE *fp = fopen(filename, "a");
        if (!fp) return;

        char buffer[4096];
        int len = sprintf_s(buffer, "%8lld, ", (long long int)counts);
        for (size_t i = 0; i < counters.size(); i++) {
            double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(counters[i]).count() * 0.001 / (double)counts;
            len += sprintf_s(buffer + len, _countof(buffer) - len - 1, "%.3e, ", time_ms);
        }
        fprintf(fp, "%s\n", buffer);
        fclose(fp);
    }
};


