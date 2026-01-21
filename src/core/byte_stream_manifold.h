#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "qfh.h"

namespace sep {

struct ByteStreamConfig {
    size_t window_bits = 256;            // Sliding window length in bits.
    size_t step_bits = 64;               // Step between consecutive windows in bits.
    size_t max_windows = 0;              // Optional cap on number of windows to analyse (0 = unlimited).
    bool lsb_first = true;               // Interpret bytes as LSB-first when converting to bits.
    size_t repetition_lookback = 0;      // Optional lookback (in windows) for repetition counts (0 = keep all).
    sep::quantum::QFHOptions qfh_options{};  // Tunable QFH parameters.
    int signature_precision = 2;         // Decimal places for signature bucketing.
};

struct ByteStreamWindow {
    size_t index = 0;
    size_t offset_bits = 0;
    size_t offset_bytes = 0;
    double coherence = 0.0;
    double stability = 0.0;
    double entropy = 0.0;
    double rupture = 0.0;
    double hazard_lambda = 0.0;
    bool collapse_detected = false;
    std::string signature;
    uint64_t repetition_count = 0;
};

struct ByteStreamSummary {
    size_t total_windows = 0;
    size_t window_bits = 0;
    size_t step_bits = 0;
    size_t analysed_bits = 0;
    size_t analysed_bytes = 0;
    double mean_coherence = 0.0;
    double mean_stability = 0.0;
    double mean_entropy = 0.0;
    double mean_rupture = 0.0;
    double mean_lambda = 0.0;
    double max_lambda = 0.0;
    uint64_t max_repetition = 0;
};

struct ByteStreamManifold {
    ByteStreamSummary summary;
    std::vector<ByteStreamWindow> windows;
    std::map<uint64_t, uint64_t> repetition_histogram;

    [[nodiscard]] nlohmann::json to_json(const ByteStreamConfig& config) const;
    [[nodiscard]] std::string to_csv() const;
};

// Loaders for various byte sources.
[[nodiscard]] std::vector<uint8_t> load_bytes_from_file(const std::string& path, size_t max_bytes = 0);
[[nodiscard]] std::vector<uint8_t> load_bytes_from_fd(int fd, size_t max_bytes = 0);
[[nodiscard]] std::vector<uint8_t> load_bytes_from_buffer(const uint8_t* data, size_t length);

// Core analysis entry point.
ByteStreamManifold analyze_byte_stream(const std::vector<uint8_t>& bytes, const ByteStreamConfig& config);

}  // namespace sep
