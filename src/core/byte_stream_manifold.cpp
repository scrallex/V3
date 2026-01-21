#include "byte_stream_manifold.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "standard_includes.h"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace sep {

namespace {

constexpr double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

std::vector<uint8_t> bytes_to_bits(const std::vector<uint8_t>& bytes, bool lsb_first) {
    std::vector<uint8_t> bits;
    bits.reserve(bytes.size() * 8);
    if (lsb_first) {
        for (uint8_t byte : bytes) {
            for (int i = 0; i < 8; ++i) {
                bits.push_back(static_cast<uint8_t>((byte >> i) & 0x1U));
            }
        }
    } else {
        for (uint8_t byte : bytes) {
            for (int i = 7; i >= 0; --i) {
                bits.push_back(static_cast<uint8_t>((byte >> i) & 0x1U));
            }
        }
    }
    return bits;
}

double compute_lambda(double entropy, double coherence, const sep::quantum::QFHOptions& opts) {
    const double weighted = opts.entropy_weight * clamp01(entropy) + opts.coherence_weight * (1.0 - clamp01(coherence));
    return clamp01(weighted);
}

std::string make_signature(double coherence, double stability, double entropy, int precision) {
    const double scale = std::pow(10.0, std::clamp(precision, 0, 6));
    auto bucket = [scale](double value) {
        const double clamped = clamp01(value);
        return std::round(clamped * scale) / scale;
    };

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(std::clamp(precision, 0, 6));
    oss << "c" << bucket(coherence)
        << "_s" << bucket(stability)
        << "_e" << bucket(entropy);
    return oss.str();
}

struct Accumulators {
    double coherence = 0.0;
    double stability = 0.0;
    double entropy = 0.0;
    double rupture = 0.0;
    double lambda = 0.0;

    void add(const ByteStreamWindow& window) {
        coherence += window.coherence;
        stability += window.stability;
        entropy += window.entropy;
        rupture += window.rupture;
        lambda += window.hazard_lambda;
    }

    [[nodiscard]] ByteStreamSummary finalize(const ByteStreamSummary& base) const {
        ByteStreamSummary summary = base;
        if (base.total_windows == 0) {
            return summary;
        }
        const double count = static_cast<double>(base.total_windows);
        summary.mean_coherence = coherence / count;
        summary.mean_stability = stability / count;
        summary.mean_entropy = entropy / count;
        summary.mean_rupture = rupture / count;
        summary.mean_lambda = lambda / count;
        return summary;
    }
};

}  // namespace

std::vector<uint8_t> load_bytes_from_file(const std::string& path, size_t max_bytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) {
        throw std::runtime_error("Failed to open byte stream file: " + path);
    }
    ifs.seekg(0, std::ios::end);
    std::streampos end_pos = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    size_t to_read = static_cast<size_t>(std::max<std::streamoff>(0, end_pos));
    if (max_bytes > 0 && max_bytes < to_read) {
        to_read = max_bytes;
    }

    std::vector<uint8_t> bytes(to_read);
    if (to_read > 0) {
        ifs.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(to_read));
    }
    bytes.resize(static_cast<size_t>(ifs.gcount()));
    return bytes;
}

std::vector<uint8_t> load_bytes_from_fd(int fd, size_t max_bytes) {
    if (fd < 0) {
        throw std::invalid_argument("File descriptor must be non-negative");
    }

    constexpr size_t kChunkSize = 4096;
    std::vector<uint8_t> bytes;
    bytes.reserve(max_bytes > 0 ? max_bytes : kChunkSize);

    size_t remaining = max_bytes;
    std::array<uint8_t, kChunkSize> buffer{};
    while (true) {
        size_t request = kChunkSize;
        if (max_bytes > 0) {
            if (remaining == 0) break;
            request = std::min(request, remaining);
        }
#ifdef _WIN32
        int read_count = _read(fd, buffer.data(), static_cast<unsigned int>(request));
#else
        ssize_t read_count = ::read(fd, buffer.data(), request);
#endif
        if (read_count <= 0) {
            break;
        }
        bytes.insert(bytes.end(), buffer.begin(), buffer.begin() + static_cast<size_t>(read_count));
        if (max_bytes > 0) {
            remaining -= static_cast<size_t>(read_count);
        }
    }
    return bytes;
}

std::vector<uint8_t> load_bytes_from_buffer(const uint8_t* data, size_t length) {
    if (!data || length == 0) {
        return {};
    }
    return std::vector<uint8_t>(data, data + length);
}

ByteStreamManifold analyze_byte_stream(const std::vector<uint8_t>& bytes, const ByteStreamConfig& config) {
    ByteStreamManifold manifold;
    manifold.summary.window_bits = config.window_bits;
    manifold.summary.step_bits = std::max<size_t>(1, config.step_bits);
    manifold.summary.analysed_bytes = bytes.size();

    if (bytes.empty() || config.window_bits == 0) {
        manifold.summary.analysed_bits = bytes.size() * 8;
        return manifold;
    }

    const auto bits = bytes_to_bits(bytes, config.lsb_first);
    manifold.summary.analysed_bits = bits.size();

    if (bits.size() < config.window_bits) {
        return manifold;
    }

    const size_t step_bits = manifold.summary.step_bits;
    const size_t window_bits = std::max<size_t>(1, config.window_bits);
    const size_t max_windows = config.max_windows;

    sep::quantum::QFHBasedProcessor processor(config.qfh_options);
    std::unordered_map<std::string, std::deque<size_t>> repetition_history;
    Accumulators accumulators;

    size_t window_index = 0;
    for (size_t start = 0; start + window_bits <= bits.size(); start += step_bits) {
        if (max_windows > 0 && window_index >= max_windows) {
            break;
        }

        std::vector<uint8_t> window_bits_vec(bits.begin() + static_cast<std::ptrdiff_t>(start),
                                             bits.begin() + static_cast<std::ptrdiff_t>(start + window_bits));

        const sep::quantum::QFHResult result = processor.analyze(window_bits_vec);
        const double coherence = clamp01(result.coherence);
        const double stability = clamp01(result.stability);
        const double entropy = clamp01(result.entropy);
        const double rupture = clamp01(result.rupture_ratio);
        const double lambda_hazard = compute_lambda(entropy, coherence, config.qfh_options);
        const bool collapsed = result.collapse_detected;
        const std::string signature = make_signature(coherence, stability, entropy, config.signature_precision);

        auto& history = repetition_history[signature];
        if (config.repetition_lookback > 0) {
            while (!history.empty() && (window_index - history.front() >= config.repetition_lookback)) {
                history.pop_front();
            }
        }
        history.push_back(window_index);
        const uint64_t repetition_count = static_cast<uint64_t>(history.size());

        ByteStreamWindow window{};
        window.index = window_index;
        window.offset_bits = start;
        window.offset_bytes = start / 8;
        window.coherence = coherence;
        window.stability = stability;
        window.entropy = entropy;
        window.rupture = rupture;
        window.hazard_lambda = lambda_hazard;
        window.collapse_detected = collapsed;
        window.signature = signature;
        window.repetition_count = repetition_count;

        manifold.windows.push_back(window);
        accumulators.add(window);
        manifold.summary.max_lambda = std::max(manifold.summary.max_lambda, window.hazard_lambda);
        manifold.summary.max_repetition = std::max(manifold.summary.max_repetition, window.repetition_count);
        manifold.repetition_histogram[window.repetition_count] += 1;

        ++window_index;
    }

    manifold.summary.total_windows = manifold.windows.size();
    manifold.summary = accumulators.finalize(manifold.summary);
    return manifold;
}

nlohmann::json ByteStreamManifold::to_json(const ByteStreamConfig& config) const {
    nlohmann::json root;
    root["config"] = {
        {"window_bits", config.window_bits},
        {"step_bits", std::max<size_t>(1, config.step_bits)},
        {"max_windows", config.max_windows},
        {"lsb_first", config.lsb_first},
        {"repetition_lookback", config.repetition_lookback},
        {"signature_precision", std::clamp(config.signature_precision, 0, 6)},
        {"qfh_options", {
            {"coherence_threshold", config.qfh_options.coherence_threshold},
            {"stability_threshold", config.qfh_options.stability_threshold},
            {"collapse_threshold", config.qfh_options.collapse_threshold},
            {"max_iterations", config.qfh_options.max_iterations},
            {"enable_damping", config.qfh_options.enable_damping},
            {"damping_factor", config.qfh_options.damping_factor},
            {"entropy_weight", config.qfh_options.entropy_weight},
            {"coherence_weight", config.qfh_options.coherence_weight}
        }}
    };

    root["summary"] = {
        {"total_windows", summary.total_windows},
        {"window_bits", summary.window_bits},
        {"step_bits", summary.step_bits},
        {"analysed_bits", summary.analysed_bits},
        {"analysed_bytes", summary.analysed_bytes},
        {"mean_coherence", summary.mean_coherence},
        {"mean_stability", summary.mean_stability},
        {"mean_entropy", summary.mean_entropy},
        {"mean_rupture", summary.mean_rupture},
        {"mean_lambda", summary.mean_lambda},
        {"max_lambda", summary.max_lambda},
        {"max_repetition", summary.max_repetition}
    };

    nlohmann::json windows_json = nlohmann::json::array();
    for (const auto& window : windows) {
        nlohmann::json item = {
            {"index", window.index},
            {"offset_bits", window.offset_bits},
            {"offset_bytes", window.offset_bytes},
            {"metrics", {
                {"coherence", window.coherence},
                {"stability", window.stability},
                {"entropy", window.entropy},
                {"rupture", window.rupture}
            }},
            {"lambda_hazard", window.hazard_lambda},
            {"signature", window.signature},
            {"state", window.collapse_detected ? "collapsed" : "live"},
            {"repetition", {
                {"count", window.repetition_count}
            }}
        };
        windows_json.push_back(std::move(item));
    }
    root["windows"] = std::move(windows_json);

    nlohmann::json histogram_json = nlohmann::json::object();
    for (const auto& [count, occurrences] : repetition_histogram) {
        histogram_json[std::to_string(count)] = occurrences;
    }
    root["histogram"] = {
        {"repetition", histogram_json}
    };

    return root;
}

std::string ByteStreamManifold::to_csv() const {
    std::ostringstream oss;
    oss << "index,offset_bits,offset_bytes,coherence,stability,entropy,rupture,lambda_hazard,signature,repetition_count,collapsed\n";
    oss << std::fixed << std::setprecision(6);
    for (const auto& window : windows) {
        oss << window.index << ','
            << window.offset_bits << ','
            << window.offset_bytes << ','
            << window.coherence << ','
            << window.stability << ','
            << window.entropy << ','
            << window.rupture << ','
            << window.hazard_lambda << ','
            << window.signature << ','
            << window.repetition_count << ','
            << (window.collapse_detected ? "true" : "false")
            << '\n';
    }
    return oss.str();
}

}  // namespace sep
