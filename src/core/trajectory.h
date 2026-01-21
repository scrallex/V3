// Trajectory utilities and damped integral result for forward projections
// Matches usage in QFH/QBSA and aligns with docs on exponential damping

#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include <nlohmann/json.hpp>

namespace sep {
namespace quantum {
namespace bitspace {

// Result of integrating a future trajectory with exponential damping
struct DampedValue {
    double final_value{0.0};     // Σ (future - current) * exp(-lambda * dt)
    double confidence{0.0};      // [0,1] confidence from path consistency
    bool   converged{false};     // heuristic flag based on stability
    std::vector<double> path;    // optional path of partial sums for diagnostics

    // Optional diagnostics (not required by current callers)
    double lambda{0.0};          // effective decay used (if known)
    std::size_t start_index{0};  // index in source series where integration started

    inline nlohmann::json toJson() const {
        return {
            {"final_value", final_value},
            {"confidence", confidence},
            {"converged", converged},
            {"path", path},
            {"lambda", lambda},
            {"start_index", start_index}
        };
    }

    static inline DampedValue fromJson(const nlohmann::json& j) {
        DampedValue dv;
        if (j.contains("final_value")) dv.final_value = j.at("final_value").get<double>();
        if (j.contains("confidence")) dv.confidence = j.at("confidence").get<double>();
        if (j.contains("converged")) dv.converged = j.at("converged").get<bool>();
        if (j.contains("path")) dv.path = j.at("path").get<std::vector<double>>();
        if (j.contains("lambda")) dv.lambda = j.at("lambda").get<double>();
        if (j.contains("start_index")) dv.start_index = j.at("start_index").get<std::size_t>();
        return dv;
    }
};

} // namespace bitspace
} // namespace quantum
} // namespace sep

