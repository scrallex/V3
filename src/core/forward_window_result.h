// Forward window result model used across QFH/Q‑Chain and serving layers
// Aligns with docs: strands carry hazard/t_remain and survival summaries

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "trajectory.h"

namespace sep {
namespace quantum {
namespace bitspace {

// Status of a strand/chain within the forward window
enum class StrandStatus {
    Live,
    Closed
};

// Compact survival summary matching REST/WS schema
struct SurvivalSummary {
    double h1{0.0};       // one-step hazard = 1 - sum_j Q_ij
    double tRemain{0.0};  // expected remaining steps (E[T])
    double S1{0.0};       // survival probability at 60s (or 1 min bucket)
    double S5{0.0};       // survival probability at 5 min
    double S15{0.0};      // survival probability at 15 min
    double tVar{0.0};     // variance of time to absorption

    inline nlohmann::json toJson() const {
        return {
            {"h1", h1},
            {"tRemain", tRemain},
            {"S1", S1},
            {"S5", S5},
            {"S15", S15},
            {"tVar", tVar}
        };
    }

    static inline SurvivalSummary fromJson(const nlohmann::json& j) {
        SurvivalSummary s;
        if (j.contains("h1")) s.h1 = j.at("h1").get<double>();
        if (j.contains("tRemain")) s.tRemain = j.at("tRemain").get<double>();
        if (j.contains("S1")) s.S1 = j.at("S1").get<double>();
        if (j.contains("S5")) s.S5 = j.at("S5").get<double>();
        if (j.contains("S15")) s.S15 = j.at("S15").get<double>();
        if (j.contains("tVar")) s.tVar = j.at("tVar").get<double>();
        return s;
    }
};

struct ForwardWindowResult {
    // Identity and window bounds (epoch millis)
    std::string id;        // optional unique strand id
    std::string instrument; // optional instrument/symbol tag
    std::uint64_t t0{0};
    std::uint64_t t_last{0};

    // Indices into the working sequence (optional; for internal use)
    std::size_t start_index{0};
    std::size_t end_index{0};

    // Current observables over the window
    double c_now{0.0};     // coherence in [0,1]
    double H_now{0.0};     // entropy in [0,1]
    double stability_now{0.0}; // stability proxy in [0,1]

    // Absorbing chain state (see docs/BIBLE.md)
    std::uint32_t current_bin{0}; // coherence bin index in {0..K-1}
    double hazard_now{0.0};       // h_i(1) = 1 - sum_j Q_ij
    double t_remain{0.0};         // expected remaining steps E[T]
    double t_var{0.0};            // Var[T]
    SurvivalSummary survival{};    // calibrated survival snapshot

    // Optional diagnostics: collection of integrated trajectories
    std::vector<DampedValue> damped_values;

    // Event mix (optional diagnostics)
    std::uint32_t null_state_count{0};
    std::uint32_t flip_count{0};
    std::uint32_t rupture_count{0};

    StrandStatus status{StrandStatus::Live};

    // Convenience: JSON projection used by REST/WS manifold payloads
    inline nlohmann::json toJson() const {
        nlohmann::json j;
        if (!id.empty()) j["id"] = id;
        if (!instrument.empty()) j["instrument"] = instrument;
        j["t0"] = t0;
        j["t_last"] = t_last;
        j["c_now"] = c_now;
        j["H_now"] = H_now;
        j["stability_now"] = stability_now;
        j["current_bin"] = current_bin;
        j["hazard_now"] = hazard_now;
        j["t_remain"] = t_remain;
        j["tVar"] = t_var;
        j["survival"] = survival.toJson();
        j["status"] = (status == StrandStatus::Live) ? "live" : "closed";
        if (!damped_values.empty()) {
            nlohmann::json arr = nlohmann::json::array();
            for (const auto& dv : damped_values) arr.push_back(dv.toJson());
            j["damped_values"] = std::move(arr);
        }
        return j;
    }

    static inline ForwardWindowResult fromJson(const nlohmann::json& j) {
        ForwardWindowResult r;
        if (j.contains("id")) r.id = j.at("id").get<std::string>();
        if (j.contains("instrument")) r.instrument = j.at("instrument").get<std::string>();
        if (j.contains("t0")) r.t0 = j.at("t0").get<std::uint64_t>();
        if (j.contains("t_last")) r.t_last = j.at("t_last").get<std::uint64_t>();
        if (j.contains("c_now")) r.c_now = j.at("c_now").get<double>();
        if (j.contains("H_now")) r.H_now = j.at("H_now").get<double>();
        if (j.contains("stability_now")) r.stability_now = j.at("stability_now").get<double>();
        if (j.contains("current_bin")) r.current_bin = j.at("current_bin").get<std::uint32_t>();
        if (j.contains("hazard_now")) r.hazard_now = j.at("hazard_now").get<double>();
        if (j.contains("t_remain")) r.t_remain = j.at("t_remain").get<double>();
        if (j.contains("tVar")) r.t_var = j.at("tVar").get<double>();
        if (j.contains("survival")) r.survival = SurvivalSummary::fromJson(j.at("survival"));
        if (j.contains("status")) {
            std::string s = j.at("status").get<std::string>();
            r.status = (s == "closed") ? StrandStatus::Closed : StrandStatus::Live;
        }
        if (j.contains("damped_values")) {
            for (const auto& e : j.at("damped_values")) {
                r.damped_values.push_back(DampedValue::fromJson(e));
            }
        }
        return r;
    }
};

} // namespace bitspace
} // namespace quantum
} // namespace sep
