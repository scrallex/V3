#include "manifold_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "qfh.h"

namespace sep {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

double compute_coherence_tau(const std::vector<uint8_t>& bits, size_t tau) {
    if (tau == 0 || bits.size() <= tau) {
        return 1.0;
    }
    size_t matches = 0;
    size_t comparisons = 0;
    for (size_t idx = tau; idx < bits.size(); ++idx) {
        ++comparisons;
        if (bits[idx] == bits[idx - tau]) {
            ++matches;
        }
    }
    if (comparisons == 0) {
        return 1.0;
    }
    return static_cast<double>(matches) / static_cast<double>(comparisons);
}

double domain_wall_ratio(const std::vector<uint8_t>& bits, size_t begin, size_t end) {
    if (end <= begin + 1) {
        return 0.0;
    }
    size_t transitions = 0;
    for (size_t idx = begin + 1; idx < end; ++idx) {
        if (bits[idx] != bits[idx - 1]) {
            ++transitions;
        }
    }
    const size_t pairs = (end - begin) - 1;
    if (pairs == 0) {
        return 0.0;
    }
    return static_cast<double>(transitions) / static_cast<double>(pairs);
}

struct DomainWallStats {
    double ratio = 0.0;
    double slope = 0.0;
};

DomainWallStats compute_domain_wall_stats(const std::vector<uint8_t>& bits) {
    DomainWallStats stats{};
    if (bits.size() < 3) {
        stats.ratio = domain_wall_ratio(bits, 0, bits.size());
        stats.slope = 0.0;
        return stats;
    }
    stats.ratio = domain_wall_ratio(bits, 0, bits.size());
    const size_t mid = bits.size() / 2;
    const double first = domain_wall_ratio(bits, 0, mid);
    const double second = domain_wall_ratio(bits, mid, bits.size());
    stats.slope = second - first;
    return stats;
}

double compute_low_frequency_share(const std::vector<uint8_t>& bits) {
    const size_t n = bits.size();
    if (n < 4) {
        return 0.0;
    }
    std::vector<double> centred(n);
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) {
        centred[i] = static_cast<double>(bits[i]);
        mean += centred[i];
    }
    mean /= static_cast<double>(n);
    for (double& value : centred) {
        value -= mean;
    }

    const size_t max_k = n / 2;
    if (max_k == 0) {
        return 0.0;
    }
    const size_t low_cut = std::max<size_t>(1, max_k / 4);
    const double two_pi_over_n = (2.0 * kPi) / static_cast<double>(n);

    double total_power = 0.0;
    double low_power = 0.0;
    for (size_t k = 1; k <= max_k; ++k) {
        double real = 0.0;
        double imag = 0.0;
        for (size_t t = 0; t < n; ++t) {
            const double angle = two_pi_over_n * static_cast<double>(k * t);
            const double value = centred[t];
            real += value * std::cos(angle);
            imag += value * std::sin(angle);
        }
        const double power = real * real + imag * imag;
        total_power += power;
        if (k <= low_cut) {
            low_power += power;
        }
    }
    if (total_power <= 1e-12) {
        return 0.0;
    }
    const double share = low_power / total_power;
    return std::clamp(share, 0.0, 1.0);
}

}  // namespace

nlohmann::json buildManifold(const std::vector<Candle>& candles, const std::string& instrument_hint) {
    nlohmann::json result;
    const std::string instrument = instrument_hint.empty() ? "UNKNOWN" : instrument_hint;
    result["instrument"] = instrument;

    if (candles.empty()) {
        result["count"] = 0;
        result["signals"] = nlohmann::json::array();
        return result;
    }

    const uint64_t t0 = candles.front().timestamp;
    const uint64_t t1 = candles.back().timestamp;

    // Build enriched bitstream reflecting momentum, volatility, and volume dynamics.
    std::vector<uint8_t> bits;
    bits.reserve(candles.size() > 1 ? (candles.size() - 1) : 0);
    for (size_t i = 1; i < candles.size(); ++i) {
        const auto& prev = candles[i - 1];
        const auto& curr = candles[i];

        const bool price_up = curr.close >= prev.close;
        const bool range_expanding = (curr.high - curr.low) >= (prev.high - prev.low);
        const bool volume_increasing = curr.volume >= prev.volume;

        uint8_t bit_value = price_up ? 1 : 0;
        if (!range_expanding && !volume_increasing) {
            bit_value = 0;  // Quiet regime dampens the signal.
        }

        bits.push_back(bit_value);
    }

    const size_t max_signals = 512;
    nlohmann::json signals = nlohmann::json::array();

    static const double signature_precision = []() {
        if (const char* env = std::getenv("ECHO_SIGNATURE_PRECISION")) {
            try {
                return std::max(0.0, std::min(6.0, std::stod(env)));
            } catch (...) {
                return 2.0;
            }
        }
        return 2.0;
    }();

    const double scale = std::pow(10.0, signature_precision);
    auto bucket = [&](double value) {
        const double clamped = std::clamp(value, 0.0, 1.0);
        return std::round(clamped * scale) / scale;
    };

    auto make_signature = [&](double c, double s, double e) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(static_cast<int>(signature_precision));
        oss << "c" << bucket(c) << "_s" << bucket(s) << "_e" << bucket(e);
        return oss.str();
    };

    static const uint64_t repetition_window_ms = []() -> uint64_t {
        if (const char* env = std::getenv("ECHO_LOOKBACK_MINUTES")) {
            try {
                double minutes = std::stod(env);
                minutes = std::clamp(minutes, 1.0, 1440.0);
                return static_cast<uint64_t>(minutes * 60.0 * 1000.0);
            } catch (...) {
                return static_cast<uint64_t>(60ULL * 60ULL * 1000ULL);
            }
        }
        return static_cast<uint64_t>(60ULL * 60ULL * 1000ULL);
    }();

    std::unordered_map<std::string, std::deque<uint64_t>> repetition_history;

    if (!bits.empty()) {
        sep::quantum::QFHOptions opts{};
        sep::quantum::QFHBasedProcessor proc(opts);

        size_t window = bits.size();
        if (window > 128) {
            window = 128;
        }
        if (window >= bits.size()) {
            if (bits.size() > 24) {
                window = bits.size() - std::min<size_t>(8, bits.size() / 4);
            } else if (bits.size() > 12) {
                window = 12;
            } else if (bits.size() > 8) {
                window = 9;
            } else {
                window = bits.size();
            }
        }

        if (window >= 8 && bits.size() >= 2) {
            size_t start_i = window;
            if (bits.size() - window > max_signals) {
                start_i = bits.size() - max_signals;
            }

            const size_t step = std::max<size_t>(1, window / 32);
            for (size_t i = start_i; i <= bits.size(); i += step) {
                const size_t begin = i - window;
                if (begin >= bits.size()) break;

                std::vector<uint8_t> sub(bits.begin() + begin, bits.begin() + i);
                const sep::quantum::QFHResult r = proc.analyze(sub);

                const uint64_t ts_ms = candles[begin + 1].timestamp;
                const double price = candles[begin + 1].close;
                const double coherence = static_cast<double>(r.coherence);
                const double stability = 1.0 - static_cast<double>(r.rupture_ratio);
                const double entropy = static_cast<double>(r.entropy);
                const double rupture = static_cast<double>(r.rupture_ratio);

                const double coh_tau_1 = compute_coherence_tau(sub, 1);
                const double coh_tau_4 = compute_coherence_tau(sub, 4);
                const double coh_tau_slope = (coh_tau_4 - coh_tau_1) / 3.0;
                const DomainWallStats wall_stats = compute_domain_wall_stats(sub);
                const double spectral_low_share = compute_low_frequency_share(sub);
                const double reynolds_ratio = std::fabs(wall_stats.slope) > 1e-6
                                                 ? std::fabs(rupture) /
                                                   std::max(1e-6, std::fabs(wall_stats.slope))
                                                 : 0.0;
                double temporal_half_life = 0.0;
                if (std::fabs(coh_tau_slope) > 1e-6) {
                    temporal_half_life = std::log(2.0) /
                                         std::max(1e-6, std::fabs(coh_tau_slope));
                }
                const double spatial_corr_length = std::fabs(wall_stats.slope) > 1e-6
                                                      ? 1.0 /
                                                        std::max(1e-6, std::fabs(wall_stats.slope))
                                                      : 0.0;
                const double pinned_alignment = 1.0;

                const std::string signature = make_signature(
                    std::clamp(coherence, 0.0, 1.0),
                    std::clamp(stability, 0.0, 1.0),
                    std::clamp(entropy, 0.0, 1.0)
                );
                auto& history = repetition_history[signature];
                while (!history.empty() && (ts_ms > history.front()) && (ts_ms - history.front() > repetition_window_ms)) {
                    history.pop_front();
                }
                history.push_back(ts_ms);
                const uint64_t first_seen = history.front();
                const uint64_t count = history.size();
                const double hazard_lambda = std::clamp(rupture, 0.0, 1.0);

                nlohmann::json signal = {
                    {"timestamp_ns", ts_ms * 1000000ULL},
                    {"price", price},
                    {"state", r.collapse_detected ? "collapsed" : "live"},
                    {"metrics", {
                        {"coherence", coherence},
                        {"stability", stability},
                        {"entropy", entropy},
                        {"rupture", rupture},
                        {"coherence_tau_1", coh_tau_1},
                        {"coherence_tau_4", coh_tau_4},
                        {"coherence_tau_slope", coh_tau_slope},
                        {"domain_wall_ratio", wall_stats.ratio},
                        {"domain_wall_slope", wall_stats.slope},
                        {"spectral_lowf_share", spectral_low_share},
                        {"reynolds_ratio", reynolds_ratio},
                        {"temporal_half_life", temporal_half_life},
                        {"spatial_corr_length", spatial_corr_length},
                        {"pinned_alignment", pinned_alignment}
                    }},
                    {"coeffs", {
                        {"lambda_hazard", hazard_lambda}
                    }},
                    {"repetition", {
                        {"signature", signature},
                        {"count_1h", static_cast<uint64_t>(count)},
                        {"first_seen_ms", first_seen}
                    }},
                    {"coherence", coherence},
                    {"stability", stability},
                    {"entropy", entropy},
                    {"rupture", rupture},
                    {"lambda_hazard", hazard_lambda},
                    {"coherence_tau_slope", coh_tau_slope},
                    {"domain_wall_ratio", wall_stats.ratio},
                    {"domain_wall_slope", wall_stats.slope},
                    {"spectral_lowf_share", spectral_low_share},
                    {"reynolds_ratio", reynolds_ratio},
                    {"temporal_half_life", temporal_half_life},
                    {"spatial_corr_length", spatial_corr_length},
                    {"pinned_alignment", pinned_alignment}
                };
                signals.push_back(std::move(signal));
            }
        }
    }
    result["count"] = static_cast<uint64_t>(candles.size());
    result["t0_ms"] = t0;
    result["t1_ms"] = t1;
    result["signals"] = std::move(signals);

    try {
        double coh = 0.0;
        double stab = 0.0;
        double ent = 0.0;
        double rup = 0.0;
        double coh_tau_slope = 0.0;
        double coh_tau_1 = 0.0;
        double coh_tau_4 = 0.0;
        double wall_ratio = 0.0;
        double wall_slope = 0.0;
        double spectral_low = 0.0;
        double reynolds = 0.0;
        double half_life = 0.0;
        double spatial_length = 0.0;
        double pinned_align = 1.0;
        if (result.contains("signals") && result["signals"].is_array() && !result["signals"].empty()) {
            const auto& s = result["signals"].back();
            if (s.contains("metrics")) {
                const auto& m = s["metrics"];
                coh = m.value("coherence", s.value("coherence", 0.0));
                stab = m.value("stability", s.value("stability", 0.0));
                ent = m.value("entropy", s.value("entropy", 0.0));
                rup = m.value("rupture", s.value("rupture", 0.0));
                coh_tau_slope = m.value("coherence_tau_slope", s.value("coherence_tau_slope", 0.0));
                coh_tau_1 = m.value("coherence_tau_1", s.value("coherence_tau_1", 0.0));
                coh_tau_4 = m.value("coherence_tau_4", s.value("coherence_tau_4", 0.0));
                wall_ratio = m.value("domain_wall_ratio", s.value("domain_wall_ratio", 0.0));
                wall_slope = m.value("domain_wall_slope", s.value("domain_wall_slope", 0.0));
                spectral_low = m.value("spectral_lowf_share", s.value("spectral_lowf_share", 0.0));
                reynolds = m.value("reynolds_ratio", s.value("reynolds_ratio", 0.0));
                half_life = m.value("temporal_half_life", s.value("temporal_half_life", 0.0));
                spatial_length = m.value("spatial_corr_length", s.value("spatial_corr_length", 0.0));
                pinned_align = m.value("pinned_alignment", s.value("pinned_alignment", 1.0));
            } else {
                coh = s.value("coherence", 0.0);
                stab = s.value("stability", 0.0);
                ent = s.value("entropy", 0.0);
                rup = s.value("rupture", 0.0);
                coh_tau_slope = s.value("coherence_tau_slope", 0.0);
                coh_tau_1 = s.value("coherence_tau_1", 0.0);
                coh_tau_4 = s.value("coherence_tau_4", 0.0);
                wall_ratio = s.value("domain_wall_ratio", 0.0);
                wall_slope = s.value("domain_wall_slope", 0.0);
                spectral_low = s.value("spectral_lowf_share", 0.0);
                reynolds = s.value("reynolds_ratio", 0.0);
                half_life = s.value("temporal_half_life", 0.0);
                spatial_length = s.value("spatial_corr_length", 0.0);
                pinned_align = s.value("pinned_alignment", 1.0);
            }
        }
        result["metrics"] = {
            {"coherence", coh},
            {"stability", stab},
            {"entropy", ent},
            {"rupture", rup},
            {"coherence_tau_slope", coh_tau_slope},
            {"coherence_tau_1", coh_tau_1},
            {"coherence_tau_4", coh_tau_4},
            {"domain_wall_ratio", wall_ratio},
            {"domain_wall_slope", wall_slope},
            {"spectral_lowf_share", spectral_low},
            {"reynolds_ratio", reynolds},
            {"temporal_half_life", half_life},
            {"spatial_corr_length", spatial_length},
            {"pinned_alignment", pinned_align}
        };
    } catch (...) {
        // Leave metrics absent on failure – downstream consumers fall back gracefully.
    }

    try {
        double sigma_eff = 0.0;
        if (candles.size() >= 3) {
            std::vector<double> rets;
            rets.reserve(candles.size() - 1);
            for (size_t i = 1; i < candles.size(); ++i) {
                const double c1 = candles[i - 1].close;
                const double c2 = candles[i].close;
                if (c1 > 0.0 && c2 > 0.0) {
                    rets.push_back(std::log(c2 / c1));
                }
            }
            if (rets.size() >= 2) {
                double mean = 0.0;
                for (double x : rets) mean += x;
                mean /= static_cast<double>(rets.size());

                double var = 0.0;
                for (double x : rets) {
                    const double d = x - mean;
                    var += d * d;
                }
                var /= static_cast<double>(rets.size() - 1);
                sigma_eff = std::sqrt(std::max(0.0, var));
            }
        }

        double lambda_pmin = 0.0;
        double t_sum_sec = 0.0;
        double r_sum = 0.0;
        if (result.contains("signals") && result["signals"].is_array() && result["signals"].size() >= 2) {
            const auto& arr = result["signals"];
            for (size_t i = 1; i < arr.size(); ++i) {
                double r = 0.0;
                if (arr[i].contains("metrics")) {
                    r = arr[i]["metrics"].value("rupture", arr[i].value("rupture", 0.0));
                } else {
                    r = arr[i].value("rupture", 0.0);
                }
                const double t_i = static_cast<double>(arr[i].value("timestamp_ns", 0ULL)) / 1e9;
                const double t_j = static_cast<double>(arr[i - 1].value("timestamp_ns", 0ULL)) / 1e9;
                const double dt = std::max(1e-6, t_i - t_j);
                r_sum += r;
                t_sum_sec += dt;
            }
            if (t_sum_sec > 0.0) {
                const double lambda_per_sec = r_sum / t_sum_sec;
                lambda_pmin = lambda_per_sec * 60.0;
            }
        }

        const double lambda_prob = 1.0 - std::exp(-std::max(0.0, lambda_pmin));
        result["coeffs"] = {
            {"sigma_eff", sigma_eff},
            {"lambda", std::max(0.0, std::min(1.0, lambda_prob))}
        };
    } catch (...) {
        // Leave coeffs absent on failure.
    }

    return result;
}

}  // namespace sep
