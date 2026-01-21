#include "qfh.h"
#include "core/standard_includes.h"
#include "trajectory.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include <optional>
#include <execution>
#include <atomic>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sep::quantum {

bool QFHEvent::operator==(const QFHEvent& other) const {
    return index == other.index && state == other.state &&
           bit_prev == other.bit_prev && bit_curr == other.bit_curr;
}

std::vector<QFHEvent> transform_rich(const std::vector<uint8_t>& bits)
{
    if (bits.size() < 2) return {};

    const std::size_t n = bits.size() - 1;
    std::vector<QFHEvent> result(n);

    // Index space [0, n)
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::atomic<bool> invalid{false};

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](std::size_t i){
        // Map i -> pair (i, i+1)
        uint8_t prev = bits[i];
        uint8_t curr = bits[i + 1];
        if ((prev != 0 && prev != 1) || (curr != 0 && curr != 1)) {
            invalid.store(true, std::memory_order_relaxed);
            return;
        }
        QFHState st = QFHState::NULL_STATE;
        if ((prev == 0 && curr == 1) || (prev == 1 && curr == 0)) {
            st = QFHState::FLIP;
        } else if (prev == 1 && curr == 1) {
            st = QFHState::RUPTURE;
        }
        result[i] = QFHEvent{static_cast<uint32_t>(i), st, prev, curr};
    });

    if (invalid.load(std::memory_order_relaxed)) return {};
    return result;
}
std::vector<QFHAggregateEvent> aggregate(const std::vector<QFHEvent>& events)
{
    if (events.empty()) {
        return {};
    }
    std::vector<QFHAggregateEvent> aggregated;
    aggregated.push_back({events[0].index, events[0].state, 1});
    for (size_t i = 1; i < events.size(); ++i) {
        if (events[i].state == aggregated.back().state) {
            aggregated.back().count++;
        } else {
            aggregated.push_back({events[i].index, events[i].state, 1});
        }
    }
    return aggregated;
}

std::optional<sep::quantum::QFHState> sep::quantum::QFHProcessor::process(uint8_t current_bit) {
    if (current_bit != 0 && current_bit != 1) {
        return std::nullopt;
    }
    if (!prev_bit.has_value()) {
        prev_bit = current_bit;
        return std::nullopt;
    }
    uint8_t prev = prev_bit.value();
    std::optional<sep::quantum::QFHState> event_state;
    if (prev == 0 && current_bit == 0) {
        event_state = sep::quantum::QFHState::NULL_STATE;
    } else if ((prev == 0 && current_bit == 1) || (prev == 1 && current_bit == 0)) {
        event_state = sep::quantum::QFHState::FLIP;
    } else if (prev == 1 && current_bit == 1) {
        event_state = sep::quantum::QFHState::RUPTURE;
    }
    prev_bit = current_bit;
    return event_state;
}

void sep::quantum::QFHProcessor::reset() {
    prev_bit.reset();
}

bitspace::DampedValue QFHBasedProcessor::integrateFutureTrajectories(const std::vector<uint8_t>& bitstream, size_t current_index) {
    bitspace::DampedValue damped_value;
    
    if (current_index >= bitstream.size()) {
        damped_value.final_value = 0.0;
        damped_value.confidence = 0.0;
        return damped_value;
    }
    
    // Step 1: Calculate dynamic decay factor λ based on local entropy and coherence
    // Analyze local pattern around current_index for entropy calculation
    size_t window_size = std::min(static_cast<size_t>(20), bitstream.size() - current_index);
    std::vector<uint8_t> local_window(bitstream.begin() + current_index, 
                                      bitstream.begin() + current_index + window_size);
    
    // Calculate local entropy and coherence directly to avoid recursion
    auto local_events = transform_rich(local_window);
    double local_entropy = 0.5; // Default entropy
    double local_coherence = 0.5; // Default coherence
    
    if (!local_events.empty()) {
        // Simple entropy calculation without recursion
        int null_count = 0, flip_count = 0, rupture_count = 0;
        int stable_count = 0, unstable_count = 0, collapsing_count = 0;
        int collapsed_count = 0, recovering_count = 0;
        for (const auto& event : local_events) {
            switch (event.state) {
                case QFHState::NULL_STATE: null_count++; break;
                case QFHState::STABLE: stable_count++; break;
                case QFHState::UNSTABLE: unstable_count++; break;
                case QFHState::COLLAPSING: collapsing_count++; break;
                case QFHState::COLLAPSED: collapsed_count++; break;
                case QFHState::RECOVERING: recovering_count++; break;
                case QFHState::FLIP: flip_count++; break;
                case QFHState::RUPTURE: rupture_count++; break;
            }
        }
        
        float total = static_cast<float>(local_events.size());
        float null_ratio = null_count / total;
        float flip_ratio = flip_count / total;
        float rupture_ratio = rupture_count / total;
        float stable_ratio = stable_count / total;
        float unstable_ratio = unstable_count / total;
        float collapsing_ratio = collapsing_count / total;
        float collapsed_ratio = collapsed_count / total;
        float recovering_ratio = recovering_count / total;
        
        auto safe_log2 = [](float x) -> float { return (x > 0.0f) ? std::log2(x) : 0.0f; };
        local_entropy = -(null_ratio * safe_log2(null_ratio) + flip_ratio * safe_log2(flip_ratio) + rupture_ratio * safe_log2(rupture_ratio) +
                         stable_ratio * safe_log2(stable_ratio) + unstable_ratio * safe_log2(unstable_ratio) +
                         collapsing_ratio * safe_log2(collapsing_ratio) + collapsed_ratio * safe_log2(collapsed_ratio) +
                         recovering_ratio * safe_log2(recovering_ratio));
        local_entropy = std::fmax(0.05, std::fmin(1.0, local_entropy / 1.585));
        local_coherence = 1.0 - local_entropy;
    }
    
    // Apply mathematical formula from bitspace_math.md:
    // λ = k1 * Entropy + k2 * (1 - Coherence)
    double k1 = options_.entropy_weight;  // Configurable entropy weight, default 0.30
    double k2 = options_.coherence_weight;  // Configurable coherence weight, default 0.20
    double lambda = k1 * local_entropy + k2 * (1.0 - local_coherence);
    lambda = std::fmax(0.01, std::fmin(1.0, lambda));  // Constrain to reasonable range
    // Expose diagnostics
    damped_value.lambda = lambda;
    damped_value.start_index = current_index;
    
    // Step 2: Integrate future trajectories with exponential damping
    // Formula: V_i = Σ(p_j - p_i) * e^(-λ(j-i))
    double accumulated_value = 0.0;
    double current_bit = static_cast<double>(bitstream[current_index]);
    
    // Store trajectory path for confidence analysis
    damped_value.path.clear();
    damped_value.path.reserve(bitstream.size() - current_index);
    damped_value.path.push_back(current_bit);
    
    for (size_t j = current_index + 1; j < bitstream.size(); ++j) {
        double future_bit = static_cast<double>(bitstream[j]);
        double time_difference = static_cast<double>(j - current_index);
        double weight = std::exp(-lambda * time_difference);
        
        // Apply the damped value formula
        double contribution = (future_bit - current_bit) * weight;
        accumulated_value += contribution;
        
        // Store intermediate trajectory points
        damped_value.path.push_back(accumulated_value);
    }
    
    damped_value.final_value = accumulated_value;
    
    // Step 3: Calculate preliminary confidence based on trajectory consistency
    // Higher consistency in trajectory indicates more predictable behavior
    if (damped_value.path.size() > 2) {
        double trajectory_variance = 0.0;
        double mean_trajectory = 0.0;
        
        // Calculate mean of trajectory
        for (double val : damped_value.path) {
            mean_trajectory += val;
        }
        mean_trajectory /= damped_value.path.size();
        
        // Calculate variance
        for (double val : damped_value.path) {
            trajectory_variance += std::pow(val - mean_trajectory, 2);
        }
        trajectory_variance /= damped_value.path.size();
        
        // Convert variance to confidence (lower variance = higher confidence)
        double stability_score = 1.0 / (1.0 + trajectory_variance);
        damped_value.confidence = std::fmax(0.0, std::fmin(1.0, stability_score));
    } else {
        damped_value.confidence = 0.5;  // Default confidence for insufficient data
    }
    
    // Mark as converged if trajectory shows stability
    damped_value.converged = (damped_value.confidence > 0.7);
    
    // (debug print removed)

    return damped_value;
}

double QFHBasedProcessor::matchKnownPaths(const std::vector<double>& current_path) {
    // For now, implement a simple pattern matching against common trajectory shapes
    // In a full implementation, this would query a historical database (e.g., Redis)
    
    if (current_path.size() < 3) {
        return 0.5;  // Default confidence for insufficient data
    }
    
    // Define some common trajectory patterns for pattern matching
    // Pattern 1: Exponential decay (common in stable signals)
    // Pattern 2: Linear trend (common in trending markets)
    // Pattern 3: Oscillating pattern (common in ranging markets)
    
    double best_similarity = 0.0;
    
    // Pattern 1: Exponential decay similarity
    std::vector<double> exponential_pattern;
    exponential_pattern.reserve(current_path.size());
    double initial_value = current_path[0];
    for (size_t i = 0; i < current_path.size(); ++i) {
        double expected_value = initial_value * std::exp(-0.1 * static_cast<double>(i));
        exponential_pattern.push_back(expected_value);
    }
    
    double exp_similarity = calculateCosineSimilarity(current_path, exponential_pattern);
    best_similarity = std::max(best_similarity, exp_similarity);
    
    // Pattern 2: Linear trend similarity
    if (current_path.size() >= 2) {
        std::vector<double> linear_pattern;
        linear_pattern.reserve(current_path.size());
        double slope = (current_path.back() - current_path.front()) / (current_path.size() - 1);
        for (size_t i = 0; i < current_path.size(); ++i) {
            double expected_value = current_path[0] + slope * static_cast<double>(i);
            linear_pattern.push_back(expected_value);
        }
        
        double linear_similarity = calculateCosineSimilarity(current_path, linear_pattern);
        best_similarity = std::max(best_similarity, linear_similarity);
    }
    
    // Pattern 3: Oscillating pattern similarity (sine wave approximation)
    std::vector<double> oscillating_pattern;
    oscillating_pattern.reserve(current_path.size());
    double amplitude = (current_path.back() - current_path.front()) / 2.0;
    double mean_value = (current_path.back() + current_path.front()) / 2.0;
    for (size_t i = 0; i < current_path.size(); ++i) {
        double expected_value = mean_value + amplitude * std::sin(2.0 * M_PI * i / current_path.size());
        oscillating_pattern.push_back(expected_value);
    }
    
    double osc_similarity = calculateCosineSimilarity(current_path, oscillating_pattern);
    best_similarity = std::max(best_similarity, osc_similarity);
    
    // Return the highest similarity score found
    return std::fmax(0.0, std::fmin(1.0, best_similarity));
}



// QFHBasedProcessor implementation
sep::quantum::QFHBasedProcessor::QFHBasedProcessor(const QFHOptions& options) : options_(options) {}

sep::quantum::QFHResult sep::quantum::QFHBasedProcessor::analyze(const std::vector<uint8_t>& bits)
{
    sep::quantum::QFHResult result;
    result.collapse_threshold = options_.collapse_threshold;

    // Transform bits to events
    result.events = transform_rich(bits);
    // std::cerr << "analyze: events size: " << result.events.size() << std::endl;

    // Aggregate events
    result.aggregated_events = aggregate(result.events);
    
    // Count event types
    for (const auto& event : result.events) {
        switch (event.state) {
            case sep::quantum::QFHState::NULL_STATE:
                result.null_state_count++;
                break;
            case sep::quantum::QFHState::STABLE:
                // Track stable events (no counter in QFHResult yet, but handle the case)
                break;
            case sep::quantum::QFHState::UNSTABLE:
                // Track unstable events (no counter in QFHResult yet, but handle the case)
                break;
            case sep::quantum::QFHState::COLLAPSING:
                // Track collapsing events (no counter in QFHResult yet, but handle the case)
                break;
            case sep::quantum::QFHState::COLLAPSED:
                // Track collapsed events (no counter in QFHResult yet, but handle the case)
                break;
            case sep::quantum::QFHState::RECOVERING:
                // Track recovering events (no counter in QFHResult yet, but handle the case)
                break;
            case sep::quantum::QFHState::FLIP:
                result.flip_count++;
                break;
            case sep::quantum::QFHState::RUPTURE:
                result.rupture_count++;
                break;
            default:
                break;
        }
    }
    
    // Calculate ratios
    if (!result.events.empty()) {
        result.rupture_ratio = static_cast<float>(result.rupture_count) /
                               static_cast<float>(result.events.size());
        result.flip_ratio = static_cast<float>(result.flip_count) /
                            static_cast<float>(result.events.size());
    }

    // Keep natural ratios without artificial enhancement
    
    // Calculate coherence based on pattern consistency (from POC research)
    // High coherence = low variance in state transitions, consistent patterns
    if (!result.events.empty()) {
        // Natural coherence calculation: inverse of entropy (no artificial adjustments)
        // Calculate Shannon entropy of state distribution
        float null_ratio = static_cast<float>(result.null_state_count) / static_cast<float>(result.events.size());
        float flip_ratio = result.flip_ratio;
        float rupture_ratio = result.rupture_ratio;
        
        auto safe_log2 = [](float x) -> float {
            return (x > 0.0f) ? std::log2(x) : 0.0f;
        };
        
        result.entropy = -(null_ratio * safe_log2(null_ratio) +
                          flip_ratio * safe_log2(flip_ratio) +
                          rupture_ratio * safe_log2(rupture_ratio));
        
        // Normalize entropy to [0,1] (max entropy for 3 states is log2(3) ≈ 1.585)
        result.entropy = std::fmax(0.05f, std::fmin(1.0f, result.entropy / 1.585f));  // Minimum 0.05 entropy
        
        // Natural coherence calculation without artificial scaling
        // Coherence = 1 - entropy (higher entropy = lower coherence)
        float pattern_coherence = 1.0f - result.entropy;

        // Stability factor: lower rupture = higher coherence
        float stability_factor = 1.0f - result.rupture_ratio;

        // Consistency factor: lower flip frequency = higher coherence
        float consistency_factor = 1.0f - result.flip_ratio;

        // Weighted combination preserving natural market variation
        result.coherence = pattern_coherence * 0.6f + stability_factor * 0.3f + consistency_factor * 0.1f;

        // Ensure coherence stays within valid range [0.01, 0.99]
        result.coherence = std::fmax(0.01f, std::fmin(0.99f, result.coherence));
    }
    
    // Integrate future trajectories for damping - this is the core enhancement
    if (!bits.empty() && bits.size() > 10) {  // Only apply enhancement for significant data
        bitspace::DampedValue dv = integrateFutureTrajectories(bits, 0);
        
        // Use trajectory-based confidence for coherence calculation
        double trajectory_confidence = matchKnownPaths(dv.path);
        
        // Blend the pattern-based coherence with trajectory-based confidence
        // This creates more stable and reliable coherence values
        float pattern_coherence = result.coherence;  // Pattern-based from above
        float trajectory_coherence = static_cast<float>(trajectory_confidence);
        
        // Conservative weighted combination: 30% trajectory-based, 70% pattern-based
        // This preserves existing behavior while adding trajectory enhancement
        result.coherence = 0.3f * trajectory_coherence + 0.7f * pattern_coherence;
        
        // Apply the damped final value to influence the coherence stability (more conservatively)
        // Smaller damped values indicate more stable futures, higher coherence
        if (std::abs(dv.final_value) < 2.0) {  // Only apply if damped value is reasonable
            float stability_factor = 1.0f / (1.0f + 0.1f * std::abs(static_cast<float>(dv.final_value)));
            result.coherence = result.coherence * stability_factor;
        }
        
        // Ensure coherence stays within valid range
        result.coherence = std::fmax(0.0f, std::fmin(1.0f, result.coherence));
    }

    // Detect collapse
    result.collapse_detected = (result.rupture_ratio >= options_.collapse_threshold);
    
    return result;
}

bool sep::quantum::QFHBasedProcessor::detectCollapse(const QFHResult& result) const {
    return result.collapse_detected || result.rupture_ratio >= options_.collapse_threshold;
}

std::vector<uint8_t> sep::quantum::QFHBasedProcessor::convertToBits(const std::vector<uint32_t>& values)
{
    std::vector<uint8_t> bits;
    bits.reserve(values.size() * 32);
    
    for (uint32_t value : values) {
        for (int i = 0; i < 32; ++i) {
            bits.push_back((value >> i) & 1);
        }
    }
    
    return bits;
}

// Helper function to calculate cosine similarity between two vectors
double QFHBasedProcessor::calculateCosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }
    
    return dot_product / (norm_a * norm_b);
}

void sep::quantum::QFHBasedProcessor::reset() {
    // Call base class reset
    QFHProcessor::reset();
    
    // Reset derived class state
    current_state_ = QFHState::STABLE;
    prev_bit_ = 0;
}

} // namespace sep::quantum
