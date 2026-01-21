#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <functional>
#include "forward_window_result.h"
#include "trajectory.h"

namespace sep {
namespace quantum {

// Forward declarations
struct QFHOptions;
struct QFHResult;
class QFHProcessor;
class QFHBasedProcessor;
struct QFHEvent;
struct QFHAggregateEvent;

// Forward declaration for DampedValue
namespace bitspace {
    struct DampedValue;
}

/**
 * QFH State enumeration
 */
enum class QFHState {
    NULL_STATE,
    STABLE,
    UNSTABLE,
    COLLAPSING,
    COLLAPSED,
    RECOVERING,
    FLIP,
    RUPTURE
};

/**
 * QFH Event structure for quantum field harmonics processing
 */
struct QFHEvent {
    uint32_t index{0};
    QFHState state{QFHState::NULL_STATE};
    uint8_t bit_prev{0};
    uint8_t bit_curr{0};
    
    // Equality operator for event comparison
    bool operator==(const QFHEvent& other) const;
};

/**
* Transform bitstream into rich QFH events
* @param bits Input bitstream
* @return Vector of QFH events representing bit transitions
*/
std::vector<QFHEvent> transform_rich(const std::vector<uint8_t>& bits);

/**
* Aggregate QFH events into grouped events
* @param events Input QFH events
* @return Vector of aggregated QFH events
*/
std::vector<QFHAggregateEvent> aggregate(const std::vector<QFHEvent>& events);

/**
 * QFH Aggregate Event for event processing
 */
struct QFHAggregateEvent {
    uint32_t index{0};
    QFHState state{QFHState::NULL_STATE};
    uint32_t count{1};
};

/**
* Transform bitstream into rich QFH events
* @param bits Input bitstream
* @return Vector of QFH events representing bit transitions
*/
std::vector<QFHEvent> transform_rich(const std::vector<uint8_t>& bits);

/**
* Aggregate QFH events into grouped events
* @param events Input QFH events
* @return Vector of aggregated QFH events
*/
std::vector<QFHAggregateEvent> aggregate(const std::vector<QFHEvent>& events);

/**
 * Configuration options for QFH processing
 */
struct QFHOptions {
    double coherence_threshold = 0.7;
    double stability_threshold = 0.8;
    double collapse_threshold = 0.5;
    int max_iterations = 1000;
    bool enable_damping = true;
    double damping_factor = 0.95;
    double entropy_weight = 0.30;    // Configurable entropy weight
    double coherence_weight = 0.20;  // Configurable coherence weight
};

/**
 * Result structure for QFH operations
 */
struct QFHResult {
    double coherence = 0.0;
    double stability = 0.0;
    double confidence = 0.0;
    bool collapse_detected = false;
    double rupture_ratio = 0.0;
    QFHState final_state = QFHState::STABLE;
    std::vector<QFHEvent> events;
    
    // Additional members required by implementations
    double collapse_threshold = 0.5;
    std::vector<QFHAggregateEvent> aggregated_events;
    uint32_t null_state_count = 0;
    uint32_t flip_count = 0;
    uint32_t rupture_count = 0;
    double flip_ratio = 0.0;
    double entropy = 0.0;
};

/**
 * Base QFH processor for quantum field harmonics
 */
class QFHProcessor {
public:
    QFHProcessor() = default;
    virtual ~QFHProcessor() = default;
    
    virtual std::optional<QFHState> process(uint8_t current_bit);
    virtual void reset();
    
protected:
    std::optional<uint8_t> prev_bit;
};

/**
 * QFH-based processor implementation
 */
class QFHBasedProcessor : public QFHProcessor {
public:
    explicit QFHBasedProcessor(const QFHOptions& options);
    ~QFHBasedProcessor() override = default;
    
    QFHResult analyze(const std::vector<uint8_t>& data);
    void reset() override;
    
    // Additional methods expected by the implementation
    bitspace::DampedValue integrateFutureTrajectories(const std::vector<uint8_t>& bitstream, size_t current_index);
    double matchKnownPaths(const std::vector<double>& trajectory);
    std::vector<uint8_t> convertToBits(const std::vector<uint32_t>& data);
    double calculateCosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);
    
    // Additional methods for QFH-based processing
    std::optional<QFHState> detectTransition(uint32_t prev_bit, uint32_t current_bit);
    bool detectCollapse(const QFHResult& result) const;
    
private:
    QFHOptions options_;
    QFHState current_state_ = QFHState::STABLE;
    uint32_t prev_bit_ = 0;
};

namespace bitspace {

// ForwardWindowResult is now defined in forward_window_result.h

// DampedValue is defined in trajectory.h to avoid duplicate definition

/**
 * QFH (Quantum Field Harmonics) configuration constants
 */
namespace qfh {
    constexpr double DEFAULT_LAMBDA = 0.1;  // Default decay constant
    constexpr int MAX_PACKAGE_SIZE = 1024;  // Maximum bit package size
    constexpr int MIN_PACKAGE_SIZE = 8;     // Minimum bit package size
}

} // namespace bitspace
} // namespace quantum
} // namespace sep