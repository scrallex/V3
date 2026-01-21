#pragma once

#include "trading_signals.h"

#include <cstdint>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sep::oanda {

struct OandaConfig {
    std::string api_key;
    std::string account_id;
    std::string environment = "practice"; // "practice" or "live"
};

class OandaClient {
public:
    explicit OandaClient(const OandaConfig& config);
    ~OandaClient();
    
    // Delete copy/move to avoid issues with potential resources
    OandaClient(const OandaClient&) = delete;
    OandaClient& operator=(const OandaClient&) = delete;
    OandaClient(OandaClient&&) = delete;
    OandaClient& operator=(OandaClient&&) = delete;
    
    /// Check if client is properly configured with API credentials
    bool isConfigured() const;
    
    /// Fetch historical candle data for an instrument
    std::vector<sep::Candle> fetchHistoricalCandles(
        const std::string& instrument,
        const std::string& from,           // RFC3339 format: "2023-01-01T00:00:00Z"
        const std::string& to,             // RFC3339 format: "2023-01-02T00:00:00Z"
        const std::string& granularity = "M1"
    );
    
    /// Factory method to create client from environment variables
    static std::unique_ptr<OandaClient> fromEnvironment();
    
private:
    OandaConfig config_;
    
    std::string getApiBaseUrl() const;
    std::vector<sep::Candle> parseCandles(const nlohmann::json& json);
    uint64_t parseTimestamp(const std::string& timeStr);
    double parseVolume(const nlohmann::json& candle);
};

} // namespace sep::oanda
