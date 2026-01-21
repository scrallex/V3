#include "oanda_client.h"
#include "io_utils.h"
#include <spdlog/spdlog.h>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <ctime>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <random>

namespace sep::oanda {

OandaClient::OandaClient(const OandaConfig& config) : config_(config) {}

OandaClient::~OandaClient() = default;

bool OandaClient::isConfigured() const {
    return !config_.api_key.empty() && !config_.account_id.empty();
}

std::string OandaClient::getApiBaseUrl() const {
    return (config_.environment == "live" || config_.environment == "LIVE")
        ? "https://api-fxtrade.oanda.com"
        : "https://api-fxpractice.oanda.com";
}

std::vector<sep::Candle> OandaClient::fetchHistoricalCandles(
    const std::string& instrument,
    const std::string& from,
    const std::string& to,
    const std::string& granularity
) {
    if (!isConfigured()) {
        spdlog::warn("OANDA client not configured - returning empty candles");
        return {};
    }

    const std::string url = getApiBaseUrl() + "/v3/instruments/" + instrument + "/candles";

    cpr::Parameters params{
        {"granularity", granularity},
        {"price", "M"},
        {"from", from},
        {"to", to}
    };

    cpr::Header headers{
        {"Authorization", "Bearer " + config_.api_key},
        {"Content-Type", "application/json"},
        {"Accept-Datetime-Format", "RFC3339"}
    };

    // Simple retry with exponential backoff + jitter
    int max_retries = 3;
    double backoff_base = 1.5;
    if (const char* mr = std::getenv("OANDA_MAX_RETRIES")) {
        try { max_retries = std::max(1, std::stoi(mr)); } catch (...) {}
    }
    if (const char* bb = std::getenv("OANDA_BACKOFF_BASE")) {
        try { backoff_base = std::max(1.1, std::stod(bb)); } catch (...) {}
    }

    std::mt19937_64 rng{static_cast<unsigned long long>(std::chrono::high_resolution_clock::now().time_since_epoch().count())};
    std::uniform_real_distribution<double> jitter(0.0, 1.0);

    for (int attempt = 1; attempt <= max_retries; ++attempt) {
        auto response = cpr::Get(cpr::Url{url}, params, headers, cpr::Timeout{30000});
        int code = response.status_code;
        bool ok = (code == 200);
        if (ok) {
            try {
                return parseCandles(nlohmann::json::parse(response.text));
            } catch (const std::exception& e) {
                spdlog::error("Failed to parse OANDA response: {}", e.what());
                return {};
            }
        }

        // On failure, log and possibly retry
        std::string snippet = response.text.substr(0, 256);
        if (code == 429) {
            spdlog::warn("OANDA rate limited (429) on attempt {}/{}", attempt, max_retries);
        } else if (code >= 500 || code == 0) {
            spdlog::warn("OANDA transient error {} on attempt {}/{}: {}", code, attempt, max_retries, snippet);
        } else {
            spdlog::error("OANDA API error {}: {}", code, snippet);
            break; // non-retryable (4xx other than 429)
        }

        if (attempt < max_retries) {
            double delay = std::pow(backoff_base, attempt - 1) + jitter(rng);
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(delay * 1000))); 
            continue;
        }
    }
    return {};
}

std::vector<sep::Candle> OandaClient::parseCandles(const nlohmann::json& json) {
    std::vector<sep::Candle> candles;
    
    if (!json.contains("candles") || !json["candles"].is_array()) {
        spdlog::error("Invalid OANDA response: no candles array");
        return candles;
    }
    
    for (const auto& candleJson : json["candles"]) {
        try {
            if (!candleJson.value("complete", false)) {
                continue; // Skip incomplete candles
            }
            
            sep::Candle candle;
            candle.timestamp = parseTimestamp(candleJson.value("time", ""));
            
            const auto& mid = candleJson.at("mid");
            candle.open = std::stod(mid.value("o", "0"));
            candle.high = std::stod(mid.value("h", "0"));
            candle.low = std::stod(mid.value("l", "0"));
            candle.close = std::stod(mid.value("c", "0"));
            
            // Handle volume parsing with multiple formats
            candle.volume = parseVolume(candleJson);
            
            candles.push_back(candle);
        } catch (const std::exception& e) {
            spdlog::debug("Skipping malformed candle: {}", e.what());
            continue;
        }
    }
    
    spdlog::info("Parsed {} candles from OANDA response", candles.size());
    return candles;
}

uint64_t OandaClient::parseTimestamp(const std::string& timeStr) {
    try {
        auto parsed = sep::io::parse_iso8601_ms(timeStr);
        if (!parsed) {
            throw std::runtime_error("invalid timestamp format");
        }
        return *parsed;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to parse timestamp '{}': {}", timeStr, e.what());
        throw;
    }
}


double OandaClient::parseVolume(const nlohmann::json& candle) {
    if (!candle.contains("volume")) {
        return 0.0;
    }
    
    const auto& volume = candle["volume"];
    try {
        if (volume.is_string()) {
            return std::stod(static_cast<std::string>(volume));
        } else if (volume.is_number_integer()) {
            return static_cast<double>(volume.get<int>());
        } else if (volume.is_number()) {
            return volume.get<double>();
        }
    } catch (const std::exception&) {
        // Fall through to default
    }
    
    return 0.0;
}

// Factory function for easy creation from environment
std::unique_ptr<OandaClient> OandaClient::fromEnvironment() {
    OandaConfig config;
    
    if (const char* api_key = std::getenv("OANDA_API_KEY")) {
        config.api_key = api_key;
    }
    
    if (const char* account_id = std::getenv("OANDA_ACCOUNT_ID")) {
        config.account_id = account_id;
    }
    
    if (const char* environment = std::getenv("OANDA_ENVIRONMENT")) {
        config.environment = environment;
    } else {
        config.environment = "practice"; // Default to practice
    }
    
    return std::make_unique<OandaClient>(config);
}

} // namespace sep::oanda
