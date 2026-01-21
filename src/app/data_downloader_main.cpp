#include "../core/trading_signals.h"
#include "../core/oanda_client.h"
#include "../core/io_utils.h"
#include <cxxopts.hpp>
#include <thread>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace {

bool isValidCandle(const sep::Candle& c) {
    if (c.timestamp == 0) return false;
    return std::isfinite(c.open) && std::isfinite(c.high) && std::isfinite(c.low) &&
           std::isfinite(c.close) && std::isfinite(c.volume);
}

std::vector<sep::Candle> sanitizeCandles(const std::vector<sep::Candle>& candles, const std::string& label) {
    std::vector<sep::Candle> cleaned;
    cleaned.reserve(candles.size());
    for (const auto& c : candles) {
        if (isValidCandle(c)) {
            cleaned.push_back(c);
        } else {
            spdlog::warn("Discarding malformed candle for {} (ts={})", label, c.timestamp);
        }
    }
    return cleaned;
}

// Map OANDA granularity to seconds (approximate)
int granularitySeconds(const std::string& g) {
    if (g == "S5") return 5;
    if (g == "S10") return 10;
    if (g == "S15") return 15;
    if (g == "S30") return 30;
    if (g == "M1") return 60;
    if (g == "M2") return 120;
    if (g == "M4") return 240;
    if (g == "M5") return 300;
    if (g == "M10") return 600;
    if (g == "M15") return 900;
    if (g == "M30") return 1800;
    if (g == "H1") return 3600;
    if (g == "H2") return 7200;
    if (g == "H3") return 10800;
    if (g == "H4") return 14400;
    if (g == "H6") return 21600;
    if (g == "H8") return 28800;
    if (g == "H12") return 43200;
    if (g == "D") return 86400;
    if (g == "W") return 7 * 86400;
    if (g == "M") return 30 * 86400;
    return 60; // default M1
}

}

int main(int argc, char** argv) {
    cxxopts::Options opts("data_downloader", "Fetch OANDA data");
    opts.add_options()
        ("instrument", "e.g. EUR_USD", cxxopts::value<std::string>())
        ("granularity", "OANDA granularity (e.g., M1, M5, H1)", cxxopts::value<std::string>()->default_value("M1"))
        ("from", "YYYY-MM-DD (midnight UTC)", cxxopts::value<std::string>())
        ("to", "YYYY-MM-DD (midnight UTC)", cxxopts::value<std::string>())
        ("from-time", "RFC3339 start time (e.g., 2025-08-31T03:10:00Z)", cxxopts::value<std::string>()->default_value(""))
        ("to-time", "RFC3339 end time   (e.g., 2025-08-31T03:30:00Z)", cxxopts::value<std::string>()->default_value(""))
        ("output", "file path or 'valkey'", cxxopts::value<std::string>())
        ("out", "file path alias of --output", cxxopts::value<std::string>()->default_value(""))
        ("stream", "Live stream mode", cxxopts::value<bool>()->default_value("false"))
        ("stream_interval", "Polling interval seconds (stream mode)", cxxopts::value<int>()->default_value("5"));
    auto args = opts.parse(argc, argv);
    // Support --out as alias of --output for orchestrator compatibility
    std::string output_path;
    if (args.count("out") && !args["out"].as<std::string>().empty()) {
        output_path = args["out"].as<std::string>();
    } else if (args.count("output")) {
        output_path = args["output"].as<std::string>();
    } else {
        spdlog::error("Missing --output/--out parameter");
        return 6;
    }

    // Create OANDA client from environment
    auto oandaClient = sep::oanda::OandaClient::fromEnvironment();
    if (!oandaClient->isConfigured()) {
        spdlog::error("OANDA client not configured - check environment variables");
        return 1;
    }

    if (args["stream"].as<bool>()) {
        // Stream mode: periodically poll recent candles and append new completes
        const std::string ins = args["instrument"].as<std::string>();
        const std::string gran = args["granularity"].as<std::string>();
        const int interval = std::max(1, args["stream_interval"].as<int>());

        if (output_path == "valkey") {
            spdlog::warn("Streaming to 'valkey' not supported in data_downloader yet; use file output.");
            return 1;
        }

        std::ofstream ofs(output_path, std::ios::app);
        if (!ofs.good()) {
            spdlog::error("Failed to open output file for append: {}", output_path);
            return 5;
        }

        spdlog::info("Starting stream mode for {} gran={} → {} (interval={}s)", ins, gran, output_path, interval);
        uint64_t last_ts_ms = 0;
        const int gsec = granularitySeconds(gran);
        while (true) {
            try {
                using namespace std::chrono;
                auto now = std::chrono::system_clock::now();
                // Pull a small trailing window to catch late completes (3x granularity)
                auto from_tp = now - std::chrono::seconds(std::max(1, 3 * gsec));
                std::string t0 = sep::io::toRfc3339Utc(from_tp);
                std::string t1 = sep::io::toRfc3339Utc(now);
                auto candles = sanitizeCandles(
                    oandaClient->fetchHistoricalCandles(ins, t0, t1, gran.empty() ? std::string("M1") : gran),
                    ins
                );
                size_t appended = 0;
                for (const auto& c : candles) {
                    if (c.timestamp <= last_ts_ms) continue;
                    nlohmann::json line;
                    line["time"] = sep::io::epochMsToRfc3339(c.timestamp);
                    line["o"] = c.open;
                    line["h"] = c.high;
                    line["l"] = c.low;
                    line["c"] = c.close;
                    line["v"] = c.volume;
                    ofs << line.dump() << '\n';
                    ++appended;
                    last_ts_ms = c.timestamp;
                }
                if (appended > 0) {
                    ofs.flush();
                    spdlog::info("Appended {} new candles; last_ts_ms={}", appended, last_ts_ms);
                }
            } catch (const std::exception& e) {
                spdlog::warn("Stream iteration error: {}", e.what());
            }
            std::this_thread::sleep_for(std::chrono::seconds(interval));
        }
        return 0;
    } else {
        // Use consolidated OANDA client for historical data
        const std::string ins = args["instrument"].as<std::string>();
        const std::string gran = args["granularity"].as<std::string>();
        const std::string from_time = args["from-time"].as<std::string>();
        const std::string to_time   = args["to-time"].as<std::string>();

        std::string t0;
        std::string t1;
        if (!from_time.empty() && !to_time.empty()) {
            t0 = from_time; t1 = to_time;
        } else {
            // Fallback to date-based window at UTC midnight
            if (!args.count("from") || !args.count("to")) {
                spdlog::error("Missing --from/--to or --from-time/--to-time parameters");
                return 6;
            }
            const std::string from_d = args["from"].as<std::string>();
            const std::string to_d   = args["to"].as<std::string>();
            t0 = from_d + "T00:00:00Z";
            t1 = to_d   + "T00:00:00Z";
        }

        // Fetch candles using consolidated client
        auto candles = sanitizeCandles(
            oandaClient->fetchHistoricalCandles(ins, t0, t1, gran.empty() ? std::string("M1") : gran),
            ins
        );
        
        if (candles.empty()) {
            spdlog::warn("No candles retrieved for {} from {} to {}", ins, t0, t1);
            return 2;
        }

        // Always emit file JSONL: each line {"time": ISO, "o","h","l","c","v"} with numeric OHLC
        std::ofstream ofs(output_path);
        if (!ofs.good()) {
            spdlog::error("Failed to open output file: {}", output_path);
            return 5;
        }

        for (const auto& candle : candles) {
            nlohmann::json line;
            line["time"] = sep::io::epochMsToRfc3339(candle.timestamp);
            line["o"] = candle.open;
            line["h"] = candle.high;
            line["l"] = candle.low;
            line["c"] = candle.close;
            line["v"] = candle.volume;
            ofs << line.dump() << "\n";
        }
        
        spdlog::info("Successfully wrote {} candles to {}", candles.size(), output_path);
    }
    return 0;
}
