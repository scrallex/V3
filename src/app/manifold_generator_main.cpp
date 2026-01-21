#include "../core/trading_signals.h"
#include "../core/qfh.h"
#include "../core/manifold_builder.h"
#include "../core/io_utils.h"
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <iostream>

using nlohmann::json;

namespace {

using sep::io::parse_yyyy_mm_dd_ms;
using sep::io::load_candles_from_file;

// Ensure parent directory exists
void ensure_parent_dir(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return;
    std::string dir = path.substr(0, pos);
    if (dir.empty()) return;
    struct stat st{};
    if (stat(dir.c_str(), &st) == 0) return;
    // Try to create recursively (best effort, simple)
    std::string cur;
    std::stringstream ss(dir);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        if (!cur.empty()) cur += "/";
        cur += part;
        if (stat(cur.c_str(), &st) != 0) {
            mkdir(cur.c_str(), 0755);
        }
    }
}

// Load candles from Valkey using schema md:candles:{instrument}:M1 and date range YYYY-MM-DD..YYYY-MM-DD
std::vector<sep::Candle> load_candles_from_valkey_spec(const std::string& spec) {
    const std::string prefix = "valkey:";
    if (spec.rfind(prefix, 0) != 0) {
        return {};
    }
    std::string rest = spec.substr(prefix.size());
    std::vector<std::string> parts;
    std::stringstream ss(rest);
    std::string tok;
    while (std::getline(ss, tok, ':')) {
        parts.push_back(tok);
    }
    if (parts.size() < 3) {
        spdlog::error("Invalid valkey spec '{}', expected valkey:{{instrument}}:{{start}}:{{end}}", spec);
        return {};
    }
    const std::string instrument = parts[0];
    const std::string start_s = parts[1];
    const std::string end_s = parts[2];
    auto t0_ms_opt = parse_yyyy_mm_dd_ms(start_s);
    auto t1_ms_opt = parse_yyyy_mm_dd_ms(end_s);
    if (!t0_ms_opt || !t1_ms_opt) {
        spdlog::error("Invalid start/end date in valkey spec '{}'", spec);
        return {};
    }
    const char* env = std::getenv("VALKEY_URL");
    if (!env || std::string(env).empty()) {
        spdlog::error("VALKEY_URL not set for valkey input");
        return {};
    }
    redisContext* c = sep::connectValkey(env);
    if (!c) {
        spdlog::error("Failed to connect to Valkey");
        return {};
    }
    uint64_t t0_ms = *t0_ms_opt;
    uint64_t t1_ms = *t1_ms_opt + 24ULL * 3600ULL * 1000ULL;
    std::string key = "md:candles:" + instrument + ":M1";
    auto candles = sep::fetchCandlesByScore(c, key, t0_ms, t1_ms);
    redisFree(c);
    std::sort(candles.begin(), candles.end(), [](const sep::Candle& a, const sep::Candle& b) {
        return a.timestamp < b.timestamp;
    });
    return candles;
}

} // namespace

int main(int argc, char** argv) {
    cxxopts::Options opts("manifold_generator", "Generate manifolds from candle data");
    opts.add_options()
        ("input", "file or valkey:{instrument}:{start}:{end}", cxxopts::value<std::string>())
        ("output", "file path or 'valkey'", cxxopts::value<std::string>()->default_value(""))
        ("cpu-only", "Force CPU only", cxxopts::value<bool>()->default_value("false"));
    auto args = opts.parse(argc, argv);

    if (!args.count("input")) {
        spdlog::error("Missing --input");
        return 2;
    }
    const std::string in = args["input"].as<std::string>();
    const std::string out = args["output"].as<std::string>();

    std::vector<sep::Candle> candles;
    std::string instrument_hint;

    if (in.rfind("valkey:", 0) == 0) {
        candles = load_candles_from_valkey_spec(in);
        // Extract instrument from spec for hint
        std::string rest = in.substr(std::string("valkey:").size());
        auto pos = rest.find(':');
        instrument_hint = (pos == std::string::npos) ? "" : rest.substr(0, pos);
    } else {
        candles = load_candles_from_file(in);
    }

    if (candles.empty()) {
        spdlog::warn("No candles loaded from input '{}'", in);
    }

    json manifold = sep::buildManifold(candles, instrument_hint);

    // Output handling
    if (!out.empty() && out == "valkey") {
        // Store gz manifold with TTL under manifold:{instrument}:{YYYY-MM-DD}
        if (candles.empty()) {
            spdlog::warn("Skipping Valkey output: no candles");
            return 0;
        }
        const char* env = std::getenv("VALKEY_URL");
        if (!env || std::string(env).empty()) {
            spdlog::error("VALKEY_URL not set; cannot output to valkey");
            return 3;
        }
        redisContext* c = sep::connectValkey(env);
        if (!c) {
            spdlog::error("Valkey connection failed");
            return 3;
        }
        // Day key by t1
        uint64_t t1_ms = candles.back().timestamp;
        std::time_t t = static_cast<time_t>(t1_ms / 1000ULL);
        std::tm* gmt = gmtime(&t);
        char buf[32]{0};
        if (gmt) {
            std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", gmt->tm_year + 1900, gmt->tm_mon + 1, gmt->tm_mday);
        } else {
            std::snprintf(buf, sizeof(buf), "1970-01-01");
        }
        std::string day_key(buf);
        std::string instr = manifold.value("instrument", instrument_hint.empty() ? "UNKNOWN" : instrument_hint);
        std::string k = "manifold:" + instr + ":" + day_key;
        try {
            sep::storeGzipManifold(c, k, manifold, 35);
            spdlog::info("Stored {} to Valkey (gz, TTL 35d)", k);
        } catch (const std::exception& e) {
            spdlog::error("Failed storing to Valkey: {}", e.what());
            redisFree(c);
            return 4;
        }
        redisFree(c);
        return 0;
    } else {
        // Write to file (default)
        std::string out_path = out;
        if (out_path.empty()) {
            // Fallback to stdout
            std::cout << manifold.dump() << std::endl;
            return 0;
        }
        ensure_parent_dir(out_path);
        std::ofstream ofs(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!ofs.good()) {
            spdlog::error("Failed opening output file {}", out_path);
            return 5;
        }
        std::string s = manifold.dump();
        ofs.write(s.data(), static_cast<std::streamsize>(s.size()));
        ofs.close();
        spdlog::info("Wrote manifold JSON to {}", out_path);
        return 0;
    }
}
