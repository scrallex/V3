#include "trading_signals.h"
#include "qfh.h"
#include "manifold_builder.h"
#include "io_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

std::optional<double> parseNumericField(const nlohmann::json& value) {
    try {
        if (value.is_number_float() || value.is_number_integer() || value.is_number_unsigned()) {
            return value.get<double>();
        }
        if (value.is_string()) {
            const auto& s = value.get_ref<const std::string&>();
            if (s.empty()) {
                return std::nullopt;
            }
            size_t idx = 0;
            double parsed = std::stod(s, &idx);
            if (idx == s.size()) {
                return parsed;
            }
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<uint64_t> parseTimestampField(const nlohmann::json& value) {
    try {
        if (value.is_number_unsigned()) {
            return value.get<uint64_t>();
        }
        if (value.is_number_integer()) {
            auto v = value.get<int64_t>();
            if (v >= 0) {
                return static_cast<uint64_t>(v);
            }
            return std::nullopt;
        }
        if (value.is_string()) {
            const auto& s = value.get_ref<const std::string&>();
            if (s.empty()) {
                return std::nullopt;
            }
            const bool digits_only = std::all_of(s.begin(), s.end(), [](unsigned char ch) { return std::isdigit(ch); });
            if (digits_only) {
                try {
                    return static_cast<uint64_t>(std::stoull(s));
                } catch (...) {
                    // fall through to ISO parser
                }
            }
            if (auto iso = sep::io::parse_iso8601_ms(s); iso) {
                return *iso;
            }
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace

nlohmann::json sep::Candle::toJson() const {
    return {
        {"t", timestamp},
        {"o", open},
        {"h", high},
        {"l", low},
        {"c", close},
        {"v", volume}
    };
}

sep::Candle sep::Candle::fromJson(const nlohmann::json& j) {
    Candle c{};

    auto timestamp = [&]() -> std::optional<uint64_t> {
        if (auto it = j.find("t"); it != j.end()) {
            if (auto ts = parseTimestampField(*it)) {
                return ts;
            }
        }
        if (auto it = j.find("timestamp"); it != j.end()) {
            if (auto ts = parseTimestampField(*it)) {
                return ts;
            }
        }
        if (auto it = j.find("time"); it != j.end()) {
            if (auto ts = parseTimestampField(*it)) {
                return ts;
            }
        }
        return std::nullopt;
    }();

    if (!timestamp) {
        throw std::runtime_error("candle JSON missing timestamp");
    }
    c.timestamp = *timestamp;

    auto mid_it = j.find("mid");
    auto bid_it = j.find("bid");
    auto ask_it = j.find("ask");
    const nlohmann::json* mid = (mid_it != j.end() && mid_it->is_object()) ? &(*mid_it) : nullptr;
    const nlohmann::json* bid = (bid_it != j.end() && bid_it->is_object()) ? &(*bid_it) : nullptr;
    const nlohmann::json* ask = (ask_it != j.end() && ask_it->is_object()) ? &(*ask_it) : nullptr;

    const std::array<const nlohmann::json*, 4> sources{&j, mid, bid, ask};

    auto pickPrice = [&](std::initializer_list<const char*> keys) -> std::optional<double> {
        for (const auto* src : sources) {
            if (!src) {
                continue;
            }
            for (const char* key : keys) {
                auto it = src->find(key);
                if (it == src->end()) {
                    continue;
                }
                if (auto v = parseNumericField(*it); v && std::isfinite(*v)) {
                    return *v;
                }
            }
        }
        return std::nullopt;
    };

    auto requirePrice = [&](std::initializer_list<const char*> keys, const char* label) -> double {
        if (auto v = pickPrice(keys)) {
            return *v;
        }
        throw std::runtime_error(std::string("candle JSON missing numeric field: ") + label);
    };

    c.open = requirePrice({"o", "open"}, "open");
    c.high = requirePrice({"h", "high"}, "high");
    c.low = requirePrice({"l", "low"}, "low");
    c.close = requirePrice({"c", "close"}, "close");

    auto volume = pickPrice({"v", "volume"});
    c.volume = (volume && std::isfinite(*volume)) ? *volume : 0.0;

    return c;
}

namespace sep {

redisContext* connectValkey(const std::string& url) {
    // Robust URL parsing supporting:
    //  - redis://user:pass@host:port/db
    //  - redis://host:port/db (no auth)
    //  - rediss://user:pass@host:port/db (TLS offload handled externally)
    char user[256] = {0}, pass[256] = {0}, host[256] = {0};
    int port = 6379;
    int db = 0;
    bool have_auth = false;

    if (sscanf(url.c_str(), "rediss://%[^:]:%[^ @]@%[^:]:%d/%d", user, pass, host, &port, &db) == 5 ||
        sscanf(url.c_str(), "redis://%[^:]:%[^ @]@%[^:]:%d/%d", user, pass, host, &port, &db) == 5) {
        have_auth = true;
    } else {
        // Try no-auth form: redis://host:port/db
        if (sscanf(url.c_str(), "redis://%[^:]:%d/%d", host, &port, &db) != 3) {
            spdlog::error("Invalid Valkey URL format: {}", url);
            return nullptr;
        }
    }

    timeval timeout = {1, 500000};  // 1.5s
    redisContext* c = redisConnectWithTimeout(host, port, timeout);
    if (c == nullptr || c->err) {
        spdlog::error("Valkey connect err: {}", c ? c->errstr : "alloc fail");
        if (c) redisFree(c);
        return nullptr;
    }

    if (have_auth) {
        redisReply* r = (redisReply*)redisCommand(c, "AUTH %s", pass);
        if (!r || r->type == REDIS_REPLY_ERROR) {
            spdlog::error("Auth fail");
            if (r) freeReplyObject(r);
            redisFree(c);
            return nullptr;
        }
        freeReplyObject(r);
    }
    // Select DB if needed
    redisCommand(c, "SELECT %d", db);
    return c;
}

void storeCandleZSET(redisContext* c, const std::string& key, const Candle& candle) {
    std::string member = candle.toJson().dump();
    redisCommand(c, "ZADD %s %llu %b", key.c_str(), candle.timestamp, member.data(), member.size());
}

std::vector<Candle> fetchCandlesByScore(redisContext* c, const std::string& key, uint64_t min_ts, uint64_t max_ts) {
    redisReply* r = (redisReply*)redisCommand(c, "ZRANGEBYSCORE %s %llu %llu", key.c_str(), min_ts, max_ts);
    std::vector<Candle> candles;
    if (r->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < r->elements; ++i) {
            std::string json_str(r->element[i]->str, r->element[i]->len);
            candles.push_back(Candle::fromJson(nlohmann::json::parse(json_str)));
        }
    }
    freeReplyObject(r);
    return candles;
}

// --- Compression helpers and manifold storage ---

std::string gzipCompress(const std::string& data) {
    z_stream zs{};
    if (deflateInit2(&zs, Z_BEST_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        throw std::runtime_error("deflateInit2 failed");
    }
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
    zs.avail_in = static_cast<uInt>(data.size());
    std::string out;
    out.resize(std::max<size_t>(128, data.size() / 2));
    int ret;
    do {
        if (zs.total_out >= out.size()) out.resize(out.size() * 2);
        zs.next_out = reinterpret_cast<Bytef*>(&out[zs.total_out]);
        zs.avail_out = static_cast<uInt>(out.size() - zs.total_out);
        ret = deflate(&zs, zs.avail_in ? Z_NO_FLUSH : Z_FINISH);
        if (ret == Z_STREAM_ERROR) {
            deflateEnd(&zs);
            throw std::runtime_error("deflate stream error");
        }
    } while (ret != Z_STREAM_END);
    deflateEnd(&zs);
    out.resize(zs.total_out);
    return out;
}

std::string gzipUncompress(const std::string& compressed) {
    z_stream zs{};
    if (inflateInit2(&zs, 15 + 32) != Z_OK) {
        throw std::runtime_error("inflateInit2 failed");
    }
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(compressed.data()));
    zs.avail_in = static_cast<uInt>(compressed.size());
    std::string out;
    out.resize(std::max<size_t>(256, compressed.size() * 2));
    int ret;
    do {
        if (zs.total_out >= out.size()) out.resize(out.size() * 2);
        zs.next_out = reinterpret_cast<Bytef*>(&out[zs.total_out]);
        zs.avail_out = static_cast<uInt>(out.size() - zs.total_out);
        ret = inflate(&zs, Z_NO_FLUSH);
        if (ret == Z_STREAM_ERROR || ret == Z_DATA_ERROR) {
            inflateEnd(&zs);
            throw std::runtime_error("inflate stream/data error");
        }
    } while (ret != Z_STREAM_END);
    inflateEnd(&zs);
    out.resize(zs.total_out);
    return out;
}

void storeGzipManifold(redisContext* c, const std::string& key, const nlohmann::json& manifold, int ttl_days) {
    std::string data = manifold.dump();
    std::string gz = gzipCompress(data);
    int ttl = ttl_days * 86400;
    redisReply* r = (redisReply*)redisCommand(c, "SET %b %b EX %d", key.data(), (size_t)key.size(), gz.data(), (size_t)gz.size(), ttl);
    if (!r || r->type == REDIS_REPLY_ERROR) {
        spdlog::error("SET {} failed", key);
    }
    if (r) freeReplyObject(r);
}

nlohmann::json fetchUnzipManifold(redisContext* c, const std::string& key) {
    nlohmann::json j;
    redisReply* r = (redisReply*)redisCommand(c, "GET %s", key.c_str());
    if (!r || r->type != REDIS_REPLY_STRING) {
        if (r) freeReplyObject(r);
        return j;
    }
    std::string comp(r->str, r->len);
    freeReplyObject(r);
    try {
        std::string data = gzipUncompress(comp);
        j = nlohmann::json::parse(data);
    } catch (const std::exception& e) {
        spdlog::error("Unzip/parse {} failed: {}", key, e.what());
    }
    return j;
}

void recomputeHotBand(redisContext* c, const std::string& instrument, int days) {
    uint64_t now = time(nullptr) * 1000;
    uint64_t start = now - (days * 86400000LL);
    auto key = "md:candles:" + instrument + ":M1";
    auto candles = fetchCandlesByScore(c, key, start, now);
    
    // Build manifold with quantum pattern analysis (integrated from manifold_generator_main.cpp)
    nlohmann::json manifold = buildManifoldFromCandles(candles, instrument);
    
    // Store compressed manifold data with TTL
    storeGzipManifold(c, "manifold:" + instrument + ":" + std::to_string(now / 86400000), manifold);
}

// Quantum pattern analysis function (extracted from manifold_generator_main.cpp)
nlohmann::json buildManifoldFromCandles(const std::vector<Candle>& candles, const std::string& instrument) {
    return buildManifold(candles, instrument);
}

}  // namespace sep
