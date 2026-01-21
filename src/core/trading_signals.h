// ... existing enums/structs ...

#pragma once

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <hiredis/hiredis.h>
#include <zlib.h>

namespace sep {

// Candle from docs
struct Candle {
    uint64_t timestamp;
    double open, high, low, close, volume;
    nlohmann::json toJson() const;
    static Candle fromJson(const nlohmann::json& j);
};

// Valkey connection (technical: hiredis redisConnectWithTimeout, auth with redisCommand("AUTH pw"))
redisContext* connectValkey(const std::string& url);  // Parse rediss://:pw@host:port/0

// Store candle to ZSET (technical: ZADD key score member, score=timestamp, member=toJson().dump())
void storeCandleZSET(redisContext* c, const std::string& key, const Candle& candle);

// Fetch candles by score range (technical: ZRANGEBYSCORE key min max, parse members as JSON)
std::vector<Candle> fetchCandlesByScore(redisContext* c, const std::string& key, uint64_t min_ts, uint64_t max_ts);

// Store gzip manifold JSON (technical: SET key compressed, EX days*86400 for TTL)
void storeGzipManifold(redisContext* c, const std::string& key, const nlohmann::json& manifold, int ttl_days = 35);

// Fetch and unzip (technical: GET key, then uncompress)
nlohmann::json fetchUnzipManifold(redisContext* c, const std::string& key);

// Gzip compress (technical: z_stream deflateInit2 level=Z_BEST_COMPRESSION, windowBits=31 for gzip)
std::string gzipCompress(const std::string& data);

// Uncompress (technical: inflateInit2 windowBits=47 for auto-detect gzip)
std::string gzipUncompress(const std::string& compressed);

// Quantum pattern analysis and manifold generation
void recomputeHotBand(redisContext* c, const std::string& instrument, int days = 30);
nlohmann::json buildManifoldFromCandles(const std::vector<Candle>& candles, const std::string& instrument);

}  // namespace sep
