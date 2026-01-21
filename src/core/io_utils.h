#pragma once

#include "trading_signals.h"

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace sep::io {

std::optional<uint64_t> parse_yyyy_mm_dd_ms(const std::string& text);
std::optional<uint64_t> parse_iso8601_ms(const std::string& text);
std::string epochMsToRfc3339(uint64_t ms);
std::string toRfc3339Utc(std::chrono::system_clock::time_point tp);
std::vector<sep::Candle> load_candles_from_file(const std::string& path);

}  // namespace sep::io

