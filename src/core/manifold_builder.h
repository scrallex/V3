#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trading_signals.h"

namespace sep {

// Shared manifold generation helper used across CLI tools and services.
nlohmann::json buildManifold(const std::vector<Candle>& candles, const std::string& instrument_hint);

}  // namespace sep

