#include "io_utils.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace sep::io {

std::optional<uint64_t> parse_yyyy_mm_dd_ms(const std::string& s) {
    if (s.size() < 10) {
        return std::nullopt;
    }
    std::tm tm{};
    std::memset(&tm, 0, sizeof(tm));
    if (std::sscanf(s.c_str(), "%4d-%2d-%2d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday) != 3) {
        return std::nullopt;
    }
    tm.tm_year -= 1900;
    tm.tm_mon -= 1;
    tm.tm_hour = 0;
    tm.tm_min = 0;
    tm.tm_sec = 0;
#if defined(_WIN32)
    time_t t = _mkgmtime(&tm);
#else
    time_t t = timegm(&tm);
#endif
    if (t < 0) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(t) * 1000ULL;
}

std::optional<uint64_t> parse_iso8601_ms(const std::string& s) {
    if (s.empty()) {
        return std::nullopt;
    }
    const bool all_digits = std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
    if (all_digits) {
        try {
            return static_cast<uint64_t>(std::stoull(s));
        } catch (...) {
            // fall through to full parser
        }
    }

    auto tpos = s.find('T');
    if (tpos == std::string::npos) {
        tpos = s.find(' ');
    }
    if (tpos == std::string::npos || tpos < 10) {
        return std::nullopt;
    }
    std::string date = s.substr(0, tpos);
    std::string rest = s.substr(tpos + 1);

    int year = 0, mon = 0, day = 0;
    if (std::sscanf(date.c_str(), "%4d-%2d-%2d", &year, &mon, &day) != 3) {
        return std::nullopt;
    }

    int tz_sign = 0;
    int tz_hour = 0;
    int tz_min = 0;
    size_t tz_pos = std::string::npos;
    for (size_t i = 0; i < rest.size(); ++i) {
        char ch = rest[i];
        if (ch == 'Z' || ch == 'z' || ch == '+' || ch == '-') {
            tz_pos = i;
            break;
        }
    }
    std::string clock = rest;
    if (tz_pos != std::string::npos) {
        clock = rest.substr(0, tz_pos);
        char tz_char = rest[tz_pos];
        if (tz_char == 'Z' || tz_char == 'z') {
            tz_sign = 0;
        } else {
            tz_sign = tz_char == '-' ? -1 : 1;
            size_t off_start = tz_pos + 1;
            if (off_start + 2 <= rest.size()) {
                tz_hour = std::stoi(rest.substr(off_start, 2));
            }
            size_t mm_start = off_start + 2;
            if (mm_start < rest.size() && rest[mm_start] == ':') {
                ++mm_start;
            }
            if (mm_start + 2 <= rest.size()) {
                tz_min = std::stoi(rest.substr(mm_start, 2));
            }
        }
    }

    int hour = 0, minute = 0, sec = 0;
    int ms_frac = 0;
    if (clock.size() < 8) {
        return std::nullopt;
    }
    hour = std::stoi(clock.substr(0, 2));
    minute = std::stoi(clock.substr(3, 2));
    sec = std::stoi(clock.substr(6, 2));
    auto dot = clock.find('.');
    if (dot != std::string::npos) {
        std::string frac = clock.substr(dot + 1);
        int digits = 0;
        for (char c : frac) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                break;
            }
            if (digits < 3) {
                ms_frac = ms_frac * 10 + (c - '0');
            }
            ++digits;
        }
        while (digits > 0 && digits < 3) {
            ms_frac *= 10;
            ++digits;
        }
    }

    std::tm tm{};
    tm.tm_year = year - 1900;
    tm.tm_mon = mon - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = sec;
#if defined(_WIN32)
    time_t t = _mkgmtime(&tm);
#else
    time_t t = timegm(&tm);
#endif
    if (t < 0) {
        return std::nullopt;
    }

    int offset_seconds = tz_sign * (tz_hour * 3600 + tz_min * 60);
    auto ms = static_cast<int64_t>(t) * 1000LL + ms_frac - static_cast<int64_t>(offset_seconds) * 1000LL;
    if (ms < 0) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(ms);
}

std::string epochMsToRfc3339(uint64_t ms) {
    using namespace std::chrono;
    const auto sec = seconds(ms / 1000ULL);
    const auto ms_part = milliseconds(ms % 1000ULL);
    system_clock::time_point tp(sec);
#if defined(_WIN32)
    std::time_t tt = system_clock::to_time_t(tp);
    std::tm tm_utc{};
    gmtime_s(&tm_utc, &tt);
#else
    std::time_t tt = system_clock::to_time_t(tp);
    std::tm tm_utc{};
    gmtime_r(&tt, &tm_utc);
#endif
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_utc) == 0) {
        return {};
    }
    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << ms_part.count() << 'Z';
    return oss.str();
}

std::string toRfc3339Utc(std::chrono::system_clock::time_point tp) {
    using namespace std::chrono;
    const auto secs = time_point_cast<seconds>(tp);
    const auto remainder = duration_cast<milliseconds>(tp - secs).count();
    std::time_t tt = system_clock::to_time_t(secs);
#if defined(_WIN32)
    std::tm tm_utc{};
    gmtime_s(&tm_utc, &tt);
#else
    std::tm tm_utc{};
    gmtime_r(&tt, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_utc);
    std::ostringstream oss;
    oss << buf << '.' << std::setfill('0') << std::setw(3) << (remainder < 0 ? 0 : remainder) << 'Z';
    return oss.str();
}

std::vector<sep::Candle> load_candles_from_file(const std::string& path) {
    std::vector<sep::Candle> out;
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.good()) {
        spdlog::error("Failed to open input file: {}", path);
        return out;
    }
    std::string file_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    auto first_non = std::find_if(file_data.begin(), file_data.end(), [](unsigned char c) { return !std::isspace(c); });
    if (first_non == file_data.end()) {
        return out;
    }
    const char c0 = *first_non;

    auto parse_numeric = [](const nlohmann::json& value, const char* field) -> std::optional<double> {
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
                double v = std::stod(s, &idx);
                if (idx != s.size()) {
                    return std::nullopt;
                }
                return v;
            }
        } catch (const std::exception& e) {
            spdlog::warn("Invalid numeric field {}: {}", field, e.what());
        }
        return std::nullopt;
    };

    auto parse_candle_obj = [&](const nlohmann::json& j) -> std::optional<sep::Candle> {
        try {
            sep::Candle c{};
            if (j.contains("t")) {
                c.timestamp = j.at("t").get<uint64_t>();
            } else if (j.contains("timestamp")) {
                if (j["timestamp"].is_number_unsigned()) {
                    c.timestamp = j["timestamp"].get<uint64_t>();
                } else if (j["timestamp"].is_string()) {
                    auto ms = parse_iso8601_ms(j["timestamp"].get<std::string>());
                    if (!ms) {
                        return std::nullopt;
                    }
                    c.timestamp = *ms;
                }
            } else if (j.contains("time")) {
                auto ms = parse_iso8601_ms(j["time"].get<std::string>());
                if (!ms) {
                    return std::nullopt;
                }
                c.timestamp = *ms;
            } else {
                return std::nullopt;
            }

            if (const auto it = j.find("o"); it != j.end()) {
                auto v = parse_numeric(*it, "o");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.open = *v;
            } else if (const auto it2 = j.find("open"); it2 != j.end()) {
                auto v = parse_numeric(*it2, "open");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.open = *v;
            } else {
                return std::nullopt;
            }

            if (const auto it = j.find("h"); it != j.end()) {
                auto v = parse_numeric(*it, "h");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.high = *v;
            } else if (const auto it2 = j.find("high"); it2 != j.end()) {
                auto v = parse_numeric(*it2, "high");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.high = *v;
            } else {
                return std::nullopt;
            }

            if (const auto it = j.find("l"); it != j.end()) {
                auto v = parse_numeric(*it, "l");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.low = *v;
            } else if (const auto it2 = j.find("low"); it2 != j.end()) {
                auto v = parse_numeric(*it2, "low");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.low = *v;
            } else {
                return std::nullopt;
            }

            if (const auto it = j.find("c"); it != j.end()) {
                auto v = parse_numeric(*it, "c");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.close = *v;
            } else if (const auto it2 = j.find("close"); it2 != j.end()) {
                auto v = parse_numeric(*it2, "close");
                if (!v || !std::isfinite(*v)) return std::nullopt;
                c.close = *v;
            } else {
                return std::nullopt;
            }

            if (const auto it = j.find("v"); it != j.end()) {
                auto v = parse_numeric(*it, "v");
                c.volume = (v && std::isfinite(*v)) ? *v : 0.0;
            } else if (const auto it2 = j.find("volume"); it2 != j.end()) {
                auto v = parse_numeric(*it2, "volume");
                c.volume = (v && std::isfinite(*v)) ? *v : 0.0;
            } else {
                c.volume = 0.0;
            }
            if (!std::isfinite(c.open) || !std::isfinite(c.high) || !std::isfinite(c.low) || !std::isfinite(c.close)) {
                return std::nullopt;
            }
            return c;
        } catch (const std::exception& e) {
            spdlog::warn("Failed to parse candle entry: {}", e.what());
            return std::nullopt;
        }
    };

    try {
        if (c0 == '[' || c0 == '{') {
            nlohmann::json j = nlohmann::json::parse(file_data);
            if (j.is_array()) {
                for (const auto& elem : j) {
                    auto candle = parse_candle_obj(elem);
                    if (candle) {
                        out.push_back(*candle);
                    }
                }
            } else if (j.is_object()) {
                if (j.contains("candles") && j["candles"].is_array()) {
                    for (const auto& elem : j["candles"]) {
                        auto candle = parse_candle_obj(elem);
                        if (candle) {
                            out.push_back(*candle);
                        }
                    }
                } else {
                    auto candle = parse_candle_obj(j);
                    if (candle) {
                        out.push_back(*candle);
                    }
                }
            }
        } else {
            std::istringstream iss(file_data);
            std::string line;
            while (std::getline(iss, line)) {
                if (line.empty()) {
                    continue;
                }
                nlohmann::json j = nlohmann::json::parse(line);
                auto candle = parse_candle_obj(j);
                if (candle) {
                    out.push_back(*candle);
                }
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to parse candles from {}: {}", path, e.what());
        out.clear();
    }

    return out;
}

}  // namespace sep::io
