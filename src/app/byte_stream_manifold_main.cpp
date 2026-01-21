#include "../core/byte_stream_manifold.h"

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using sep::ByteStreamConfig;

namespace {

struct CLIConfig {
    std::optional<std::string> input_path;
    std::optional<int> input_fd;
    size_t max_bytes = 0;
    std::string output_path;
    std::string format = "json";
    bool pretty = false;
};

ByteStreamConfig build_config(const cxxopts::ParseResult& args) {
    ByteStreamConfig config;
    const size_t window_bits_from_bits = args.count("window-bits") ? args["window-bits"].as<size_t>() : 0;
    const size_t window_bits_from_bytes = args.count("window-bytes") ? args["window-bytes"].as<size_t>() * 8 : 0;
    if (window_bits_from_bits > 0) {
        config.window_bits = window_bits_from_bits;
    } else if (window_bits_from_bytes > 0) {
        config.window_bits = window_bits_from_bytes;
    }

    const size_t step_bits_from_bits = args.count("step-bits") ? args["step-bits"].as<size_t>() : 0;
    const size_t step_bits_from_bytes = args.count("step-bytes") ? args["step-bytes"].as<size_t>() * 8 : 0;
    if (step_bits_from_bits > 0) {
        config.step_bits = step_bits_from_bits;
    } else if (step_bits_from_bytes > 0) {
        config.step_bits = step_bits_from_bytes;
    }

    if (args.count("max-windows")) {
        config.max_windows = args["max-windows"].as<size_t>();
    }
    if (args.count("msb-first")) {
        config.lsb_first = !args["msb-first"].as<bool>();
    }
    if (args.count("repetition-lookback")) {
        config.repetition_lookback = args["repetition-lookback"].as<size_t>();
    }
    if (args.count("signature-precision")) {
        config.signature_precision = std::clamp(args["signature-precision"].as<int>(), 0, 6);
    }

    if (args.count("coherence-threshold")) {
        config.qfh_options.coherence_threshold = args["coherence-threshold"].as<double>();
    }
    if (args.count("stability-threshold")) {
        config.qfh_options.stability_threshold = args["stability-threshold"].as<double>();
    }
    if (args.count("collapse-threshold")) {
        config.qfh_options.collapse_threshold = args["collapse-threshold"].as<double>();
    }
    if (args.count("max-iterations")) {
        config.qfh_options.max_iterations = static_cast<int>(args["max-iterations"].as<size_t>());
    }
    if (args.count("disable-damping")) {
        config.qfh_options.enable_damping = !args["disable-damping"].as<bool>();
    }
    if (args.count("damping-factor")) {
        config.qfh_options.damping_factor = args["damping-factor"].as<double>();
    }
    if (args.count("entropy-weight")) {
        config.qfh_options.entropy_weight = args["entropy-weight"].as<double>();
    }
    if (args.count("coherence-weight")) {
        config.qfh_options.coherence_weight = args["coherence-weight"].as<double>();
    }

    return config;
}

CLIConfig build_cli_config(const cxxopts::ParseResult& args) {
    CLIConfig cli;
    if (args.count("input")) {
        cli.input_path = args["input"].as<std::string>();
    }
    if (args.count("fd")) {
        cli.input_fd = args["fd"].as<int>();
    }
    cli.max_bytes = args["max-bytes"].as<size_t>();
    if (args.count("format")) {
        cli.format = args["format"].as<std::string>();
    }
    if (args.count("output")) {
        cli.output_path = args["output"].as<std::string>();
    }
    cli.pretty = args["pretty"].as<bool>();
    return cli;
}

std::vector<uint8_t> load_bytes(const CLIConfig& cli) {
    try {
        if (cli.input_path) {
            return sep::load_bytes_from_file(*cli.input_path, cli.max_bytes);
        }
        if (cli.input_fd) {
            return sep::load_bytes_from_fd(*cli.input_fd, cli.max_bytes);
        }
        // Default to stdin when no explicit source is provided.
        spdlog::info("Reading byte stream from stdin (fd=0)");
        return sep::load_bytes_from_fd(0, cli.max_bytes);
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("Failed to load byte stream: ") + ex.what());
    }
}

void write_output(const CLIConfig& cli, const std::string& payload) {
    if (cli.output_path.empty()) {
        std::cout << payload;
        if (!payload.empty() && payload.back() != '\n') {
            std::cout << '\n';
        }
        return;
    }
    std::ofstream ofs(cli.output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.good()) {
        throw std::runtime_error("Failed to open output file: " + cli.output_path);
    }
    ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

}  // namespace

int main(int argc, char** argv) {
    cxxopts::Options options("byte_stream_manifold", "Run QFH/SRI analysis on arbitrary byte streams");

    options.add_options()
        ("input", "Path to input file", cxxopts::value<std::string>())
        ("fd", "File descriptor/socket to read (defaults to stdin when omitted)", cxxopts::value<int>())
        ("max-bytes", "Maximum bytes to read from input (0 = all)", cxxopts::value<size_t>()->default_value("0"))
        ("window-bits", "Sliding window length in bits", cxxopts::value<size_t>())
        ("window-bytes", "Sliding window length in bytes", cxxopts::value<size_t>())
        ("step-bits", "Stride between windows in bits", cxxopts::value<size_t>())
        ("step-bytes", "Stride between windows in bytes", cxxopts::value<size_t>())
        ("max-windows", "Cap the number of windows analysed (0 = all)", cxxopts::value<size_t>()->default_value("0"))
        ("msb-first", "Interpret bytes MSB first instead of LSB first", cxxopts::value<bool>()->default_value("false"))
        ("repetition-lookback", "Lookback horizon (in windows) for repetition counting (0 = entire run)", cxxopts::value<size_t>()->default_value("0"))
        ("signature-precision", "Decimal places used when bucketing signatures", cxxopts::value<int>()->default_value("2"))
        ("coherence-threshold", "QFH coherence collapse threshold", cxxopts::value<double>())
        ("stability-threshold", "QFH stability collapse threshold", cxxopts::value<double>())
        ("collapse-threshold", "QFH rupture collapse threshold", cxxopts::value<double>())
        ("max-iterations", "Maximum iterations for QFH analyser", cxxopts::value<size_t>())
        ("disable-damping", "Disable trajectory damping", cxxopts::value<bool>()->default_value("false"))
        ("damping-factor", "Custom damping factor", cxxopts::value<double>())
        ("entropy-weight", "Weight applied to entropy when computing lambda", cxxopts::value<double>())
        ("coherence-weight", "Weight applied to (1 - coherence) when computing lambda", cxxopts::value<double>())
        ("format", "Output format: json|csv", cxxopts::value<std::string>()->default_value("json"))
        ("output", "Write results to file instead of stdout", cxxopts::value<std::string>())
        ("pretty", "Pretty-print JSON output", cxxopts::value<bool>()->default_value("false"))
        ("help", "Show usage");

    cxxopts::ParseResult args;
    try {
        args = options.parse(argc, argv);
    } catch (const std::exception& ex) {
        spdlog::error("Failed to parse arguments: {}", ex.what());
        std::cerr << options.help() << std::endl;
        return 1;
    }

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!args.count("input") && !args.count("fd")) {
        spdlog::info("No --input or --fd provided; consuming stdin (fd=0).");
    }

    const CLIConfig cli = build_cli_config(args);
    ByteStreamConfig config = build_config(args);

    if (config.window_bits == 0) {
        spdlog::warn("Window length defaults to 256 bits.");
        config.window_bits = 256;
    }
    if (config.step_bits == 0) {
        config.step_bits = std::max<size_t>(1, config.window_bits / 4);
    }

    std::vector<uint8_t> bytes;
    try {
        bytes = load_bytes(cli);
    } catch (const std::exception& ex) {
        spdlog::error("{}", ex.what());
        return 2;
    }

    if (bytes.empty()) {
        spdlog::warn("No bytes read from input; nothing to analyse.");
    }

    sep::ByteStreamManifold manifold = sep::analyze_byte_stream(bytes, config);

    try {
        if (cli.format == "json") {
            auto doc = manifold.to_json(config);
            const std::string output = cli.pretty ? doc.dump(2) : doc.dump();
            write_output(cli, output);
        } else if (cli.format == "csv") {
            write_output(cli, manifold.to_csv());
        } else {
            spdlog::error("Unsupported format '{}'. Use json or csv.", cli.format);
            return 3;
        }
    } catch (const std::exception& ex) {
        spdlog::error("Failed to serialise output: {}", ex.what());
        return 4;
    }

    spdlog::info("Analysed {} windows across {} bytes ({} bits).",
                 manifold.summary.total_windows,
                 manifold.summary.analysed_bytes,
                 manifold.summary.analysed_bits);
    spdlog::info("Mean coherence {:.4f}, entropy {:.4f}, lambda {:.4f}.",
                 manifold.summary.mean_coherence,
                 manifold.summary.mean_entropy,
                 manifold.summary.mean_lambda);
    return 0;
}
