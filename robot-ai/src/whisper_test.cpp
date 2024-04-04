#include <iostream>
#include <format>
#include <boost/program_options.hpp>
#include <robot-ai/whisper_wrapper.hpp>

void parse_args(int argc, char* argv[], whs::whisper_config& whisper_config)
{
    // clang-format off
    namespace po = boost::program_options;
    po::options_description desc{"whisper options"};
    desc.add_options()
        ("help,h",                                      "Print help")
        ("threads,t",       po::value<int32_t>(),       "Number of threads")
        ("audio-ctx",       po::value<int32_t>(),       "Audio context size")
        ("vad-thold",       po::value<float>(),         "Vad threshold")
        ("freq-thold",      po::value<float>(),         "Frequency threshold")
        ("no-gpu",                                      "Don't use gpu")
        ("whisper-model",   po::value<std::string>(),   "whisper model")
        ("commands",        po::value<std::string>(),   "Command file name")
        ("whisper-context", po::value<std::string>(),   "whisper context");

    po::variables_map variable_map;
    po::store(po::parse_command_line(argc, argv, desc), variable_map);
    po::notify(variable_map);

    if (variable_map.count("help") != 0u)
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (variable_map.count("threads") != 0u)
        whisper_config.n_threads = variable_map["threads"].as<int32_t>();

    if (variable_map.count("capture") != 0u)
        whisper_config.capture_id = variable_map["capture"].as<int32_t>();

    if (variable_map.count("audio-ctx") != 0u)
        whisper_config.audio_ctx = variable_map["audio-ctx"].as<int32_t>();

    if (variable_map.count("vad-thold") != 0u)
        whisper_config.vad_threshold = variable_map["vad-thold"].as<float>();

    if (variable_map.count("freq-thold") != 0u)
        whisper_config.freq_threshold = variable_map["freq-thold"].as<float>();

    if (variable_map.count("no-gpu") != 0u)
        whisper_config.use_gpu = false;

    if (variable_map.count("whisper-model") != 0u)
        whisper_config.model = variable_map["whisper-model"].as<std::string>();

    if (variable_map.count("commands") != 0u)
        whisper_config.commands = variable_map["commands"].as<std::string>();

    if (variable_map.count("whisper-context") != 0u)
        whisper_config.context = variable_map["whisper-context"].as<std::string>();

    // clang-format on
}

auto main(int argc, char *argv[]) -> int
{
    auto config = whs::whisper_get_default_config();
    parse_args(argc, argv, config);
    auto whisper = whs::whisper::build_whisper(config);

    if (!whisper)
        return 1;

    whisper->on_command = [&](const auto& cmd) { std::cout << std::format("[whisper_test]: {}", cmd) << std::endl; };

    whisper->whisper_loop();

    return 0;
}