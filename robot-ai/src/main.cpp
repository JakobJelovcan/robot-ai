#include <opendaq/opendaq.h>
#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <robot-ai/llama_wrapper.hpp>
#include <robot-ai/whisper_wrapper.hpp>

using namespace std::chrono_literals;

void parse_args(int argc, char* argv[], whs::whisper_config& whisper_config, lma::llama_config& llama_config);
void invoke_command(daq::FunctionBlockPtr& fb, const std::string& command);

auto get_robot_fb(daq::DevicePtr& device) -> daq::FunctionBlockPtr;

auto main(int argc, char* argv[]) -> int
{
    auto whisper_config = whs::whisper_get_default_config();
    auto llama_config = lma::llama_get_default_config();
    parse_args(argc, argv, whisper_config, llama_config);

    const auto instance = daq::Instance();
    auto device = instance.addDevice("daq.opcua://192.168.10.1");
    auto robot_fb = get_robot_fb(device);

    auto whisper = whs::whisper::build_whisper(whisper_config);
    auto llama = lma::llama::build_llama(llama_config);

    if (!whisper || !llama)
        exit(EXIT_FAILURE);

    whisper->on_command = [&](const std::string& cmd) { invoke_command(robot_fb, cmd); };

    whisper->whisper_loop();

    return 0;
}

void parse_args(int argc, char* argv[], whs::whisper_config& whisper_config, lma::llama_config& llama_config)
{
    // clang-format off
    namespace po = boost::program_options;
    po::options_description desc{"whisper options"};
    desc.add_options()
        ("help,h",                                      "Print help")
        ("threads,t",       po::value<int32_t>(),       "Number of threads")
        ("gpu-layers",      po::value<int32_t>(),       "Number of gpu layers")
        ("prompt-ms",       po::value<int32_t>(),       "Prompt ms")
        ("command-ms",      po::value<int32_t>(),       "Command ms")
        ("capture,c",       po::value<int32_t>(),       "Capture device id")
        ("max-tokens",      po::value<int32_t>(),       "Max tokens")
        ("audio-ctx",       po::value<int32_t>(),       "Audio context")
        ("vad-thold",       po::value<float>(),         "Vad threshold")
        ("freq-thold",      po::value<float>(),         "Frequency threshold")
        ("no-gpu",                                      "Don't use gpu")
        ("whisper-model",   po::value<std::string>(),   "whisper model")
        ("llama-model",     po::value<std::string>(),   "llama model")
        ("commands",        po::value<std::string>(),   "Command file name")
        ("prompt,p",        po::value<std::string>(),   "Prompt")
        ("context",         po::value<std::string>(),   "Context");

    po::variables_map variable_map;
    po::store(po::parse_command_line(argc, argv, desc), variable_map);
    po::notify(variable_map);

    if (variable_map.count("help") != 0u)
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (variable_map.count("threads") != 0u)
        llama_config.n_threads = whisper_config.n_threads = variable_map["threads"].as<int32_t>();

    if (variable_map.count("gpu-layers") != 0u)
        llama_config.n_gpu_layers = variable_map["gpu-layers"].as<int32_t>();

    if (variable_map.count("prompt-ms") != 0u)
        whisper_config.prompt_ms = variable_map["prompt-ms"].as<int32_t>();

    if (variable_map.count("command-ms") != 0u)
        whisper_config.command_ms = variable_map["command-ms"].as<int32_t>();

    if (variable_map.count("capture") != 0u)
        whisper_config.capture_id = variable_map["capture"].as<int32_t>();

    if (variable_map.count("max-tokens") != 0u)
        whisper_config.max_tokens = variable_map["max-tokens"].as<int32_t>();

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

    if (variable_map.count("llama-model") != 0u)
        llama_config.model = variable_map["llama-model"].as<std::string>();

    if (variable_map.count("commands") != 0u)
        whisper_config.commands = variable_map["commands"].as<std::string>();

    if (variable_map.count("prompt") != 0u)
        whisper_config.prompt = variable_map["prompt"].as<std::string>();

    if (variable_map.count("context") != 0u)
        whisper_config.context = variable_map["context"].as<std::string>();

    // clang-format on
}

void invoke_command(daq::FunctionBlockPtr& fb, const std::string& command)
{
    if (!fb.assigned())
        return;

    daq::ProcedurePtr procedure = fb.getPropertyValue("InvokeCommand");
    procedure(command);

    std::cout << std::format("[main] Invoked command: {}", command) << std::endl;
}

auto get_robot_fb(daq::DevicePtr& device) -> daq::FunctionBlockPtr
{
    for (auto fb : device.getFunctionBlocks())
    {
        if (fb.getName() == "robot_control_0")
            return fb;
    }
    return nullptr;
}