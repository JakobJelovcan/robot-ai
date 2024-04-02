#include <opendaq/opendaq.h>
#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <openDAQ-ai/whisper_wrapper.hpp>

using namespace std::chrono_literals;

daq::DevicePtr obsidian_dev;
daq::FunctionBlockPtr robot_fb;

void parse_args(int argc, char* argv[], whs::whisper_config& config);
void send_command(const std::string& command);
daq::FunctionBlockPtr find_robot_fb(daq::DevicePtr& device);

auto main(int argc, char* argv[]) -> int
{
    const auto instance = daq::Instance();
    obsidian_dev = instance.addDevice("daq.opcua://192.168.10.1");
    robot_fb = find_robot_fb(obsidian_dev);

    auto whisper_config = whs::whisper_get_default_config();
    auto whisper = whs::whisper::build_whisper(whisper_config);

    if (!whisper)
        exit(EXIT_FAILURE);

    whisper->on_command = send_command;

    whisper->whisper_loop();

    return 0;
}

void parse_args(int argc, char* argv[], whs::whisper_config& config)
{
    // clang-format off
    namespace po = boost::program_options;
    po::options_description desc{"whisper options"};
    desc.add_options()
        ("help,h",                                      "Print help")
        ("threads,t",       po::value<int32_t>(),       "Number of threads")
        ("prompt-ms,pms",   po::value<int32_t>(),       "Prompt ms")
        ("command-ms,cms",  po::value<int32_t>(),       "Command ms")
        ("capture,c",       po::value<int32_t>(),       "Capture device id")
        ("max-tokens,mt",   po::value<int32_t>(),       "Max tokens")
        ("audio-ctx,ac",    po::value<int32_t>(),       "Audio context")
        ("vad-thold,vth",   po::value<float>(),         "Vad threshold")
        ("freq-thold,fth",  po::value<float>(),         "Frequency threshold")
        ("no-gpu,ng",                                   "Don't use gpu")
        ("model,m",         po::value<std::string>(),   "Model")
        ("commands,cmd",    po::value<std::string>(),   "Command file name")
        ("prompt,p",        po::value<std::string>(),   "Prompt")
        ("context,ctx",     po::value<std::string>(),   "Context");

    po::variables_map variable_map;
    po::store(po::parse_command_line(argc, argv, desc), variable_map);
    po::notify(variable_map);

    if (variable_map.count("help") != 0u)
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (variable_map.count("threads") != 0u)
        config.n_threads = variable_map["threads"].as<int32_t>();

    if (variable_map.count("prompt-ms") != 0u)
        config.prompt_ms = variable_map["prompt-ms"].as<int32_t>();

    if (variable_map.count("command-ms") != 0u)
        config.command_ms = variable_map["command-ms"].as<int32_t>();

    if (variable_map.count("capture") != 0u)
        config.capture_id = variable_map["capture"].as<int32_t>();

    if (variable_map.count("max-tokens") != 0u)
        config.max_tokens = variable_map["max-tokens"].as<int32_t>();

    if (variable_map.count("audio-ctx") != 0u)
        config.audio_ctx = variable_map["audio-ctx"].as<int32_t>();

    if (variable_map.count("vad-thold") != 0u)
        config.vad_threshold = variable_map["vad-thold"].as<float>();

    if (variable_map.count("freq-thold") != 0u)
        config.freq_threshold = variable_map["freq-thold"].as<float>();

    if (variable_map.count("no-gpu") != 0u)
        config.use_gpu = false;

    if (variable_map.count("model") != 0u)
        config.model = variable_map["model"].as<std::string>();

    if (variable_map.count("commands") != 0u)
        config.commands = variable_map["commands"].as<std::string>();

    if (variable_map.count("prompt") != 0u)
        config.prompt = variable_map["prompt"].as<std::string>();

    if (variable_map.count("context") != 0u)
        config.context = variable_map["context"].as<std::string>();

    // clang-format on
}

void send_command(const std::string& command)
{
    if (!robot_fb.assigned())
        return;

    daq::ProcedurePtr procedure = robot_fb.getPropertyValue("InvokeCommand");
    procedure(command);

    std::cout << std::format("[main] Invoked command: {}", command) << std::endl;
}

daq::FunctionBlockPtr find_robot_fb(daq::DevicePtr& device)
{
    for (auto fb : device.getFunctionBlocks())
    {
        if (fb.getName() == "robot_control_0")
            return fb;
    }
    return nullptr;
}
