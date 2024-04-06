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
    whisper->start_whisper();
    llama->init_context();

    std::cout << "Press \"enter\" to exit..." << std::endl;
    std::cin.get();

    whisper->stop_whisper();

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
        ("audio-ctx",       po::value<int32_t>(),       "Audio context size")
        ("vad-thold",       po::value<float>(),         "Vad threshold")
        ("freq-thold",      po::value<float>(),         "Frequency threshold")
        ("no-gpu",                                      "Don't use gpu")
        ("whisper-model",   po::value<std::string>(),   "whisper model")
        ("llama-model",     po::value<std::string>(),   "llama model")
        ("commands",        po::value<std::string>(),   "Command file name")
        ("llama-context",   po::value<std::string>(),   "llama context")
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
        llama_config.n_threads = whisper_config.n_threads = variable_map["threads"].as<int32_t>();

    if (variable_map.count("capture") != 0u)
        whisper_config.capture_id = variable_map["capture"].as<int32_t>();

    if (variable_map.count("audio-ctx") != 0u)
        whisper_config.audio_ctx = variable_map["audio-ctx"].as<int32_t>();

    if (variable_map.count("vad-thold") != 0u)
        whisper_config.vad_threshold = variable_map["vad-thold"].as<float>();

    if (variable_map.count("freq-thold") != 0u)
        whisper_config.freq_threshold = variable_map["freq-thold"].as<float>();

    if (variable_map.count("no-gpu") != 0u)
        llama_config.use_gpu = whisper_config.use_gpu = false;

    if (variable_map.count("whisper-model") != 0u)
        whisper_config.model = variable_map["whisper-model"].as<std::string>();

    if (variable_map.count("llama-model") != 0u)
        llama_config.model = variable_map["llama-model"].as<std::string>();

    if (variable_map.count("commands") != 0u)
        whisper_config.commands = variable_map["commands"].as<std::string>();

    if (variable_map.count("whisper-context") != 0u)
        whisper_config.context = variable_map["whisper-context"].as<std::string>();

    if (variable_map.count("llama-context") != 0u)
        llama_config.context = variable_map["llama-context"].as<std::string>();

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