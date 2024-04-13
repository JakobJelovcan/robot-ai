#include <opendaq/opendaq.h>
#include <boost/asio.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/serial_port.hpp>
#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <robot-ai/llama_wrapper.hpp>
#include <robot-ai/whisper_wrapper.hpp>

using namespace std::chrono_literals;
using namespace std::string_literals;

struct robot_config
{
    std::string serial_port;
    std::string robot_ip;
    int32_t baud_rate;
    int32_t byte_size;
};

auto robot_get_default_config() -> robot_config;
void parse_args(int argc, char* argv[], whs::whisper_config& whisper_config, lma::llama_config& llama_config, robot_config& robot_config);
void process_llama_response(const std::string& rsp, boost::asio::serial_port& port, daq::FunctionBlockPtr& fb);
auto get_robot_fb(daq::DevicePtr& device) -> daq::FunctionBlockPtr;

auto main(int argc, char* argv[]) -> int
{
    // llama & whisper arguments
    auto whisper_config = whs::whisper_get_default_config();
    auto llama_config = lma::llama_get_default_config();
    auto robot_config = robot_get_default_config();
    parse_args(argc, argv, whisper_config, llama_config, robot_config);

    // serial port
    boost::asio::io_service io_service;
    boost::asio::serial_port serial_port{io_service, robot_config.serial_port};
    serial_port.set_option(boost::asio::serial_port::baud_rate(robot_config.baud_rate));
    serial_port.set_option(boost::asio::serial_port::character_size(robot_config.byte_size));
    serial_port.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
    serial_port.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
    serial_port.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));

    // openDAQ device & function_block
    const auto instance = daq::Instance();
    auto device = instance.addDevice(std::format("daq.opcua://{}", robot_config.robot_ip));
    auto robot_fb = get_robot_fb(device);

    // llama & whisper init
    auto whisper = whs::whisper::build_whisper(whisper_config);
    auto llama = lma::llama::build_llama(llama_config);

    if (!whisper || !llama)
        exit(EXIT_FAILURE);

    // llama & whisper start
    whisper->on_command = [&](const std::string& rsp) { process_llama_response(rsp, serial_port, robot_fb); };
    whisper->start_whisper();
    llama->init();

    std::cout << "Press \"enter\" to exit..." << std::endl;
    std::cin.get();

    whisper->stop_whisper();

    return 0;
}

void parse_args(int argc, char* argv[], whs::whisper_config& whisper_config, lma::llama_config& llama_config, robot_config& robot_config)
{
    // clang-format off
    namespace po = boost::program_options;
    po::options_description desc{"whisper options"};
    desc.add_options()
        ("help,h",                                      "Print help")
        ("threads,t",       po::value<int32_t>(),       "Number of threads")
        ("gpu-layers",      po::value<int32_t>(),       "GPU layers")
        ("audio-ctx",       po::value<int32_t>(),       "Audio context size")
        ("vad-thold",       po::value<float>(),         "Vad threshold")
        ("freq-thold",      po::value<float>(),         "Frequency threshold")
        ("no-gpu",                                      "Don't use gpu")
        ("whisper-model",   po::value<std::string>(),   "whisper model")
        ("llama-model",     po::value<std::string>(),   "llama model")
        ("commands",        po::value<std::string>(),   "Command file name")
        ("llama-context",   po::value<std::string>(),   "llama context")
        ("whisper-context", po::value<std::string>(),   "whisper context")
        ("serial-port",     po::value<std::string>(),   "serial port")
        ("baud-rate",       po::value<int32_t>(),       "baud rate")
        ("byte-size",       po::value<int32_t>(),       "byte size")
        ("robot-ip",        po::value<std::string>(),   "robot ip");

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

    if (variable_map.count("serial-port") != 0u)
        robot_config.serial_port = variable_map["serial-port"].as<std::string>();

    if (variable_map.count("robot-ip") != 0u)
        robot_config.robot_ip = variable_map["robot-ip"].as<std::string>();

    if (variable_map.count("baud-rate") != 0u)
        robot_config.baud_rate = variable_map["baud-rate"].as<int32_t>();

    if (variable_map.count("byte-size") != 0u)
        robot_config.byte_size = variable_map["byte-size"].as<int32_t>();

    // clang-format on
}

void process_llama_response(const std::string& rsp, boost::asio::serial_port& port, daq::FunctionBlockPtr& fb)
{
    const auto pour_beer = std::regex_search(rsp, std::regex("^.*(\\*pours.*beer\\*).*$"));

    if (pour_beer)
    {
        const std::array<uint8_t, 2> data = {0, 1};
        boost::asio::write(port, boost::asio::buffer(data.data(), data.size()));
    }

    const auto speech = std::regex_replace(rsp, std::regex(R"((\[.*?\])|(\(.*?\))|([^a-zA-Z0-9\.,\?!\s\:\'\-]))"), "");

    if (fb.assigned())
    {
        daq::ProcedurePtr procedure = fb.getPropertyValue("InvokeCommand");
        procedure(speech);
    }
}

auto get_robot_fb(daq::DevicePtr& device) -> daq::FunctionBlockPtr
{
    if (!device.assigned())
        return nullptr;

    for (auto fb : device.getFunctionBlocks())
    {
        if (fb.getName() == "robot_control_0")
            return fb;
    }
    return nullptr;
}

auto robot_get_default_config() -> robot_config
{
    return {.serial_port = "COM7", .robot_ip = "192.168.10.1", .baud_rate = 9600, .byte_size = 8};
}