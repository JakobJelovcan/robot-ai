#include <boost/program_options.hpp>
#include <format>
#include <iostream>
#include <robot-ai/llama_wrapper.hpp>

void parse_args(int argc, char* argv[], lma::llama_config& llama_config)
{
    // clang-format off
    namespace po = boost::program_options;
    po::options_description desc{"whisper options"};
    desc.add_options()
        ("help,h",                                      "Print help")
        ("threads,t",       po::value<int32_t>(),       "Number of threads")
        ("gpu-layers",      po::value<int32_t>(),       "GPU layers")
        ("no-gpu",                                      "Don't use gpu")
        ("llama-model",     po::value<std::string>(),   "llama model")
        ("llama-context",   po::value<std::string>(),   "llama context");

    po::variables_map variable_map;
    po::store(po::parse_command_line(argc, argv, desc), variable_map);
    po::notify(variable_map);

    if (variable_map.count("help") != 0u)
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (variable_map.count("threads") != 0u)
        llama_config.n_threads = variable_map["threads"].as<int32_t>();

    if (variable_map.count("gpu-layers") != 0u)
        llama_config.n_gpu_layers = variable_map["gpu-layers"].as<int32_t>();

    if (variable_map.count("no-gpu") != 0u)
        llama_config.use_gpu = false;

    if (variable_map.count("llama-model") != 0u)
        llama_config.model = variable_map["llama-model"].as<std::string>();

    if (variable_map.count("llama-context") != 0u)
        llama_config.context = variable_map["llama-context"].as<std::string>();

    // clang-format on
}

auto main(int argc, char* argv[]) -> int
{
    auto config = lma::llama_get_default_config();
    parse_args(argc, argv, config);

    auto llama = lma::llama::build_llama(config);

    if (!llama)
        return 1;

    llama->init();

    while (true)
    {
        std::cout << "You: ";
        std::string prompt;

        std::getline(std::cin, prompt);

        if (prompt.empty())
            break;

        std::cout << std::format("Darko: {}", llama->generate_from_prompt(prompt)) << std::endl;
    }

    return 0;
}