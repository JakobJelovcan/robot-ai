#include <openDAQ-ai/common.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <openDAQ-ai/main.hpp>
#include <strstream>

using namespace std::chrono_literals;

auto main(int argc, char* argv[]) -> int
{
    whisper_params params;
    parse_args(argc, argv, params);

    whisper_context_params_t cparams = whisper_context_default_params();
    cparams.use_gpu = params.use_gpu;

    whisper_context_t* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    audio_async audio{30 * 1000};
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE))
    {
        std::fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        std::exit(1);
    }

    audio.resume();
    std::this_thread::sleep_for(1s);
    audio.clear();

    process_commands(ctx, audio, params);

    whisper_free(ctx);

    return 0;
}

void parse_args(int argc, char* argv[], whisper_params& params)
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
        ("context,ctx",     po::value<std::string>(),   "Context")
        ("grammar",         po::value<std::string>(),   "Grammar")
        ("grammar-penalty", po::value<float>(),         "Grammar penalty");

    po::variables_map variable_map;
    po::store(po::parse_command_line(argc, argv, desc), variable_map);
    po::notify(variable_map);

    if (variable_map.count("help") != 0u)
    {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (variable_map.count("threads") != 0u)
        params.n_threads = variable_map["threads"].as<int32_t>();

    if (variable_map.count("prompt-ms") != 0u)
        params.prompt_ms = variable_map["prompt-ms"].as<int32_t>();

    if (variable_map.count("command-ms") != 0u)
        params.command_ms = variable_map["command-ms"].as<int32_t>();

    if (variable_map.count("capture") != 0u)
        params.capture_id = variable_map["capture"].as<int32_t>();

    if (variable_map.count("max-tokens") != 0u)
        params.max_tokens = variable_map["max-tokens"].as<int32_t>();

    if (variable_map.count("audio-ctx") != 0u)
        params.audio_ctx = variable_map["audio-ctx"].as<int32_t>();

    if (variable_map.count("vad-thold") != 0u)
        params.vad_thold = variable_map["vad-thold"].as<float>();

    if (variable_map.count("freq-thold") != 0u)
        params.freq_thold = variable_map["freq-thold"].as<float>();

    if (variable_map.count("no-gpu") != 0u)
        params.use_gpu = false;

    if (variable_map.count("model") != 0u)
        params.model = variable_map["model"].as<std::string>();

    if (variable_map.count("commands") != 0u)
        params.commands = variable_map["commands"].as<std::string>();

    if (variable_map.count("prompt") != 0u)
        params.prompt = variable_map["prompt"].as<std::string>();

    if (variable_map.count("context") != 0u)
        params.context = variable_map["context"].as<std::string>();

    if (variable_map.count("grammar") != 0u)
        params.grammar = variable_map["grammar"].as<std::string>();

    if (variable_map.count("grammar-penalty") != 0u)
        params.grammar_penalty = variable_map["grammar-penalty"].as<float>();

    // clang-format on
}

auto transcribe(whisper_context_t* ctx,
                const whisper_params& params,
                const std::vector<float> pcmf32,
                const std::string& grammar_rule,
                float& log_prob_min,
                float& log_prob_sum,
                size_t& n_tokens,
                int64_t& time_ms) -> std::string
{
    const auto start_time = std::chrono::high_resolution_clock::now();

    log_prob_min = 0.0f;
    log_prob_sum = 0.0f;
    n_tokens = 0;
    time_ms = 0;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.translate = false;
    wparams.no_context = true;
    wparams.no_timestamps = true;
    wparams.single_segment = true;
    wparams.max_tokens = params.max_tokens;
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.audio_ctx = params.audio_ctx;
    wparams.speed_up = false;
    wparams.temperature = 0.4f;
    wparams.temperature_inc = 1.0f;
    wparams.greedy.best_of = 5;
    wparams.beam_search.beam_size = 5;
    wparams.initial_prompt = params.context.data();

    const auto& grammar_parsed = params.grammar_parsed;
    auto grammar_rules = grammar_parsed.c_rules();

    if (!params.grammar_parsed.rules.empty() && !grammar_rule.empty())
    {
        if (grammar_parsed.symbol_ids.find(grammar_rule) == grammar_parsed.symbol_ids.end())
        {
            std::fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, grammar_rule.c_str());
        }
        else
        {
            wparams.grammar_rules = grammar_rules.data();
            wparams.n_grammar_rules = grammar_rules.size();
            wparams.i_start_rule = grammar_parsed.symbol_ids.at(grammar_rule);
            wparams.grammar_penalty = params.grammar_penalty;
        }
    }

    if (whisper_full(ctx, wparams, pcmf32.data(), (int) pcmf32.size()) != 0)
        return "";

    std::string res;

    const auto n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i)
    {
        const auto text = whisper_full_get_segment_text(ctx, i);

        res += text;

        const auto n = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n; ++j)
        {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            if (token.plog > 0.0f)
                exit(0);

            log_prob_min = std::min(log_prob_min, token.plog);
            log_prob_sum += token.plog;
            ++n_tokens;
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    return res;
}

auto get_words(const std::string& str) -> std::vector<std::string>
{
    std::vector<std::string> words;
    boost::split(words, str, boost::is_any_of("\n\t "));

    return words;
}

auto extract_prompt_and_command(const std::string& text, size_t prompt_length) -> std::pair<std::string, std::string>
{
    const auto words = get_words(text);
    std::string prompt;
    std::string command;

    for (size_t i = 0; i < words.size(); ++i)
    {
        if (i < prompt_length)
            prompt += words[i] + " ";
        else
            command += words[i] + " ";
    }

    return std::make_pair(prompt, command);
}

auto read_commands(const std::string& file_name) -> std::vector<std::string>
{
    std::vector<std::string> commands;
    std::ifstream file{file_name};

    if (!file)
        return commands;

    std::string line;
    while (std::getline(file, line))
    {
        line = trim(line);
        if (line.empty())
            continue;

        std::transform(std::begin(line), std::end(line), std::begin(line), tolower);

        commands.push_back(line);
    }

    return commands;
}

auto tokenize_commands(const std::string& file_name, whisper_context_t* ctx) -> std::vector<std::vector<whisper_token>>
{
    const auto allowed_commands = read_commands(file_name);

    std::vector<std::vector<whisper_token>> allowed_tokens;
    for (const auto& cmd : allowed_commands)
    {
        std::array<whisper_token, 1024> tokens{};

        allowed_tokens.emplace_back();

        for (size_t l = 0; l < cmd.size(); ++l)
        {
            const auto str = std::string(" ") + cmd.substr(0, l + 1);
            const int n = whisper_tokenize(ctx, str.c_str(), tokens.data(), 1024);

            if (n == 1)
                allowed_tokens.back().push_back(tokens[0]);
        }
    }

    return allowed_tokens;
}

void process_commands(whisper_context_t* ctx, audio_async& audio, const whisper_params& params)
{
    bool is_running = true;
    bool ask_prompt = true;

    float log_prob_min = 0.0f;
    float log_prob_sum = 0.0f;
    size_t n_tokens = 0;

    std::vector<float> pcmf32;

    const auto allowed_tokens = tokenize_commands(params.commands, ctx);

    const auto prompt_length = get_words(params.prompt).size();

    while (is_running)
    {
        is_running = sdl_poll_events();

        std::this_thread::sleep_for(100ms);

        audio.get(2000, pcmf32);

        if (vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false))
        {
            int64_t time_ms = 0;

            audio.get(params.command_ms, pcmf32);

            const auto text = trim(transcribe(ctx, params, pcmf32, "", log_prob_min, log_prob_sum, n_tokens, time_ms));

            const auto [prompt, command] = extract_prompt_and_command(text, prompt_length);

            const auto match = similarity(prompt, params.prompt);

            if ((match > 0.7f) && (command.size() > 0))
                std::cout << "Command: " << command << std::endl;

            std::cout << std::endl;

            audio.clear();
        }
    }
}
