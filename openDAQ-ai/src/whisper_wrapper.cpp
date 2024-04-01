#include <whisper-common/common-sdl.h>
#include <whisper-common/common.h>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <openDAQ-ai/whisper_wrapper.hpp>

namespace whs
{
    using namespace std::chrono_literals;

    whisper::whisper(const whisper_config& config)
        : config{config}
        , audio{audio_buffer_size}
    {
        if (!std::filesystem::exists(config.model))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, config.model));

        if (!audio.init(config.capture_id, WHISPER_SAMPLE_RATE))
            throw std::runtime_error(std::format("{}: error: audio initialization failed", __func__));

        whisper_context_params_t ctx_params = whisper_context_default_params();
        ctx_params.use_gpu = config.use_gpu;

        ctx = whisper_init_from_file_with_params(config.model.c_str(), ctx_params);

        commands = read_commands(config.commands);
        prompt_tokens = tokenize_prompt(config.prompt);
        command_tokens = tokenize_commands(commands);
    }

    whisper::~whisper()
    {
        audio.pause();
        whisper_free(ctx);
    }

    auto whisper::transcribe(const std::vector<float>& pcmf32) -> std::string
    {
        auto params = whisper_get_full_prompt_params();
        if (whisper_full(ctx, params, pcmf32.data(), (int) pcmf32.size()) != 0)
            return "";

        std::string result;
        const auto n_segments = whisper_full_n_segments(ctx);
        for (auto i = 0; i < n_segments; ++i)
            result += whisper_full_get_segment_text(ctx, i);

        return ::trim(result);
    }

    auto whisper::transcribe_commands(const std::vector<float>& pcmf32) -> std::optional<std::string>
    {
        auto params = whisper_get_full_command_params();
        if (whisper_full(ctx, params, pcmf32.data(), (int) pcmf32.size()) != 0)
            return std::nullopt;

        const std::span<float> logits(whisper_get_logits(ctx), whisper_n_vocab(ctx));
        const auto [prob, index] = find_best_command(logits);

        return (prob > 0.7f) ? std::make_optional(commands[index]) : std::nullopt;
    }

    auto whisper::split_prompt_and_command(const std::string& str) -> std::pair<std::string, std::string>
    {
        const auto prompt_length = get_words(config.prompt).size();

        std::string prompt, command;
        const auto words = get_words(str);

        for (size_t i = 0; i < words.size(); ++i)
        {
            if (i < prompt_length)
                prompt += words[i] + " ";
            else
                command += words[i] + " ";
        }

        return std::make_pair(::trim(prompt), ::trim(command));
    }

    auto whisper::find_best_command(const std::span<float>& logits) -> std::pair<float, uint32_t>
    {
        std::vector<float> probs(logits.size(), 0.0f);

        float max = *std::max_element(std::cbegin(logits), std::cend(logits));

        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i)
        {
            probs[i] = std::exp(logits[i] - max);
            sum += probs[i];
        }

        for (auto& prob : probs)
            prob /= sum;

        std::vector<std::pair<float, int>> probs_id;

        double psum = 0.0;
        for (size_t i = 0; i < commands.size(); ++i)
        {
            probs_id.emplace_back(probs[command_tokens[i][0]], i);
            probs_id.back().first += (float) std::accumulate(std::cbegin(command_tokens[i]), std::cend(command_tokens[i]), 0);
            probs_id.back().first /= (float) command_tokens.size();
            psum += probs_id.back().first;
        }

        for (auto& pair : probs_id)
            pair.first /= (float) psum;

        return *std::max_element(probs_id.cbegin(), probs_id.cend(), [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    void whisper::whisper_loop()
    {
        bool is_running = true;

        std::vector<float> pcmf32;

        audio.resume();
        std::this_thread::sleep_for(1000ms);
        audio.clear();

        while (is_running)
        {
            is_running = sdl_poll_events();

            std::this_thread::sleep_for(100ms);

            audio.get(2000, pcmf32);

            if (vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, config.vad_threshold, config.freq_threshold, false))
            {
                std::cout << "Detected sound" << std::endl;
                audio.get(config.command_ms, pcmf32);

                const auto transcription = transcribe(pcmf32);
                const auto [prompt, command] = split_prompt_and_command(transcription);

                std::cout << std::format("[whisper_wrapper] Transcription: '{}'", transcription) << std::endl;

                const auto sim = similarity(prompt, config.prompt);
                if (sim > 0.7f && on_command)
                    on_command(command);

                audio.clear();
            }
        }
    }

    auto whisper::read_commands(const std::string& file_name) -> std::vector<std::string>
    {
        std::vector<std::string> commands;

        std::ifstream ifs{file_name};
        if (!ifs)
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, file_name));

        std::string line;

        while (std::getline(ifs, line))
        {
            line = ::trim(line);
            if (line.empty())
                continue;

            std::transform(std::begin(line), std::end(line), std::begin(line), ::tolower);
            commands.push_back(line);
        }

        return commands;
    }

    auto whisper::tokenize_prompt(const std::string& prompt) -> std::vector<whisper_token>
    {
        std::vector<whisper_token> tokens(max_token_count);
        const auto n = whisper_tokenize(ctx, prompt.c_str(), tokens.data(), max_token_count);

        if (n < 0)
            throw std::runtime_error(std::format("{}: error: failed to tokenize the prompt '{}'", __func__, prompt));

        tokens.resize(n);
        return tokens;
    }

    auto whisper::tokenize_commands(const std::vector<std::string>& commands) -> std::vector<std::vector<whisper_token>>
    {
        std::vector<std::vector<whisper_token>> all_tokens;
        for (const auto& command : commands)
        {
            std::array<whisper_token, max_token_count> tokens{0};
            all_tokens.emplace_back();

            for (auto l = 0; l < command.size(); ++l)
            {
                const auto str = std::format(" {}", command.substr(0, l + 1));
                const auto n = whisper_tokenize(ctx, str.c_str(), tokens.data(), max_token_count);

                if (n < 0)
                    throw std::runtime_error(std::format("{}: error: failed to tokenize command '{}'", __func__, str));

                if (n == 1)
                    all_tokens.back().push_back(tokens[0]);
            }
        }

        return all_tokens;
    }

    auto whisper::whisper_get_full_command_params() const -> whisper_full_params
    {
        auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.print_progress = false;
        params.print_special = false;
        params.print_realtime = false;
        params.print_timestamps = false;
        params.translate = false;
        params.no_context = true;
        params.single_segment = true;
        params.max_tokens = 1;
        params.language = "en";
        params.n_threads = config.n_threads;
        params.audio_ctx = -1;
        params.speed_up = false;

        params.prompt_tokens = prompt_tokens.data();
        params.prompt_n_tokens = prompt_tokens.size();

        return params;
    }

    auto whisper::whisper_get_full_prompt_params() const -> whisper_full_params
    {
        auto params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

        params.print_progress = false;
        params.print_special = false;
        params.print_realtime = false;
        params.print_timestamps = false;
        params.translate = false;
        params.no_context = true;
        params.no_timestamps = true;
        params.single_segment = true;
        params.max_tokens = config.max_tokens;
        params.language = "en";
        params.n_threads = config.n_threads;
        params.audio_ctx = -1;
        params.speed_up = false;
        params.temperature = 0.4f;
        params.temperature_inc = 1.0f;
        params.greedy.best_of = 5;
        params.beam_search.beam_size = 5;
        params.initial_prompt = config.context.c_str();

        return params;
    }

    auto whisper::build_whisper(const whisper_config& config) -> whisper_ptr
    {
        try
        {
            return std::make_unique<whisper>(config);
        }
        catch (const std::exception& e)
        {
            std::cerr << std::format("Failed to build whisper: {}", e.what()) << std::endl;
            return nullptr;
        }
    }

    auto whisper_get_default_config() -> whisper_config
    {
        return {
            .n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency()),
            .command_ms = 8000,
            .prompt_ms = 5000,
            .capture_id = -1,
            .max_tokens = 32,
            .vad_threshold = 0.6f,
            .freq_threshold = 100.0,
            .use_gpu = true,
            .model = "/home/jakob/git/openDAQ-ai/models/ggml-large-v2-q5_0.bin",
            .prompt = "hey darko",
            .commands = "/home/jakob/git/openDAQ-ai/commands/commands.txt",
            .context = "hello how is it going always use lowercase no punctuation goodbye one two three start stop i you me they hey darko wake wave sleep jump lowercase no ponctuation hello hey darko go to sleep wake up wave by",
        };
    }

    auto get_words(const std::string& str) -> std::vector<std::string>
    {
        std::vector<std::string> words;
        boost::split(words, str, boost::is_any_of("\n\t "));
        return words;
    }
};