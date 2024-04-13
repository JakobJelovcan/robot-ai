#include <whisper/common-sdl.h>
#include <whisper/common.h>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <regex>
#include <robot-ai/whisper_wrapper.hpp>
#include <strstream>

namespace whs
{
    using namespace std::chrono_literals;

    whisper::whisper(const whisper_config& config)
        : config{config}
        , audio{audio_buffer_size}
        , ctx{nullptr}
    {
        if (!std::filesystem::exists(config.model))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, config.model));

        if (!audio.init(config.capture_id, WHISPER_SAMPLE_RATE))
            throw std::runtime_error(std::format("{}: error: audio initialization failed", __func__));

        whisper_context_params_t ctx_params = whisper_context_default_params();
        ctx_params.use_gpu = config.use_gpu;

        ctx = whisper_init_from_file_with_params(config.model.c_str(), ctx_params);
        if (!ctx)
            throw std::runtime_error(std::format("{}: error: failed to load context", __func__));

        commands = load_commands(config.commands);
        initial_context = load_context(config.context);
    }

    whisper::~whisper()
    {
        whisper_free(ctx);
    }

    auto whisper::transcribe(const std::vector<float>& pcmf32) -> std::string
    {
        auto params = whisper_get_full_params();
        if (whisper_full(ctx, params, pcmf32.data(), (int) pcmf32.size()) != 0)
            return "";

        std::string result;
        const auto n_segments = whisper_full_n_segments(ctx);
        for (auto i = 0; i < n_segments; ++i)
            result += whisper_full_get_segment_text(ctx, i);

        return ::trim(result);
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

        prompt = ::trim(std::regex_replace(prompt, std::regex("[^a-zA-Z ]"), ""));
        command = ::trim(command);

        return std::make_pair(prompt, command);
    }

    void whisper::start_whisper()
    {
        std::scoped_lock lock{sync};
        if (whisper_thread.joinable())
            return;

        whisper_thread = std::jthread([&](std::stop_token token) { whisper_loop(token); });
    }

    void whisper::stop_whisper()
    {
        std::scoped_lock lock{sync};
        if (!whisper_thread.joinable())
            return;

        whisper_thread.request_stop();
        whisper_thread.join();
    }

    void whisper::whisper_loop(std::stop_token token)
    {
        audio.resume();
        std::this_thread::sleep_for(1000ms);
        audio.clear();

        while (true)
        {
            if (token.stop_requested())
                return;

            std::this_thread::sleep_for(100ms);
            audio.get(2000, pcmf32);

            if (vad_simple(pcmf32, WHISPER_SAMPLE_RATE, 1000, config.vad_threshold, config.freq_threshold, false))
            {
                std::cout << "[whisper_wrapper] Detected sound. Processing" << std::endl;
                audio.get(config.command_ms, pcmf32);

                const auto transcription = transcribe(pcmf32);
                const auto [prompt, command] = split_prompt_and_command(transcription);

                const auto sim = similarity(prompt, config.prompt);
                std::cout << std::format("[whisper_wrapper] (Match: {:.0f}%) Transcription: '{}'", sim * 100.0f, transcription)
                          << std::endl;

                if (sim > similarity_treshold && on_command)
                    on_command(command);

                audio.clear();
            }
        }
    }

    auto whisper::load_commands(const std::string& file_name) -> std::vector<std::string>
    {
        if (!std::filesystem::exists(file_name))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, file_name));

        std::vector<std::string> commands;
        std::ifstream ifs{file_name};
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

    auto whisper::load_context(const std::string& file_name) -> std::string
    {
        if (!std::filesystem::exists(file_name))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, file_name));

        std::ifstream ifs{file_name};
        std::stringstream ss;
        ss << ifs.rdbuf();
        return ss.str();
    }

    auto whisper::whisper_get_full_params() const -> whisper_full_params
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
        params.audio_ctx = config.audio_ctx;
        params.speed_up = false;
        params.temperature = 0.4f;
        params.temperature_inc = 1.0f;
        params.greedy.best_of = 5;
        params.beam_search.beam_size = 5;
        params.initial_prompt = initial_context.c_str();

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
            .audio_ctx = -1,
            .vad_threshold = 0.6f,
            .freq_threshold = 100.0,
            .use_gpu = true,
            .model = "./models/ggml-small.en.bin",
            .prompt = "hey darko",
            .commands = "./commands/commands.txt",
            .context = "./contexts/whisper-darko.txt",
        };
    }

    auto get_words(const std::string& str) -> std::vector<std::string>
    {
        std::vector<std::string> words;
        boost::split(words, str, boost::is_any_of("\n\t "));
        return words;
    }
};