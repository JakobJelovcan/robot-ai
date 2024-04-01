#pragma once

#include <whisper-common/common-sdl.h>
#include <whisper.h>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>

namespace whs
{
    typedef struct whisper_context whisper_context_t;
    typedef struct whisper_context_params whisper_context_params_t;

    class whisper;
    using whisper_ptr = std::unique_ptr<whisper>;

    struct whisper_config
    {
        int32_t n_threads;
        int32_t command_ms;
        int32_t prompt_ms;
        int32_t capture_id;
        int32_t max_tokens;
        float vad_threshold;
        float freq_threshold;
        bool use_gpu;
        std::string model;
        std::string prompt;
        std::string commands;
        std::string context;
    };

    class whisper
    {
    public:
        whisper(const whisper_config& config);
        ~whisper();
        std::function<void(const std::string&)> on_command;
        void whisper_loop();

        static auto build_whisper(const whisper_config& config) -> whisper_ptr;

    protected:
    private:
        static constexpr size_t max_token_count{1024};
        static constexpr size_t audio_buffer_size{30 * 1000};

        const whisper_config config;
        whisper_context* ctx;
        audio_async audio;
        std::vector<std::string> commands;
        std::vector<whisper_token> prompt_tokens;
        std::vector<std::vector<whisper_token>> command_tokens;

        auto transcribe(const std::vector<float>& pcmf32) -> std::string;
        auto transcribe_commands(const std::vector<float>& pcmf32) -> std::optional<std::string>;
        auto find_best_command(const std::span<float>& logits) -> std::pair<float, uint32_t>;
        auto split_prompt_and_command(const std::string& str) -> std::pair<std::string, std::string>;
        auto read_commands(const std::string& file_name) -> std::vector<std::string>;
        auto tokenize_prompt(const std::string& prompt) -> std::vector<whisper_token>;
        auto tokenize_commands(const std::vector<std::string>& commands) -> std::vector<std::vector<whisper_token>>;
        auto whisper_get_full_command_params() const -> whisper_full_params;
        auto whisper_get_full_prompt_params() const -> whisper_full_params;
    };

    auto whisper_get_default_config() -> whisper_config;
    auto get_words(const std::string& str) -> std::vector<std::string>;
}