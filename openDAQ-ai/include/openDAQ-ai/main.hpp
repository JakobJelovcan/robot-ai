#pragma once

#include <openDAQ-ai/common-sdl.h>
#include <openDAQ-ai/grammar-parser.h>
#include <whisper.h>
#include <thread>
#include <vector>

struct whisper_params
{
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t prompt_ms = 5000;
    int32_t command_ms = 8000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx = 0;

    float vad_thold = 0.6f;
    float freq_thold = 100.0f;

    float grammar_penalty = 100.0f;

    grammar_parser::parse_state grammar_parsed;

    bool use_gpu = true;

    std::string language = "en";
    std::string model = "models/ggml-base.en.bin";
    std::string commands;
    std::string prompt = "Hey Darko";
    std::string context;
    std::string grammar;
};

typedef struct whisper_context_params whisper_context_params_t;
typedef struct whisper_context whisper_context_t;

void parse_args(int argc, char* argv[], whisper_params& params);
auto transcribe(whisper_context_t* ctx,
                const whisper_params& params,
                const std::vector<float> pcmf32,
                const std::string& grammar_rule,
                float& log_prob_min,
                float& log_prob_sum,
                size_t& n_tokens,
                int64_t& time_ms) -> std::string;
auto get_words(const std::string& str) -> std::vector<std::string>;
auto extract_prompt_and_command(const std::string& text, size_t prompt_length) -> std::pair<std::string, std::string>;
auto read_commands(const std::string& file_name) -> std::vector<std::string>;
auto tokenize_commands(const std::string& file_name, whisper_context_t* ctx) -> std::vector<std::vector<whisper_token>>;
void process_commands(whisper_context_t* ctx, audio_async& audio, const whisper_params& params);
