#pragma once
#include <llama/common.h>
#include <llama/llama.h>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lma
{
    using namespace std::string_view_literals;

    class llama;
    using llama_ptr = std::unique_ptr<llama>;

    struct llama_config
    {
        int32_t n_threads;
        int32_t n_ctx;
        int32_t n_gpu_layers;
        float repetition_penalty;
        bool use_gpu;
        std::string model;
        std::string context;
    };

    class llama
    {
    public:
        llama(const llama_config& config);
        ~llama();

        void init();
        auto generate_from_prompt(const std::string& prompt) -> std::string;

        static auto build_llama(const llama_config& config) -> llama_ptr;

    protected:
    private:
        static constexpr size_t max_history{256};
        static constexpr std::array antiprompts{"[Answer]"sv, "[Question]"sv};

        const llama_config config;
        std::vector<llama_token> embd_context;
        std::vector<llama_token> embd_history;
        std::mutex sync;

        llama_model* model;
        llama_context* ctx;
        llama_batch batch;

        auto tokenize_prompt(std::string prompt) -> std::vector<llama_token>;
        auto load_context(const std::string& file_name) -> std::vector<llama_token>;
        auto remove_antiprompt(std::string& str) -> bool;
        auto predict_next_token() -> llama_token;
    };

    auto llama_get_default_config() -> llama_config;
}