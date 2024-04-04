#pragma once
#include <llama/common.h>
#include <llama/llama.h>
#include <memory>
#include <string>
#include <vector>

namespace lma
{
    class llama;
    using llama_ptr = std::unique_ptr<llama>;

    struct llama_config
    {
        int32_t n_threads;
        int32_t n_ctx;
        int32_t n_gpu_layers;
        bool use_gpu;
        std::string model;
        std::string context;
    };

    class llama
    {
    public:
        llama(const llama_config& config);
        ~llama();

        void init_context();
        void reset_context();
        auto generate_from_prompt(const std::string& prompt) -> std::string;

        static auto build_llama(const llama_config& config) -> llama_ptr;

    protected:
    private:
        static constexpr size_t max_history{256};
        const llama_config config;
        bool initialized;
        std::vector<llama_token> context_tokens;
        std::vector<llama_token> embd_history;

        llama_model* model;
        llama_context* ctx;
        llama_batch batch;

        auto tokenize_prompt(std::string prompt) -> std::vector<llama_token>;
        auto load_context(const std::string& file_name) -> std::vector<llama_token>;
        auto predict_token() -> llama_token;
    };

    auto llama_get_default_config() -> llama_config;
}