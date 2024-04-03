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
        static constexpr int n_prev{64};
        static const constexpr std::string antiprompt {"You:"};

        const llama_config config;
        int n_keep;
        int n_past;

        llama_model* model;
        llama_context* ctx;
        llama_batch batch;
        std::vector<llama_token> embd_inp;

        auto tokenize_prompt(std::string prompt) -> std::vector<llama_token>;
        auto predict_token() -> llama_token;
    };

    auto llama_get_default_config() -> llama_config;
}