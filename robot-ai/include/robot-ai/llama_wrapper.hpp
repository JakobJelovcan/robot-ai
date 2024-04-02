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
        int32_t n_gpu_layers;
        std::string model;
    };

    class llama
    {
    public:
        llama(const llama_config& config);
        ~llama();

        auto generate_from_prompt(const std::string& prompt) -> std::string;

        static auto build_llama(const llama_config& config) -> llama_ptr;

    protected:
    private:
        llama_model* model;
        llama_context* ctx;
    };

    auto llama_get_default_config() -> llama_config;
}