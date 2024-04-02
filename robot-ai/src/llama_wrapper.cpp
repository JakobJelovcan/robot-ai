#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <robot-ai/llama_wrapper.hpp>

namespace lma
{
    llama::llama(const llama_config& config)
    {
        if (!std::filesystem::exists(config.model))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, config.model));

        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

        auto m_params = llama_model_default_params();
        m_params.n_gpu_layers = config.n_gpu_layers;

        model = llama_load_model_from_file(config.model.c_str(), m_params);

        if (!model)
            throw std::runtime_error(std::format("{}: error: failed to load the model", __func__));

        auto c_params = llama_context_default_params();
        c_params.seed = 1;
        c_params.n_ctx = 2048;
        c_params.n_threads = config.n_threads;
        c_params.n_threads_batch = config.n_threads;

        ctx = llama_new_context_with_model(model, c_params);

        if (!ctx)
            throw std::runtime_error(std::format("{}: error: failed to create context", __func__));
    }

    llama::~llama()
    {
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
    }

    auto llama::generate_from_prompt(const std::string& prompt) -> std::string
    {
        auto tokens = llama_tokenize(ctx, prompt, true);

        auto batch = llama_batch_init(512, 0, 1);

        for (size_t i = 0; i < tokens.size(); ++i)
            llama_batch_add(batch, tokens[i], (int) i, {0}, false);

        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0)
            throw std::runtime_error(std::format("{}: error: failed to decode the batch", __func__));

        std::string result;

        for (int i = batch.n_tokens; i < 32; ++i)
        {
            const auto n_vocab = llama_n_vocab(model);
            const auto logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; ++token_id)
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});

            llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            if (new_token_id == llama_token_eos(model) || i == 31)
                break;

            result += llama_token_to_piece(ctx, new_token_id);

            llama_batch_clear(batch);

            llama_batch_add(batch, new_token_id, i, {0}, true);

            if (llama_decode(ctx, batch) != 0)
                throw std::runtime_error(std::format("{}: error: batch decoding failed", __func__));
        }

        return result;
    }

    auto llama::build_llama(const llama_config& config) -> llama_ptr
    {
        try
        {
            return std::make_unique<llama>(config);
        }
        catch (const std::exception& e)
        {
            std::cerr << std::format("Failed to build llama {}", __func__, e.what()) << std::endl;
            return nullptr;
        }
    }

    auto llama_get_default_config() -> llama_config
    {
        return {.n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency()),
                .n_gpu_layers = 99,
                .model = "/home/jakob/git/openDAQ-ai/models/phi-2.Q4_0.gguf"};
    }
}