#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <regex>
#include <robot-ai/llama_wrapper.hpp>
#include <span>
#include <strstream>

namespace lma
{
    llama::llama(const llama_config& config)
        : config{config}
        , initialized{false}
        , ctx{nullptr}
        , model{nullptr}
        , batch{0}
    {
        if (!std::filesystem::exists(config.model))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, config.model));

        // Init model
        llama_backend_init();
        auto m_params = llama_model_default_params();
        m_params.n_gpu_layers = config.n_gpu_layers;
        model = llama_load_model_from_file(config.model.c_str(), m_params);

        if (!model)
            throw std::runtime_error(std::format("{}: error: failed to load the model", __func__));

        // Init context
        auto c_params = llama_context_default_params();
        c_params.seed = 1;
        c_params.n_ctx = config.n_ctx;
        c_params.n_threads = config.n_threads;
        c_params.n_threads_batch = config.n_threads;
        ctx = llama_new_context_with_model(model, c_params);

        if (!ctx)
            throw std::runtime_error(std::format("{}: error: failed to create context", __func__));

        // Load context data
        context_tokens = load_context(config.context);

        // Init batch
        batch = llama_batch_init((int32_t) llama_n_ctx(ctx), 0, 1);
    }

    llama::~llama()
    {
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
    }

    void llama::init_context()
    {
        initialized = true;
        embd_history = context_tokens;

        for (llama_pos i = 0; i < embd_history.size(); ++i)
            llama_batch_add(batch, embd_history[i], i, {0}, (i == embd_history.size() - 1));

        if (llama_decode(ctx, batch) != 0)
            throw std::runtime_error(std::format("{}: error: failed to decoded the batch", __func__));
    }

    void llama::reset_context()
    {
        if (!initialized)
            return;

        initialized = false;
        llama_batch_clear(batch);
        embd_history.clear();
    }

    auto llama::generate_from_prompt(const std::string& prompt) -> std::string
    {
        if (!initialized)
            init_context();

        auto embd = tokenize_prompt(prompt);
        bool done = false;
        std::string result;
        while (true)
        {
            if (embd.size() > 0)
            {
                if (embd_history.size() + (int) embd.size() > llama_n_ctx(ctx))
                {
                    const auto keep = (int32_t) std::min(max_history, embd_history.size());
                    embd.insert(std::begin(embd), std::end(embd_history) - keep, std::end(embd_history));
                    embd_history.erase(std::begin(embd_history), std::end(embd_history) - keep);
                }

                llama_batch_clear(batch);
                for (llama_pos i = 0; i < embd.size(); ++i)
                    llama_batch_add(batch, embd[i], (llama_pos) embd_history.size() + i, {0}, (i == embd.size() - 1));

                if (llama_decode(ctx, batch) != 0)
                    throw std::runtime_error(std::format("{}: error: failed to decode the batch", __func__));
            }

            embd_history.insert(std::end(embd_history), std::begin(embd), std::end(embd));

            embd.clear();

            if (done)
                break;

            const auto new_token_id = predict_next_token();

            done |= (new_token_id == llama_token_eos(model));
            if (!done)
            {
                embd.push_back(new_token_id);
                result += llama_token_to_piece(ctx, new_token_id);
            }
            std::cout << llama_token_to_piece(ctx, new_token_id);

            done |= remove_antiprompt(result);
        }

        return result;
    }

    auto llama::tokenize_prompt(std::string prompt) -> std::vector<llama_token>
    {
        prompt = std::regex_replace(prompt, std::regex("(\\[.*?\\])|(\\(.*?\\))|([^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-])"), "");
        prompt = prompt.substr(0, prompt.find('\n'));
        prompt = std::regex_replace(prompt, std::regex("(^\\s+)|(\\s+$)"), "");

        return llama_tokenize(ctx, std::format(" {}\nDarko:", prompt), false);
    }

    auto llama::predict_next_token() -> llama_token
    {
        const auto vocab_size = llama_n_vocab(model);
        std::span<float> logits(llama_get_logits(ctx), vocab_size);

        std::vector<llama_token_data> candidates;
        candidates.reserve(vocab_size);

        for (llama_token token_id = 0; token_id < vocab_size; ++token_id)
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});

        llama_token_data_array candidates_p{candidates.data(), candidates.size(), false};

        // new line and eos should not be affected by repetition penalties
        const auto nl_logit = logits[llama_token_nl(model)];
        const auto eos_logit = logits[llama_token_eos(model)];

        const auto history_keep = std::min(max_history, embd_history.size());
        const auto history_skip = embd_history.size() - history_keep;
        llama_sample_repetition_penalties(
            ctx, &candidates_p, &embd_history[history_skip], history_keep, config.repetition_penalty, 0.0f, 0.0f);

        logits[llama_token_nl(model)] = nl_logit;
        logits[llama_token_eos(model)] = eos_logit;

        return llama_sample_token_greedy(ctx, &candidates_p);
    }

    auto llama::remove_antiprompt(std::string& str) -> bool
    {
        for (const auto& antiprompt : antiprompts)
        {
            const auto offset = str.size() - std::min(str.size(), antiprompt.size());
            if (auto pos = str.find(antiprompt, offset); pos != std::string::npos)
            {
                str = str.substr(0, pos);
                return true;
            }
        }
        return false;
    }

    auto llama::load_context(const std::string& file_name) -> std::vector<llama_token>
    {
        if (!std::filesystem::exists(file_name))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, config.context));

        std::ifstream ifs{file_name};
        std::stringstream ss;
        ss << " " << ifs.rdbuf();

        return llama_tokenize(model, ss.str(), true);
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
        return {
            .n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency()),
            .n_ctx = 2048,
            .n_gpu_layers = 99,
            .repetition_penalty = 1.1764f,
            .use_gpu = true,
            .model = "./models/phi-2.Q4_0.gguf",
            .context = "./contexts/llama-darko.txt",
        };
    }
}