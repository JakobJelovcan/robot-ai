#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <regex>
#include <robot-ai/llama_wrapper.hpp>

namespace lma
{
    llama::llama(const llama_config& config)
        : config{config}
    {
        if (!std::filesystem::exists(config.model))
            throw std::runtime_error(std::format("{}: error: file '{}' does not exist", __func__, config.model));

        llama_backend_init();

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

    void llama::init()
    {
        const auto initial_context = std::format(" {}", config.context);
        batch = llama_batch_init((int32_t) llama_n_ctx(ctx), 0, 1);
        embd_inp = llama_tokenize(ctx, initial_context, true);
        n_keep = (int) embd_inp.size();
        n_past = n_keep;

        for (llama_pos i = 0; i < embd_inp.size(); ++i)
            llama_batch_add(batch, embd_inp[i], i, {0}, (i == embd_inp.size() - 1));

        if (llama_decode(ctx, batch) != 0)
            throw std::runtime_error(std::format("{}: error: failed to decoded the batch", __func__));
    }

    auto llama::generate_from_prompt(const std::string& prompt) -> std::string
    {
        auto embd = tokenize_prompt(prompt);
        bool done = false;
        std::string result;
        while (true)
        {
            if (embd.size() > 0)
            {
                if (n_past + (int) embd.size() > llama_n_ctx(ctx))
                {
                    n_past = n_keep;
                    embd.insert(std::begin(embd), std::end(embd_inp) - n_prev, std::end(embd_inp));
                    embd_inp.erase(std::begin(embd_inp), std::end(embd_inp) - n_prev);
                }

                llama_batch_clear(batch);
                for (llama_pos i = 0; i < embd.size(); ++i)
                    llama_batch_add(batch, embd[i], n_past + i, {0}, (i == embd.size() - 1));

                if (llama_decode(ctx, batch) != 0)
                    throw std::runtime_error(std::format("{}: error: failed to decode the batch", __func__));
            }

            embd_inp.insert(std::end(embd_inp), std::begin(embd), std::end(embd));
            n_past += embd.size();

            embd.clear();

            if (done)
                break;

            const auto new_token_id = predict_token();

            if (new_token_id != llama_token_eos(model))
            {
                embd.push_back(new_token_id);
                result += llama_token_to_piece(ctx, new_token_id);
            }

            if (result.find(antiprompt.c_str(), result.length() - antiprompt.length(), antiprompt.length()) != std::string::npos)
                done = true;
        }

        return result.substr(0, result.find(antiprompt.c_str(), result.length() - antiprompt.length(), antiprompt.length()));
    }

    auto llama::tokenize_prompt(std::string prompt) -> std::vector<llama_token>
    {
        prompt = std::regex_replace(prompt, std::regex("(\\[.*?\\])|(\\(.*?\\))|([^a-zA-Z0-9\\.,\\?!\\s\\:\\'\\-])"), "");
        prompt = prompt.substr(0, prompt.find('\n'));
        prompt = std::regex_replace(prompt, std::regex("(^\\s+)|(\\s+$)"), "");

        return llama_tokenize(ctx, std::format(" {}\nDarko:", prompt), false);
    }

    auto llama::predict_token() -> llama_token
    {
        auto logits = llama_get_logits(ctx);
        auto vocab_size = llama_n_vocab(model);

        logits[llama_token_eos(model)] = 0;

        std::vector<llama_token_data> candidates;
        candidates.reserve(vocab_size);

        for (llama_token token_id = 0; token_id < vocab_size; ++token_id)
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});

        llama_token_data_array candidates_p{candidates.data(), candidates.size(), false};

        const auto nl_logit = logits[llama_token_nl(model)];
        llama_sample_repetition_penalties(ctx, &candidates_p, embd_inp.data() + std::max(0, 256), 256, 1.1764f, 0.0f, 0.0f);
        logits[llama_token_nl(model)] = nl_logit;

        return llama_sample_token_greedy(ctx, &candidates_p);
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
            .model = "/home/jakob/git/openDAQ-ai/models/phi-2.Q4_0.gguf",
            .context =
                "You are a robot called Darko. Your task is to interract with the customers at the event in a helpful, kind, "
                "honest, friendly manner. You will always immediately answer the customers question with precision. You will "
                "never give answers that are too long and you will always answer with correct grammar."
                "The conversation does not contain any annotations like (30 seconds passed...) or (to himself), just what Darko and You"
                "say to each other"
                "Darko responds with short and concise answers."
                "You: Hello, Darko!"
                "Darko: Hello, how may I help you today?"
                "You: What is a cat?"
                "Darko: A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae."
                "You: Name a color"
                "Darko: Blue"};
    }
}