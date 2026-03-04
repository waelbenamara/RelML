#include "relml/agent/Agent.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace relml {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// libcurl write callback — appends received data into a std::string.
// ---------------------------------------------------------------------------
static std::size_t curl_write_cb(
    char* ptr, std::size_t /*size*/, std::size_t nmemb, void* userdata)
{
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(ptr, nmemb);
    return nmemb;
}

// ---------------------------------------------------------------------------
// .env loader — read KEY=value from a file (e.g. .env in project root).
// ---------------------------------------------------------------------------
static std::string getenv_from_dotenv(const std::string& key) {
    const char* path_env = std::getenv("RELML_DOTENV");
    std::string path = path_env && path_env[0] ? path_env : ".env";
    std::ifstream f(path);
    if (!f.is_open()) return {};

    std::string line;
    while (std::getline(f, line)) {
        // trim leading space
        std::size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty() || line[0] == '#') continue;

        std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        // trim trailing space from key
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
        if (k != key) continue;
        // trim value and optional quotes
        std::size_t v_start = v.find_first_not_of(" \t");
        if (v_start != std::string::npos) v = v.substr(v_start);
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r')) v.pop_back();
        if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
        if (v.size() >= 2 && v.front() == '\'' && v.back() == '\'') v = v.substr(1, v.size() - 2);
        return v;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

Agent::Agent(AgentConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.system_prompt.empty())
        cfg_.system_prompt = getenv_from_dotenv("SYSTEM_PROMPT");
    std::string env_model = getenv_from_dotenv("ANTHROPIC_MODEL");
    if (!env_model.empty())
        cfg_.model = env_model;
}

void Agent::reset() {
    history_.clear();
}

std::string Agent::resolve_api_key() const {
    if (!cfg_.api_key.empty()) return cfg_.api_key;

    const char* env = std::getenv("ANTHROPIC_API_KEY");
    if (env && *env) return env;

    std::string from_env_file = getenv_from_dotenv("ANTHROPIC_API_KEY");
    if (!from_env_file.empty()) return from_env_file;

    throw std::runtime_error(
        "Agent: no API key provided. Set AgentConfig::api_key, "
        "ANTHROPIC_API_KEY in the environment, or ANTHROPIC_API_KEY=... in a .env file.");
}

std::string Agent::build_request_body(const std::string& user_message) const {
    json messages = json::array();

    // Replay existing history
    for (const auto& msg : history_) {
        messages.push_back({{"role", msg.role}, {"content", msg.content}});
    }

    // Append the new user turn
    messages.push_back({{"role", "user"}, {"content", user_message}});

    json body = {
        {"model",      cfg_.model},
        {"max_tokens", cfg_.max_tokens},
        {"messages",   messages}
    };

    if (!cfg_.system_prompt.empty())
        body["system"] = cfg_.system_prompt;

    return body.dump();
}

AgentResponse Agent::parse_response(const std::string& json_body) const {
    json resp;
    try {
        resp = json::parse(json_body);
    } catch (const json::exception& e) {
        throw std::runtime_error(
            std::string("Agent: failed to parse API response JSON: ") + e.what()
            + "\nRaw body: " + json_body);
    }

    // Surface API-level errors clearly
    if (resp.contains("error")) {
        std::string msg = resp["error"].value("message", "unknown error");
        throw std::runtime_error("Agent: API error — " + msg);
    }

    AgentResponse out;

    // Extract text from the first content block
    if (resp.contains("content") && resp["content"].is_array()
        && !resp["content"].empty())
    {
        const auto& block = resp["content"][0];
        if (block.contains("text"))
            out.text = block["text"].get<std::string>();
    }

    // Token usage (best-effort)
    if (resp.contains("usage")) {
        out.input_tokens  = resp["usage"].value("input_tokens",  0);
        out.output_tokens = resp["usage"].value("output_tokens", 0);
    }

    return out;
}

AgentResponse Agent::send(const std::string& user_message) {
    const std::string api_key  = resolve_api_key();
    const std::string endpoint = "https://api.anthropic.com/v1/messages";
    const std::string body     = build_request_body(user_message);

    // ---------- initialise curl ----------
    CURL* curl = curl_easy_init();
    if (!curl)
        throw std::runtime_error("Agent: curl_easy_init() failed");

    std::string response_buf;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("x-api-key: "         + api_key).c_str());
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");
    headers = curl_slist_append(headers, "content-type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL,            endpoint.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,     body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,  static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &response_buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        60L);   // seconds

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK)
        throw std::runtime_error(
            std::string("Agent: curl request failed: ") + curl_easy_strerror(res));

    // ---------- parse and update history ----------
    AgentResponse reply = parse_response(response_buf);

    history_.push_back({"user",      user_message});
    history_.push_back({"assistant", reply.text});

    return reply;
}

} // namespace relml