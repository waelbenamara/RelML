#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace relml {

// A single message in the conversation history.
struct AgentMessage {
    std::string role;    // "user" or "assistant"
    std::string content;
};

// Configuration for the Agent.
struct AgentConfig {
    std::string api_key;                                     // else ANTHROPIC_API_KEY env, else .env file
    std::string model      = "claude-sonnet-4-20250514";     // else ANTHROPIC_MODEL in .env
    int         max_tokens = 1024;
    std::string system_prompt;                               // optional; else SYSTEM_PROMPT in .env
};

// AgentResponse holds the text reply and raw token usage info.
struct AgentResponse {
    std::string text;
    int         input_tokens  = 0;
    int         output_tokens = 0;
};

// Agent wraps the Anthropic /v1/messages endpoint.
// It maintains conversation history across calls to send().
//
// Usage:
//   AgentConfig cfg;
//   cfg.system_prompt = "You are a helpful data scientist.";
//   Agent agent(cfg);
//
//   AgentResponse r = agent.send("Hello!");
//   std::cout << r.text << "\n";
//
// The conversation history is preserved so subsequent calls to send()
// build on previous turns. Call reset() to start fresh.
class Agent {
public:
    explicit Agent(AgentConfig cfg);

    // Send a user message and return the model's reply.
    // The user message and assistant reply are both appended to history_.
    AgentResponse send(const std::string& user_message);

    // Clear conversation history (does not touch config).
    void reset();

    // Read-only access to the current conversation history.
    const std::vector<AgentMessage>& history() const { return history_; }

private:
    AgentConfig              cfg_;
    std::vector<AgentMessage> history_;

    std::string resolve_api_key() const;
    std::string build_request_body(const std::string& user_message) const;
    AgentResponse parse_response(const std::string& json_body) const;
};

} // namespace relml