#pragma once

#include "relml/agent/Agent.h"
#include "relml/Database.h"
#include "relml/training/TaskSpec.h"

#include <string>

namespace relml {

// Parses a natural language query into a TaskSpec by calling the Anthropic API.
// The database schema is serialised and injected into the system prompt so the
// model can reason about which table and column to target.
class TaskParser {
public:
    explicit TaskParser(AgentConfig cfg = {});

    // Parse nl_query against db's schema.
    // Throws std::runtime_error if the agent response cannot be decoded.
    TaskSpec parse(const std::string& nl_query, const Database& db) const;

private:
    mutable Agent agent_;

    // Serialise the database schema into a compact JSON string for the prompt.
    static std::string schema_to_json(const Database& db);

    // Decode the agent's JSON reply into a TaskSpec.
    static TaskSpec decode_json(const std::string& json, const Database& db);
};

} // namespace relml
