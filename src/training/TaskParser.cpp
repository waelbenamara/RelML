#include "relml/training/TaskParser.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace relml {

using json = nlohmann::json;

static const char* SYSTEM_PROMPT = R"(
You are a machine learning task parser for relational databases.
Given a database schema and a natural language query, output ONLY a single JSON
object — no preamble, no markdown, no explanation.

The JSON must have exactly these fields:

{
  "target_table":  "<table name>",
  "target_column": "<column to predict>",
  "task_type":     "binary_classification" | "regression" | "multiclass_classification",
  "label_transform": {
    "kind":      "threshold" | "normalize" | "buckets",
    "threshold": <float>,         // only for "threshold"
    "inclusive": <bool>,          // only for "threshold": true means >=
    "buckets":   [<float>, ...]   // only for "buckets": sorted boundary list
  },
  "inference_mode": "row_based" | "entity_synthesis",
  "inference_filters": [...],     // only used when inference_mode = "row_based"
  "inference_agg":    "none" | "mean" | "fraction" | "count",
  "entity_refs": {},              // only used when inference_mode = "entity_synthesis"
  "split_strategy": "temporal" | "random"
}

STEP 1 - task_type and label_transform
---------------------------------------
- "will X do Y?", "probability that", "is X likely to", "chance of"
    binary_classification + threshold
- "what rating?", "expected value?", "how much?", "predicted cost?", "estimate the"
    regression + normalize  (ALWAYS normalize, never identity)
- "which category?", "what tier?", "classify into"
    multiclass_classification + buckets

STEP 2 - inference_mode  (most important decision)
---------------------------------------
An observation table (ratings, trips, transactions) records links between entities.
The model trains on these existing rows. At inference time:

Use "entity_synthesis" when the query names SPECIFIC IDs from MULTIPLE referenced
entities and asks for a prediction about a combination that may not exist as a row.
  "What would user 5 rate movie 56?"
    entity_refs: {"userId": "5", "movieId": "56"}
  "What is the expected fuel cost for driver 12 with truck 7?"
    entity_refs: {"driverId": "12", "truckId": "7"}
  "Will driver 42 finish in the top 3 in race 99?"
    entity_refs: {"driverId": "42", "raceId": "99"}
  For entity_synthesis: inference_filters=[], inference_agg="none", entity_refs={...}

Use "row_based" when the query asks about PATTERNS over existing records, or filters
on a SINGLE entity to aggregate over its existing rows.
  "What fraction of ratings are likely 4 or above?"
    inference_filters=[], inference_agg="fraction"
  "How does user 42 tend to rate movies?"
    inference_filters=[{userId=42}], inference_agg="mean"
  For row_based: entity_refs={}

Rule: if the query names IDs from two or more different entity tables (users AND movies,
drivers AND races, drivers AND trucks), use entity_synthesis.

STEP 3 - split_strategy
---------------------------------------
"temporal" if target table has time_col in schema, else "random".

STEP 4 - inference_filters (row_based only)
---------------------------------------
Each filter MUST be an object with exactly these three string fields:
  {"column": "<fk_col_in_target_table>", "op": "<operator>", "value": "<string>"}

Examples:
  "user 5"   -> {"column": "userId",  "op": "=", "value": "5"}
  "movie 99" -> {"column": "movieId", "op": "=", "value": "99"}
  "above 4"  -> {"column": "rating",  "op": ">", "value": "4"}

WRONG format (never do this): {"userId": "5"}
CORRECT format:               {"column": "userId", "op": "=", "value": "5"}

At most one entity equality filter per row_based query.
inference_agg: "none"=per-row, "mean"=average, "fraction"=positive rate, "count"=count positive
)";

static AgentConfig make_parser_config(AgentConfig cfg) {
    cfg.system_prompt = SYSTEM_PROMPT;
    return cfg;
}

TaskParser::TaskParser(AgentConfig cfg)
    : agent_(make_parser_config(std::move(cfg)))
{}

std::string TaskParser::schema_to_json(const Database& db) {
    json tables = json::array();

    for (const auto& [tname, table] : db.tables) {
        json cols = json::array();
        for (const auto& col : table.columns) {
            std::string type_str;
            switch (col.type) {
                case ColumnType::NUMERICAL:   type_str = "numerical";   break;
                case ColumnType::CATEGORICAL: type_str = "categorical"; break;
                case ColumnType::TIMESTAMP:   type_str = "timestamp";   break;
                case ColumnType::TEXT:        type_str = "text";        break;
            }
            json c = {{"name", col.name}, {"type", type_str}};
            if (table.pkey_col && *table.pkey_col == col.name) c["pk"] = true;
            for (const auto& fk : table.foreign_keys)
                if (fk.column == col.name) c["fk_to"] = fk.target_table;
            cols.push_back(c);
        }

        json t = {{"table", tname}, {"columns", cols}};
        if (table.pkey_col) t["pkey_col"] = *table.pkey_col;
        if (table.time_col)  t["time_col"]  = *table.time_col;

        json fks = json::array();
        for (const auto& fk : table.foreign_keys)
            fks.push_back({{"column", fk.column}, {"references", fk.target_table}});
        if (!fks.empty()) t["foreign_keys"] = fks;

        tables.push_back(t);
    }

    return tables.dump(2);
}

TaskSpec TaskParser::decode_json(const std::string& raw, const Database& db) {
    std::string stripped = raw;
    auto fence_start = stripped.find("```");
    if (fence_start != std::string::npos) {
        auto content_start = stripped.find('\n', fence_start);
        auto fence_end     = stripped.rfind("```");
        if (content_start != std::string::npos && fence_end > content_start)
            stripped = stripped.substr(content_start + 1, fence_end - content_start - 1);
    }

    json j;
    try {
        j = json::parse(stripped);
    } catch (const json::exception& e) {
        throw std::runtime_error(
            std::string("TaskParser: invalid JSON from agent: ") + e.what()
            + "\nRaw reply:\n" + raw);
    }

    TaskSpec spec;
    spec.target_table  = j.at("target_table").get<std::string>();
    spec.target_column = j.at("target_column").get<std::string>();

    if (!db.has_table(spec.target_table))
        throw std::runtime_error("TaskParser: unknown table '" + spec.target_table + "'");
    if (!db.get_table(spec.target_table).has_column(spec.target_column))
        throw std::runtime_error("TaskParser: unknown column '" + spec.target_column
                                 + "' in '" + spec.target_table + "'");

    std::string tt = j.at("task_type").get<std::string>();
    if      (tt == "binary_classification")     spec.task_type = TaskSpec::TaskType::BinaryClassification;
    else if (tt == "regression")                spec.task_type = TaskSpec::TaskType::Regression;
    else if (tt == "multiclass_classification") spec.task_type = TaskSpec::TaskType::MulticlassClassification;
    else throw std::runtime_error("TaskParser: unknown task_type '" + tt + "'");

    const auto& lt = j.at("label_transform");
    std::string lk = lt.at("kind").get<std::string>();
    if (lk == "threshold") {
        spec.label_transform.kind      = LabelTransform::Kind::Threshold;
        spec.label_transform.threshold = lt.at("threshold").get<float>();
        spec.label_transform.inclusive = lt.value("inclusive", true);
    } else if (lk == "identity" || lk == "normalize") {
        spec.label_transform.kind = (spec.task_type == TaskSpec::TaskType::Regression)
            ? LabelTransform::Kind::Normalize
            : LabelTransform::Kind::Identity;
    } else if (lk == "buckets") {
        spec.label_transform.kind    = LabelTransform::Kind::Buckets;
        spec.label_transform.buckets = lt.at("buckets").get<std::vector<float>>();
    } else {
        throw std::runtime_error("TaskParser: unknown label_transform.kind '" + lk + "'");
    }

    std::string im = j.value("inference_mode", "row_based");
    if (im == "entity_synthesis") {
        spec.inference_mode = TaskSpec::InferenceMode::EntitySynthesis;
        if (j.contains("entity_refs") && j["entity_refs"].is_object())
            for (auto& [k, v] : j["entity_refs"].items())
                spec.entity_refs[k] = v.get<std::string>();
    } else {
        spec.inference_mode = TaskSpec::InferenceMode::RowBased;
        if (j.contains("inference_filters")) {
            for (const auto& f : j["inference_filters"]) {
                if (!f.contains("column") || !f.contains("op") || !f.contains("value")) {
                    throw std::runtime_error(
                        "TaskParser: malformed inference_filter — expected "
                        "{\"column\":..., \"op\":..., \"value\":...}, got:\n"
                        + f.dump(2)
                        + "\n\nFull LLM reply:\n" + raw);
                }
                InferenceFilter inf;
                inf.column = f.at("column").get<std::string>();
                inf.op     = f.at("op").get<std::string>();
                inf.value  = f.at("value").get<std::string>();
                spec.inference_filters.push_back(std::move(inf));
            }
        }
        std::string agg = j.value("inference_agg", "none");
        if      (agg == "mean")     spec.inference_agg = TaskSpec::AggType::Mean;
        else if (agg == "fraction") spec.inference_agg = TaskSpec::AggType::Fraction;
        else if (agg == "count")    spec.inference_agg = TaskSpec::AggType::Count;
        else                        spec.inference_agg = TaskSpec::AggType::None;
    }

    std::string ss = j.value("split_strategy", "random");
    if (ss == "temporal") {
        spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
        const Table& tbl = db.get_table(spec.target_table);
        if (tbl.time_col) spec.split_time_col = *tbl.time_col;
    } else {
        spec.split_strategy = TaskSpec::SplitStrategy::Random;
    }

    return spec;
}

TaskSpec TaskParser::parse(const std::string& nl_query, const Database& db) const {
    std::string schema = schema_to_json(db);

    std::ostringstream prompt;
    prompt << "Database schema:\n" << schema
           << "\n\nQuery: " << nl_query;

    agent_.reset();
    AgentResponse resp = agent_.send(prompt.str());
    return decode_json(resp.text, db);
}

} // namespace relml