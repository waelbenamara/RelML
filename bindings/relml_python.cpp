#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "relml/Database.h"
#include "relml/Table.h"
#include "relml/Column.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/graph/HeteroGraph.h"
#include "relml/training/TaskSpec.h"
#include "relml/training/Trainer.h"

namespace py = pybind11;
using namespace relml;

// ---------------------------------------------------------------------------
// Helpers to build relml::Table from Python-side data.
// Python passes column data as lists; we infer types from Python types.
// ---------------------------------------------------------------------------

static ColumnType infer_python_type(const py::object& values) {
    // Check the first non-None value to determine type
    for (auto val : values) {
        if (val.is_none()) continue;
        if (py::isinstance<py::float_>(val) || py::isinstance<py::int_>(val))
            return ColumnType::NUMERICAL;
        if (py::isinstance<py::str>(val))
            return ColumnType::CATEGORICAL;
        break;
    }
    return ColumnType::TEXT;
}

// Build a relml::Column from a Python list of values.
// col_type is passed explicitly from the DuckDB schema.
static Column build_column(
    const std::string& name,
    ColumnType         type,
    const py::list&    values)
{
    Column col(name, type);
    col.data.reserve(values.size());

    for (auto val : values) {
        if (val.is_none()) {
            col.data.push_back(std::monostate{});
            continue;
        }
        switch (type) {
            case ColumnType::NUMERICAL:
                col.data.push_back(val.cast<double>());
                break;
            case ColumnType::CATEGORICAL:
            case ColumnType::TEXT:
                col.data.push_back(val.cast<std::string>());
                break;
            case ColumnType::TIMESTAMP:
                // Python side passes unix seconds as int
                col.data.push_back(static_cast<int64_t>(val.cast<long long>()));
                break;
        }
    }
    return col;
}

// ---------------------------------------------------------------------------
// TrainedModel: wraps a Trainer and exposes predict methods to Python.
// Holds the Database and HeteroGraph by value so they outlive the call.
// ---------------------------------------------------------------------------
struct TrainedModel {
    Database    db;
    HeteroGraph graph;
    TaskSpec    spec;
    std::unique_ptr<Trainer> trainer;

    // Score every row in the target table. Returns a flat list of floats.
    std::vector<float> predict_all() {
        return trainer->predict_all(db, graph);
    }

    // Entity synthesis: predict for a specific combination of entity IDs.
    // entity_refs maps fk_column_name -> entity_id_string
    float predict_entity(const std::unordered_map<std::string, std::string>& entity_refs) {
        return trainer->synthesize_prediction(entity_refs, db, graph);
    }
};

// ---------------------------------------------------------------------------
// Main training entry point called from Python.
// Accepts the database as a dict of table dicts, and a task description.
// ---------------------------------------------------------------------------
std::shared_ptr<TrainedModel> train(
    // db_tables: { table_name -> { "columns": [...], "pkey_col": str|None,
    //              "time_col": str|None, "foreign_keys": [(col, target), ...] } }
    const py::dict&  db_tables,
    // task: { "sql_table": str, "target_column": str, "task_type": str,
    //         "label_transform": {...}, "split_strategy": str, "time_col": str|None,
    //         "inference_mode": str, "entity_refs": {...}, "inference_agg": str }
    const py::dict&  task_dict,
    // train config
    std::size_t channels   = 64,
    std::size_t gnn_layers = 2,
    std::size_t hidden     = 64,
    float       dropout    = 0.3f,
    float       lr         = 3e-4f,
    std::size_t epochs     = 30,
    std::size_t batch_size = 0
)
{
    // ---- Build relml::Database from Python dicts -------------------------
    Database db("python_db");

    for (auto item : db_tables) {
        std::string tname = item.first.cast<std::string>();
        py::dict    tdict = item.second.cast<py::dict>();

        Table table(tname);

        if (tdict.contains("pkey_col") && !tdict["pkey_col"].is_none())
            table.pkey_col = tdict["pkey_col"].cast<std::string>();

        if (tdict.contains("time_col") && !tdict["time_col"].is_none())
            table.time_col = tdict["time_col"].cast<std::string>();

        if (tdict.contains("foreign_keys")) {
            for (auto fk : tdict["foreign_keys"].cast<py::list>()) {
                auto pair = fk.cast<py::tuple>();
                table.foreign_keys.push_back({
                    pair[0].cast<std::string>(),
                    pair[1].cast<std::string>()
                });
            }
        }

        py::list col_specs = tdict["columns"].cast<py::list>();
        for (auto cs : col_specs) {
            py::dict cd = cs.cast<py::dict>();
            std::string col_name = cd["name"].cast<std::string>();
            std::string col_type_str = cd["type"].cast<std::string>();
            py::list    values = cd["values"].cast<py::list>();

            ColumnType ct;
            if      (col_type_str == "numerical")   ct = ColumnType::NUMERICAL;
            else if (col_type_str == "categorical")  ct = ColumnType::CATEGORICAL;
            else if (col_type_str == "timestamp")    ct = ColumnType::TIMESTAMP;
            else                                     ct = ColumnType::TEXT;

            table.add_column(build_column(col_name, ct, values));
        }
        db.add_table(std::move(table));
    }

    // ---- Build TaskSpec from task dict -----------------------------------
    TaskSpec spec;
    spec.target_table  = task_dict["target_table"].cast<std::string>();
    spec.target_column = task_dict["target_column"].cast<std::string>();

    std::string tt = task_dict["task_type"].cast<std::string>();
    if      (tt == "binary_classification")     spec.task_type = TaskSpec::TaskType::BinaryClassification;
    else if (tt == "regression")                spec.task_type = TaskSpec::TaskType::Regression;
    else if (tt == "multiclass_classification") spec.task_type = TaskSpec::TaskType::MulticlassClassification;

    // Label transform
    py::dict lt = task_dict["label_transform"].cast<py::dict>();
    std::string lk = lt["kind"].cast<std::string>();
    if (lk == "threshold") {
        spec.label_transform.kind      = LabelTransform::Kind::Threshold;
        spec.label_transform.threshold = lt["threshold"].cast<float>();
        spec.label_transform.inclusive = lt.contains("inclusive")
            ? lt["inclusive"].cast<bool>() : true;
    } else if (lk == "normalize") {
        spec.label_transform.kind = LabelTransform::Kind::Normalize;
    } else if (lk == "buckets") {
        spec.label_transform.kind    = LabelTransform::Kind::Buckets;
        spec.label_transform.buckets = lt["buckets"].cast<std::vector<float>>();
    }

    // Split strategy
    std::string ss = task_dict.contains("split_strategy")
        ? task_dict["split_strategy"].cast<std::string>() : "random";
    if (ss == "temporal") {
        spec.split_strategy = TaskSpec::SplitStrategy::Temporal;
        if (task_dict.contains("time_col") && !task_dict["time_col"].is_none())
            spec.split_time_col = task_dict["time_col"].cast<std::string>();
    }

    // Inference
    std::string im = task_dict.contains("inference_mode")
        ? task_dict["inference_mode"].cast<std::string>() : "row_based";
    if (im == "entity_synthesis") {
        spec.inference_mode = TaskSpec::InferenceMode::EntitySynthesis;
        if (task_dict.contains("entity_refs"))
            spec.entity_refs = task_dict["entity_refs"]
                .cast<std::unordered_map<std::string, std::string>>();
    }

    std::string agg = task_dict.contains("inference_agg")
        ? task_dict["inference_agg"].cast<std::string>() : "none";
    if      (agg == "mean")     spec.inference_agg = TaskSpec::AggType::Mean;
    else if (agg == "fraction") spec.inference_agg = TaskSpec::AggType::Fraction;
    else if (agg == "count")    spec.inference_agg = TaskSpec::AggType::Count;

    // ---- Build graph and train -------------------------------------------
    FKDetector::detect(db);
    HeteroGraph graph = GraphBuilder::build(db);

    TrainConfig cfg;
    cfg.channels   = channels;
    cfg.gnn_layers = gnn_layers;
    cfg.hidden     = hidden;
    cfg.dropout    = dropout;
    cfg.lr         = lr;
    cfg.epochs     = epochs;
    cfg.batch_size = batch_size;
    cfg.task       = spec;

    auto trainer = std::make_unique<Trainer>(cfg, db, graph);
    TaskSplit split = spec.build_split(db);
    trainer->fit(split, db, graph);

    auto model = std::make_shared<TrainedModel>();
    model->db      = std::move(db);
    model->graph   = std::move(graph);
    model->spec    = spec;
    model->trainer = std::move(trainer);
    return model;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_relml_core, m) {
    m.doc() = "RelML C++ core bindings";

    py::class_<TrainedModel, std::shared_ptr<TrainedModel>>(m, "TrainedModel")
        .def("predict_all",    &TrainedModel::predict_all)
        .def("predict_entity", &TrainedModel::predict_entity);

    m.def("train", &train,
        py::arg("db_tables"),
        py::arg("task_dict"),
        py::arg("channels")   = 64,
        py::arg("gnn_layers") = 2,
        py::arg("hidden")     = 64,
        py::arg("dropout")    = 0.3f,
        py::arg("lr")         = 3e-4f,
        py::arg("epochs")     = 30,
        py::arg("batch_size") = 0
    );
}