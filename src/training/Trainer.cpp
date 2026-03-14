#include "relml/training/Trainer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace relml {

// ---------------------------------------------------------------------------
// BCELoss
// ---------------------------------------------------------------------------

std::pair<float, std::vector<float>> BCELoss::forward_backward(
    const std::vector<float>& logits,
    const std::vector<float>& targets) const
{
    std::size_t N = logits.size();
    std::vector<float> d(N);
    float loss = 0.f;

#pragma omp parallel for reduction(+:loss) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        float x   = logits[i];
        float y   = targets[i];
        float pw  = (y > 0.5f) ? pos_weight : 1.f;
        float sig = 1.f / (1.f + std::exp(-x));
        loss   += pw * (-y * x
                        + std::log(1.f + std::exp(x - 2.f * std::max(x, 0.f)))
                        + std::max(x, 0.f));
        d[i]    = pw * (sig - y) / static_cast<float>(N);
    }
    return {loss / static_cast<float>(N), d};
}

// ---------------------------------------------------------------------------
// MSELoss
// ---------------------------------------------------------------------------

std::pair<float, std::vector<float>> MSELoss::forward_backward(
    const std::vector<float>& preds,
    const std::vector<float>& targets) const
{
    std::size_t N = preds.size();
    std::vector<float> d(N);
    float loss = 0.f;

#pragma omp parallel for reduction(+:loss) schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        float e = preds[i] - targets[i];
        loss   += e * e;
        d[i]    = 2.f * e / static_cast<float>(N);
    }
    return {loss / static_cast<float>(N), d};
}

// ---------------------------------------------------------------------------
// CrossEntropyLoss
// ---------------------------------------------------------------------------

// Weighted cross-entropy forward + backward.
//
// Standard CE:    loss_i = -log(p[cls_i])
// Weighted CE:    loss_i = -w[cls_i] * log(p[cls_i])
//
// Gradient w.r.t. logit k for sample i:
//   Standard:  (softmax_k - 1[k==cls]) / N
//   Weighted:  w[cls_i] * (softmax_k - 1[k==cls]) / N
//
// With inverse-frequency weights the minority class gets proportionally larger
// gradient signal, preventing the model from ignoring rare outcomes entirely.
// class_weights is populated automatically by Trainer::fit() when empty.
std::pair<float, std::vector<float>> CrossEntropyLoss::forward_backward(
    const std::vector<float>& logits,
    const std::vector<float>& targets,
    std::size_t K) const
{
    std::size_t N = targets.size();
    std::vector<float> d(N * K);
    float loss = 0.f;

    bool use_weights = (class_weights.size() == K);

    for (std::size_t i = 0; i < N; ++i) {
        const float* row = logits.data() + i * K;
        float*       dr  = d.data()      + i * K;

        float max_logit = *std::max_element(row, row + K);
        float sum_exp   = 0.f;
        for (std::size_t k = 0; k < K; ++k) sum_exp += std::exp(row[k] - max_logit);
        float log_sum_exp = std::log(sum_exp) + max_logit;

        std::size_t cls = static_cast<std::size_t>(targets[i]);
        float w = use_weights ? class_weights[cls] : 1.f;

        loss += w * (log_sum_exp - row[cls]);

        for (std::size_t k = 0; k < K; ++k) {
            float softmax_k = std::exp(row[k] - max_logit) / sum_exp;
            dr[k] = w * (softmax_k - (k == cls ? 1.f : 0.f)) / static_cast<float>(N);
        }
    }
    return {loss / static_cast<float>(N), d};
}

// ---------------------------------------------------------------------------
// CSV-based task loader (legacy, used by test_training.cpp)
// ---------------------------------------------------------------------------

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string f;
    while (std::getline(ss, f, ',')) fields.push_back(f);
    return fields;
}

TaskSplit load_task(const std::string& csv_path, const Database& db,
                    const std::string& target_node)
{
    std::ifstream file(csv_path);
    if (!file.is_open())
        throw std::runtime_error("load_task: cannot open " + csv_path);

    const Table&  t      = db.get_table(target_node);
    const Column& pk_col = t.get_column(*t.pkey_col);

    std::unordered_map<int64_t, int64_t> pk_to_row;
    for (std::size_t i = 0; i < pk_col.size(); ++i)
        if (!pk_col.is_null(i))
            pk_to_row[static_cast<int64_t>(pk_col.get_numerical(i))] = i;

    std::string line;
    std::getline(file, line);
    auto header = split_csv_line(line);

    int col_id = -1, col_label = -1, col_split = -1;
    const std::string& pk_name = *t.pkey_col;
    for (int i = 0; i < (int)header.size(); ++i) {
        if (header[i] == pk_name)  col_id    = i;
        if (header[i] == "label")  col_label = i;
        if (header[i] == "split")  col_split = i;
    }
    if (col_id < 0 || col_label < 0 || col_split < 0)
        throw std::runtime_error("load_task: missing required columns");

    TaskSplit split;
    std::size_t skipped = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto fields = split_csv_line(line);
        if ((int)fields.size() <= std::max({col_id, col_label, col_split})) continue;
        int64_t pk_val = std::stoll(fields[col_id]);
        float   label  = std::stof(fields[col_label]);
        auto    sp     = fields[col_split];
        auto    it     = pk_to_row.find(pk_val);
        if (it == pk_to_row.end()) { ++skipped; continue; }
        TaskSample s{it->second, label};
        if      (sp == "train") split.train.push_back(s);
        else if (sp == "val")   split.val.push_back(s);
        else if (sp == "test")  split.test.push_back(s);
    }
    if (skipped)
        std::cerr << "  Warning: " << skipped << " rows had unresolved PKs\n";
    return split;
}

// ---------------------------------------------------------------------------
// Trainer construction
// ---------------------------------------------------------------------------

Trainer::Trainer(const TrainConfig& cfg, const Database& db, const HeteroGraph& graph)
    : encoder(cfg.channels),
      gnn(cfg.channels, cfg.gnn_layers, graph.node_types(), graph.edge_types()),
      head(cfg.channels, cfg.hidden, cfg.dropout, cfg.task.output_dim()),
      bce_loss_{cfg.pos_weight},
      optimizer(cfg.lr),
      cfg_(cfg)
{
    encoder.fit(db);

    auto ep = encoder.parameters();
    auto gp = gnn.parameters();
    auto hp = head.parameters();
    num_encoder_params_ = ep.size();
    all_params_.insert(all_params_.end(), ep.begin(), ep.end());
    all_params_.insert(all_params_.end(), gp.begin(), gp.end());
    all_params_.insert(all_params_.end(), hp.begin(), hp.end());

    std::size_t total = 0;
    for (auto* p : all_params_) total += p->size();
    std::cout << "  Trainable tensors: " << all_params_.size()
              << "  total floats: "      << total << "\n";
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::vector<float> Trainer::gather(
    const NodeFeatures& nf, const std::vector<TaskSample>& samples) const
{
    std::size_t N = samples.size();
    std::vector<float> out(N * cfg_.channels);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i) {
        int64_t row = samples[i].node_idx;
        std::copy(nf.data.begin() + row * cfg_.channels,
                  nf.data.begin() + (row + 1) * cfg_.channels,
                  out.begin() + i * cfg_.channels);
    }
    return out;
}

std::vector<float> Trainer::labels(const std::vector<TaskSample>& samples) const {
    std::vector<float> y(samples.size());
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < samples.size(); ++i) y[i] = samples[i].label;
    return y;
}

std::pair<float, std::vector<float>> Trainer::compute_loss(
    const std::vector<float>& logits,
    const std::vector<float>& targets) const
{
    switch (cfg_.task.task_type) {
        case TaskSpec::TaskType::BinaryClassification:
            return bce_loss_.forward_backward(logits, targets);
        case TaskSpec::TaskType::Regression:
            return mse_loss_.forward_backward(logits, targets);
        case TaskSpec::TaskType::MulticlassClassification:
            return ce_loss_.forward_backward(logits, targets, cfg_.task.output_dim());
    }
    return bce_loss_.forward_backward(logits, targets);
}

float Trainer::forward_pass_batch(
    const std::unordered_map<std::string, NodeFeatures>& h_dict,
    const std::vector<TaskSample>& batch,
    std::unordered_map<std::string, std::vector<float>>& d_h_full_accum,
    bool train)
{
    std::size_t N = batch.size();
    std::vector<float> h_batch = gather(h_dict.at(cfg_.task.target_table), batch);
    std::vector<float> y       = labels(batch);
    std::vector<float> logits  = head.forward(h_batch, N);

    auto [loss, d_logits] = compute_loss(logits, y);
    if (!train) return loss;

    std::vector<float> d_h_batch = head.backward(d_logits);
    float* dst_base = d_h_full_accum.at(cfg_.task.target_table).data();
    for (std::size_t i = 0; i < N; ++i) {
        int64_t      row = batch[i].node_idx;
        float*       dst = dst_base + row * cfg_.channels;
        const float* src = d_h_batch.data() + i * cfg_.channels;
        for (std::size_t c = 0; c < cfg_.channels; ++c) dst[c] += src[c];
    }
    return loss;
}

// ---------------------------------------------------------------------------
// apply_output_transform
// ---------------------------------------------------------------------------

float Trainer::apply_output_transform(float raw) const {
    switch (cfg_.task.task_type) {
        case TaskSpec::TaskType::BinaryClassification:
            return 1.f / (1.f + std::exp(-raw));

        case TaskSpec::TaskType::Regression:
            if (cfg_.task.label_transform.kind == LabelTransform::Kind::Normalize)
                return raw * cfg_.task.label_transform.norm_std
                           + cfg_.task.label_transform.norm_mean;
            return raw;

        case TaskSpec::TaskType::MulticlassClassification:
            return raw;  // caller handles argmax over the full logit vector
    }
    return raw;
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

EvalMetrics Trainer::evaluate(
    const std::vector<TaskSample>& samples,
    const Database& db,
    const HeteroGraph& graph)
{
    head.eval();

    auto x_dict = encoder.transform(db);
    auto h_dict = gnn.forward(x_dict, graph);
    std::vector<float> h_batch = gather(h_dict.at(cfg_.task.target_table), samples);
    std::vector<float> logits  = head.forward(h_batch, samples.size());
    std::vector<float> y       = labels(samples);

    auto [loss, _ignore] = compute_loss(logits, y);

    EvalMetrics m;
    m.loss = loss;

    switch (cfg_.task.task_type) {
        case TaskSpec::TaskType::BinaryClassification: {
            std::vector<float> probs(logits.size());
            for (std::size_t i = 0; i < logits.size(); ++i)
                probs[i] = 1.f / (1.f + std::exp(-logits[i]));
            m = compute_metrics(probs, y);
            m.loss = loss;
            break;
        }
        case TaskSpec::TaskType::Regression:
            m = compute_regression_metrics(logits, y);
            m.loss = loss;
            break;
        case TaskSpec::TaskType::MulticlassClassification:
            m = compute_multiclass_metrics(logits, y, cfg_.task.output_dim());
            m.loss = loss;
            break;
    }

    head.train();
    return m;
}

// ---------------------------------------------------------------------------
// predict_all
// ---------------------------------------------------------------------------

std::vector<float> Trainer::predict_all(const Database& db, const HeteroGraph& graph) {
    head.eval();

    auto x_dict = encoder.transform(db);
    auto h_dict = gnn.forward(x_dict, graph);

    const NodeFeatures& nf = h_dict.at(cfg_.task.target_table);
    std::size_t N = nf.num_nodes;

    std::vector<TaskSample> all_samples(N);
    for (std::size_t i = 0; i < N; ++i)
        all_samples[i] = {static_cast<int64_t>(i), 0.f};

    std::vector<float> h_batch = gather(nf, all_samples);
    std::vector<float> logits  = head.forward(h_batch, N);

    std::vector<float> preds(N);
    if (cfg_.task.task_type == TaskSpec::TaskType::MulticlassClassification) {
        std::size_t K = cfg_.task.output_dim();
        for (std::size_t i = 0; i < N; ++i) {
            const float* row = logits.data() + i * K;
            preds[i] = static_cast<float>(
                std::distance(row, std::max_element(row, row + K)));
        }
    } else {
        for (std::size_t i = 0; i < N; ++i)
            preds[i] = apply_output_transform(logits[i]);
    }

    head.train();
    return preds;
}

// ---------------------------------------------------------------------------
// synthesize_prediction
// ---------------------------------------------------------------------------

float Trainer::synthesize_prediction(
    const std::unordered_map<std::string, std::string>& entity_refs,
    const Database&    db,
    const HeteroGraph& graph)
{
    head.eval();

    auto x_dict = encoder.transform(db);
    auto h_dict = gnn.forward(x_dict, graph);

    const Table& target = db.get_table(cfg_.task.target_table);

    std::vector<float> pooled(cfg_.channels, 0.f);
    int found = 0;

    for (const auto& [fk_col, entity_id_str] : entity_refs) {
        std::string ref_table;
        for (const auto& fk : target.foreign_keys)
            if (fk.column == fk_col) { ref_table = fk.target_table; break; }

        if (ref_table.empty()) {
            std::cerr << "  Warning: FK column '" << fk_col
                      << "' not found in '" << cfg_.task.target_table << "', skipping\n";
            continue;
        }

        const Table& rtable = db.get_table(ref_table);
        if (!rtable.pkey_col) continue;
        const Column& pk = rtable.get_column(*rtable.pkey_col);

        float   target_id = std::stof(entity_id_str);
        int64_t row_idx   = -1;
        for (std::size_t i = 0; i < pk.size(); ++i) {
            if (!pk.is_null(i) && static_cast<float>(pk.get_numerical(i)) == target_id) {
                row_idx = static_cast<int64_t>(i);
                break;
            }
        }

        if (row_idx < 0) {
            std::cerr << "  Warning: entity " << fk_col << "=" << entity_id_str
                      << " not found in '" << ref_table << "'\n";
            continue;
        }

        const NodeFeatures& nf  = h_dict.at(ref_table);
        const float*        emb = nf.data.data() + row_idx * cfg_.channels;
        for (std::size_t c = 0; c < cfg_.channels; ++c)
            pooled[c] += emb[c];
        ++found;
    }

    if (found == 0)
        throw std::runtime_error(
            "synthesize_prediction: none of the entity references could be resolved");

    for (float& v : pooled) v /= static_cast<float>(found);

    std::vector<float> logits = head.forward(pooled, 1);

    float pred;
    if (cfg_.task.task_type == TaskSpec::TaskType::MulticlassClassification) {
        std::size_t K = cfg_.task.output_dim();
        pred = static_cast<float>(
            std::distance(logits.begin(),
                          std::max_element(logits.begin(), logits.begin() + K)));
    } else {
        pred = apply_output_transform(logits[0]);
    }

    head.train();
    return pred;
}

// ---------------------------------------------------------------------------
// save_weights / load_weights
// ---------------------------------------------------------------------------

void Trainer::save_weights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_weights: cannot open " + path);

    std::size_t n = all_params_.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));

    float mu  = cfg_.task.label_transform.norm_mean;
    float sig = cfg_.task.label_transform.norm_std;
    f.write(reinterpret_cast<const char*>(&mu),  sizeof(mu));
    f.write(reinterpret_cast<const char*>(&sig), sizeof(sig));

    for (const auto* p : all_params_) {
        std::size_t sz = p->size();
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        f.write(reinterpret_cast<const char*>(p->data.data()), sz * sizeof(float));
    }
}

void Trainer::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_weights: cannot open " + path);

    std::size_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (n != all_params_.size())
        throw std::runtime_error("load_weights: parameter count mismatch ("
            + std::to_string(n) + " vs " + std::to_string(all_params_.size()) + ")");

    float mu = 0.f, sig = 1.f;
    f.read(reinterpret_cast<char*>(&mu),  sizeof(mu));
    f.read(reinterpret_cast<char*>(&sig), sizeof(sig));
    cfg_.task.label_transform.norm_mean = mu;
    cfg_.task.label_transform.norm_std  = sig;

    for (auto* p : all_params_) {
        std::size_t sz = 0;
        f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        if (sz != p->size())
            throw std::runtime_error("load_weights: parameter size mismatch");
        f.read(reinterpret_cast<char*>(p->data.data()), sz * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------

EvalMetrics Trainer::fit(
    const TaskSplit& task, const Database& db, const HeteroGraph& graph)
{
    if (cfg_.task.task_type == TaskSpec::TaskType::BinaryClassification
        && cfg_.pos_weight == 1.f)
    {
        float n_pos = 0.f, n_neg = 0.f;
        for (const auto& s : task.train) (s.label > 0.5f ? n_pos : n_neg) += 1.f;
        if (n_pos > 0.f && n_neg > 0.f) {
            bce_loss_.pos_weight = n_neg / n_pos;
            std::cout << "  pos_weight auto: " << std::fixed
                      << std::setprecision(3) << bce_loss_.pos_weight << "\n";
        }
    }

    // Auto-compute inverse-frequency class weights for multiclass tasks.
    // w[k] = (N / K) / count[k], normalized so the average weight is 1.
    // Example for football (43% home / 23% draw / 33% away):
    //   w[0]=0.78  w[1]=1.45  w[2]=1.01
    // The draw class gets ~2x the gradient signal of the home-win class,
    // which stops the model from predicting draws only 8% of the time.
    if (cfg_.task.task_type == TaskSpec::TaskType::MulticlassClassification
        && ce_loss_.class_weights.empty())
    {
        std::size_t K = cfg_.task.output_dim();
        std::vector<float> counts(K, 0.f);
        for (const auto& s : task.train)
            counts[static_cast<std::size_t>(s.label)] += 1.f;

        float N_train = static_cast<float>(task.train.size());
        ce_loss_.class_weights.resize(K);
        std::cout << "  class_weights auto (home/draw/away): ";
        for (std::size_t k = 0; k < K; ++k) {
            ce_loss_.class_weights[k] = (counts[k] > 0.f)
                ? (N_train / static_cast<float>(K)) / counts[k]
                : 1.f;
            std::cout << std::fixed << std::setprecision(3)
                      << ce_loss_.class_weights[k]
                      << (k + 1 < K ? "  " : "\n");
        }
    }

    EvalMetrics      best_val;
    std::vector<float> best_snapshot;

    auto snapshot = [&]() {
        best_snapshot.clear();
        for (auto* p : all_params_)
            best_snapshot.insert(best_snapshot.end(), p->data.begin(), p->data.end());
    };
    auto restore = [&]() {
        std::size_t offset = 0;
        for (auto* p : all_params_) {
            std::copy(best_snapshot.begin() + offset,
                      best_snapshot.begin() + offset + p->size(),
                      p->data.begin());
            offset += p->size();
        }
    };

    snapshot();
    head.train();

    std::size_t n_train  = task.train.size();
    std::size_t batch_sz = (cfg_.batch_size > 0 && cfg_.batch_size < n_train)
                           ? cfg_.batch_size : n_train;
    if (batch_sz < n_train)
        std::cout << "  Mini-batch training: batch_size=" << batch_sz << "\n";

    bool is_reg        = (cfg_.task.task_type == TaskSpec::TaskType::Regression);
    bool is_multiclass = (cfg_.task.task_type == TaskSpec::TaskType::MulticlassClassification);

    std::cout << "\n" << std::string(88, '-') << "\n";
    if (is_reg)
        std::cout << " Epoch |  Train Loss  |  Val RMSE  |  Val MAE   |  Val R²  | Time (s)\n";
    else if (is_multiclass)
        std::cout << " Epoch |  Train Loss  |   Val Acc  |   Val F1  |           | Time (s)\n";
    else
        std::cout << " Epoch |  Train Loss  |   Val AP  |  Val AUC  |  Val Acc | Time (s)\n";
    std::cout << std::string(88, '-') << "\n";

    for (std::size_t epoch = 1; epoch <= cfg_.epochs; ++epoch) {
        auto t0 = std::chrono::steady_clock::now();

        for (auto* p : all_params_) p->zero_grad();

        auto x_dict = encoder.transform(db);
        auto h_dict = gnn.forward(x_dict, graph);

        std::unordered_map<std::string, std::vector<float>> d_h_full;
        for (const auto& [nt, nf] : h_dict)
            d_h_full[nt].assign(nf.num_nodes * cfg_.channels, 0.f);

        float train_loss_sum = 0.f;
        for (std::size_t start = 0; start < n_train; start += batch_sz) {
            std::size_t end = std::min(start + batch_sz, n_train);
            std::vector<TaskSample> batch(task.train.begin() + start,
                                          task.train.begin() + end);
            float loss = forward_pass_batch(h_dict, batch, d_h_full, true);
            train_loss_sum += loss * static_cast<float>(batch.size());
        }
        float train_loss = n_train > 0
            ? train_loss_sum / static_cast<float>(n_train) : 0.f;

        auto d_x = gnn.backward(d_h_full, graph);
        encoder.backward(d_x);
        optimizer.step(all_params_);

        EvalMetrics val_m = evaluate(task.val, db, graph);

        auto   t1 = std::chrono::steady_clock::now();
        double s  = std::chrono::duration<double>(t1 - t0).count();

        std::cout << std::setw(5) << epoch << "  | "
                  << std::fixed << std::setprecision(6) << train_loss << "   | ";
        if (is_reg)
            std::cout << std::setprecision(4) << val_m.rmse << "      | "
                      << val_m.mae << "     | "
                      << val_m.r2  << " | ";
        else if (is_multiclass)
            std::cout << std::setprecision(4) << val_m.accuracy << "    | "
                      << val_m.f1       << "    |           | ";
        else
            std::cout << std::setprecision(4) << val_m.average_precision << "    | "
                      << val_m.roc_auc  << "    | "
                      << val_m.accuracy << " | ";
        std::cout << std::setprecision(2) << s << "\n";

        bool improved;
        if (is_reg)
            improved = (best_val.rmse == 0.f || val_m.rmse < best_val.rmse);
        else if (is_multiclass)
            improved = (val_m.accuracy > best_val.accuracy);
        else
            improved = (val_m.average_precision > best_val.average_precision);

        if (improved) {
            best_val = val_m;
            snapshot();
        }
    }

    restore();
    std::cout << std::string(88, '-') << "\n\nBest val:\n";
    best_val.print("  ");

    EvalMetrics test_m = evaluate(task.test, db, graph);
    std::cout << "Test:\n";
    test_m.print("  ");

    return best_val;
}

} // namespace relml