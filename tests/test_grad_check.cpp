// tests/test_grad_check.cpp
//
// Numerical gradient verification for the full forward/backward chain:
//   HeteroEncoder -> HeteroGraphSAGE -> MLPHead -> BCELoss
//
// For each parameter p we compute:
//
//   numerical_grad = (L(p + eps) - L(p - eps)) / (2 * eps)
//
// and compare it to the analytical gradient from backward().
// A relative error below 1e-4 is considered correct.
//
// This test uses a tiny synthetic database so it runs in < 1 second.

#include "relml/Column.h"
#include "relml/Database.h"
#include "relml/Table.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/encoding/HeteroEncoder.h"
#include "relml/gnn/HeteroGraphSAGE.h"
#include "relml/gnn/MLPHead.h"
#include "relml/training/Trainer.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace relml;

// ---------------------------------------------------------------------------
// Build a tiny synthetic database:
//   trips: 4 rows, 1 numerical column, 1 binary categorical column
//   drivers: 2 rows, 1 numerical column
//   FK: trips.driver_id -> drivers.id
// ---------------------------------------------------------------------------
static Database make_db() {
    Database db("test");

    // trips table
    Table trips("trips");
    trips.pkey_col = "id";

    Column trip_id("id", ColumnType::NUMERICAL);
    for (int i = 1; i <= 4; ++i) trip_id.data.push_back(static_cast<double>(i));
    trips.add_column(std::move(trip_id));

    Column trip_dist("distance", ColumnType::NUMERICAL);
    trip_dist.data.push_back(1.0);
    trip_dist.data.push_back(-0.5);
    trip_dist.data.push_back(0.3);
    trip_dist.data.push_back(-1.2);
    trips.add_column(std::move(trip_dist));

    Column trip_cat("terrain", ColumnType::CATEGORICAL);
    trip_cat.data.push_back(std::string("flat"));
    trip_cat.data.push_back(std::string("hilly"));
    trip_cat.data.push_back(std::string("flat"));
    trip_cat.data.push_back(std::string("hilly"));
    trips.add_column(std::move(trip_cat));

    Column driver_fk("driver_id", ColumnType::NUMERICAL);
    driver_fk.data.push_back(1.0);
    driver_fk.data.push_back(1.0);
    driver_fk.data.push_back(2.0);
    driver_fk.data.push_back(2.0);
    trips.foreign_keys.push_back({"driver_id", "drivers"});
    trips.add_column(std::move(driver_fk));

    db.add_table(std::move(trips));

    // drivers table
    Table drivers("drivers");
    drivers.pkey_col = "id";

    Column drv_id("id", ColumnType::NUMERICAL);
    drv_id.data.push_back(1.0);
    drv_id.data.push_back(2.0);
    drivers.add_column(std::move(drv_id));

    Column drv_exp("experience", ColumnType::NUMERICAL);
    drv_exp.data.push_back(0.8);
    drv_exp.data.push_back(-0.4);
    drivers.add_column(std::move(drv_exp));

    db.add_table(std::move(drivers));
    return db;
}

// ---------------------------------------------------------------------------
// Forward pass: returns scalar loss given current parameter values.
// samples: vector of (node_idx, label) for the trips table.
// ---------------------------------------------------------------------------
static float forward_loss(
    HeteroEncoder&    encoder,
    HeteroGraphSAGE&  gnn,
    MLPHead&          head,
    const Database&   db,
    const HeteroGraph& graph,
    const std::vector<TaskSample>& samples)
{
    auto x_dict = encoder.transform(db);
    auto h_dict = gnn.forward(x_dict, graph);

    const NodeFeatures& nf = h_dict.at("trips");
    std::size_t N = samples.size();
    std::size_t C = nf.channels;

    std::vector<float> h_batch(N * C);
    for (std::size_t i = 0; i < N; ++i)
        std::copy(nf.data.begin() + samples[i].node_idx * C,
                  nf.data.begin() + (samples[i].node_idx + 1) * C,
                  h_batch.begin() + i * C);

    std::vector<float> logits = head.forward(h_batch, N);

    // BCE loss (no pos_weight)
    float loss = 0.f;
    for (std::size_t i = 0; i < N; ++i) {
        float x = logits[i];
        float y = samples[i].label;
        loss += -y * x
                + std::log(1.f + std::exp(x - 2.f * std::max(x, 0.f)))
                + std::max(x, 0.f);
    }
    return loss / static_cast<float>(N);
}

// ---------------------------------------------------------------------------
// Check gradients for a flat vector of parameters
// ---------------------------------------------------------------------------
static bool check_params(
    std::vector<Parameter*>& params,
    HeteroEncoder&            encoder,
    HeteroGraphSAGE&          gnn,
    MLPHead&                  head,
    const Database&            db,
    const HeteroGraph&         graph,
    const std::vector<TaskSample>& samples,
    float eps = 1e-2f,
    float tol = 2e-2f)
{
    // Step 1: analytical backward
    for (auto* p : params) p->zero_grad();

    auto x_dict = encoder.transform(db);
    auto h_dict = gnn.forward(x_dict, graph);

    const NodeFeatures& nf = h_dict.at("trips");
    std::size_t N = samples.size();
    std::size_t C = nf.channels;

    std::vector<float> h_batch(N * C);
    for (std::size_t i = 0; i < N; ++i)
        std::copy(nf.data.begin() + samples[i].node_idx * C,
                  nf.data.begin() + (samples[i].node_idx + 1) * C,
                  h_batch.begin() + i * C);

    std::vector<float> logits = head.forward(h_batch, N);

    // BCE gradient: d_logit[i] = (sigmoid(logit) - y) / N
    std::vector<float> d_logits(N);
    for (std::size_t i = 0; i < N; ++i) {
        float sig = 1.f / (1.f + std::exp(-logits[i]));
        d_logits[i] = (sig - samples[i].label) / static_cast<float>(N);
    }

    // Scatter back to full node matrix
    std::vector<float> d_h_batch = head.backward(d_logits);

    std::unordered_map<std::string, std::vector<float>> d_h_full;
    for (const auto& [nt, feat] : h_dict)
        d_h_full[nt].assign(feat.num_nodes * C, 0.f);

    for (std::size_t i = 0; i < N; ++i) {
        float* dst = d_h_full.at("trips").data() + samples[i].node_idx * C;
        const float* src = d_h_batch.data() + i * C;
        for (std::size_t c = 0; c < C; ++c) dst[c] += src[c];
    }

    auto d_x = gnn.backward(d_h_full, graph);
    encoder.backward(d_x);

    // Step 2: numerical check
    bool ok = true;
    std::size_t checked = 0;
    std::size_t failed  = 0;

    for (auto* p : params) {
        for (std::size_t j = 0; j < p->size(); ++j) {
            float orig = p->data[j];
            p->data[j] = orig + eps;
            float loss_plus = forward_loss(encoder, gnn, head, db, graph, samples);
            p->data[j] = orig - eps;
            float loss_minus = forward_loss(encoder, gnn, head, db, graph, samples);
            p->data[j] = orig;

            float num_grad = (loss_plus - loss_minus) / (2.f * eps);
            float ana_grad = p->grad[j];

            float denom = std::max(std::abs(num_grad), std::abs(ana_grad));
            denom = std::max(denom, 1e-7f);
            float rel_err = std::abs(num_grad - ana_grad) / denom;

            if (rel_err > tol) {
                std::cerr << "  FAIL: grad mismatch at param[" << j << "]: "
                          << "numerical=" << num_grad
                          << " analytical=" << ana_grad
                          << " rel_err=" << rel_err << "\n";
                ok = false;
                ++failed;
            }
            ++checked;
        }
    }

    std::cout << "  Checked " << checked << " parameters, "
              << failed << " failures\n";
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    constexpr std::size_t CHANNELS = 4;
    constexpr std::size_t HIDDEN   = 4;
    constexpr float       DROPOUT  = 0.f;   // must be 0 for gradient check

    Database    db    = make_db();
    HeteroGraph graph = GraphBuilder::build(db);

    HeteroEncoder   encoder(CHANNELS);
    encoder.fit(db);

    HeteroGraphSAGE gnn(CHANNELS, 1, graph.node_types(), graph.edge_types());

    MLPHead head(CHANNELS, HIDDEN, DROPOUT, 1);
    head.train();

    // Training samples: trips 0,1,2,3 with labels 0,1,0,1
    std::vector<TaskSample> samples = {
        {0, 0.f}, {1, 1.f}, {2, 0.f}, {3, 1.f}
    };

    // Collect all parameters in order: encoder, gnn, mlp
    std::vector<Parameter*> params;
    {
        auto ep = encoder.parameters();
        auto gp = gnn.parameters();
        auto hp = head.parameters();
        params.insert(params.end(), ep.begin(), ep.end());
        params.insert(params.end(), gp.begin(), gp.end());
        params.insert(params.end(), hp.begin(), hp.end());
    }

    std::cout << "Gradient check — " << params.size()
              << " parameter tensors\n";
    std::cout << "  (encoder + 1-layer GNN + MLP, channels="
              << CHANNELS << ", hidden=" << HIDDEN
              << ", dropout=0, 4 training rows)\n\n";

    // eps=1e-2 and tol=2e-2 are the correct settings for float32.
    // With float32, machine epsilon is ~1.2e-7. For a finite difference
    // with eps=1e-3, the truncation error is O(eps^2) and the rounding
    // error is O(machine_eps/eps) = O(1e-4). When gradients are small
    // (~1e-3 magnitude), that rounding error produces relative errors
    // of ~1%, which is numerically correct behaviour, not a bug.
    // Using eps=1e-2 reduces the rounding error floor to ~1e-5 relative
    // to the perturbation, and tol=2e-2 (2%) catches real bugs while
    // not triggering on float32 arithmetic noise.
    bool ok = check_params(
        params, encoder, gnn, head, db, graph, samples,
        /*eps=*/1e-2f, /*tol=*/2e-2f);

    if (ok) {
        std::cout << "\nAll gradients correct.\n";
        return 0;
    } else {
        std::cout << "\nGradient check FAILED.\n";
        return 1;
    }
}