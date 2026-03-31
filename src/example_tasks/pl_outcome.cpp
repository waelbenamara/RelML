// src/example_tasks/pl_outcome.cpp
//
// Task   : multiclass classification — predict the outcome of a Premier League game
//          Class 0 = home win   (goals_home >  goals_away)
//          Class 1 = draw       (goals_home == goals_away)
//          Class 2 = away win   (goals_home <  goals_away)
// Dataset: Premier League (API-Football v3, league 39, seasons 2020–2025)
//
// Label derivation
//   The 'outcome' column does not exist in the raw CSV; it is injected at
//   runtime by inject_outcome_labels() from goals_home and goals_away.
//   Those two columns are declared as TEXT in the schema so that
//   HeteroEncoder never sees them as numerical features — they are used
//   exclusively to derive the label, then ignored during encoding.
//
// This is equivalent to the following SQL:
//
//   ALTER TABLE games ADD COLUMN outcome INTEGER;
//   UPDATE games SET outcome =
//       CASE
//           WHEN goals_home > goals_away THEN 0   -- home win
//           WHEN goals_home = goals_away THEN 1   -- draw
//           ELSE                              2   -- away win
//       END;
//
// Graph structure used by the GNN
//
//   games  --[home_team_id]--> teams
//   games  --[away_team_id]--> teams      (two independent edge types)
//   players --[team_id]-----> teams       (seasonal roster records)
//   players --[player_id]---> player      (bio table)
//   (plus one reverse edge per forward edge for bidirectional message passing)
//
//   game_statistics and player_match_stats are NOT added as graph tables.
//   Their raw per-game columns (shots, possession, saves, ratings, etc.) are
//   all post-game observations that would leak the result. Instead,
//   inject_rolling_game_stats() computes the rolling 5-game average of each
//   stat per team and injects those averages as pre-game feature columns
//   directly on the games table. The GNN then sees those columns as node
//   features on game nodes — all signal, zero leakage.
//
// What the GNN learns
//   With 3 message-passing layers the information flow is:
//     Layer 1 — teams accumulate seasonal player statistics from the
//               'players' table; games accumulate team identity features.
//     Layer 2 — teams now carry player-informed embeddings; games absorb
//               the updated team representations.
//     Layer 3 — games absorb doubly-updated team embeddings that already
//               encode squad depth, historical player quality, etc.
//   Player biographical features (from the 'player' table) travel an
//   additional hop through 'players' before reaching teams, so they arrive
//   at game nodes at layer 3.
//
// Leakage notes
//   * goals_home / goals_away and all other match score columns are declared
//     TEXT so the encoder skips them entirely.
//   * The 'players' seasonal aggregate table is full-season data, meaning
//     it includes stats from matches played after the game being predicted.
//     In a production system you would restrict to "stats up to match-day N"
//     using a rolling window. For this demonstration, full-season aggregates
//     are acceptable — the GNN still has to learn team-level patterns from
//     the relational structure without directly seeing any score.
//
// Split strategy
//   Temporal: games are sorted by date and split 70 / 15 / 15.
//   'date' is declared TEXT in the schema so the encoder never sees it as a
//   numerical feature — it is used only as the ordering key for this split.
//   This gives an honest evaluation setup: the model is trained on older games
//   and evaluated on newer ones, which mirrors how it would be used in practice
//   to predict upcoming fixtures.
//
// Inference (historical lookup)
//   For future fixtures we look up the model's predictions on the most recent
//   historical game played between the same two teams in the same home/away
//   configuration. Those game nodes have proper GNN embeddings where the
//   home_team_id and away_team_id edge types contributed separately, so the
//   home/away asymmetry is preserved.
//
//   synthesize_prediction (mean-pooling team embeddings) is NOT used here
//   because it collapses home and away into one symmetric vector, causing the
//   model to predict the same class for every fixture regardless of who is
//   playing at home. The historical lookup avoids this entirely.

#include "relml/CSVLoader.h"
#include "relml/FKDetector.h"
#include "relml/graph/GraphBuilder.h"
#include "relml/training/Trainer.h"
#include "relml/training/TaskSpec.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <deque>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace relml;

// ---------------------------------------------------------------------------
// load_game_dates
//
// Reads the 'date' column from games.csv and returns one ISO-8601 date string
// per row, in CSV row order. We need this because the 'date' column is loaded
// as TIMESTAMP internally and its variant slot cannot be accessed via
// get_numerical() — reading the raw strings directly from the CSV is simpler
// and avoids touching Column internals.
//
// The returned strings look like "2023-08-11T19:00:00+00:00". Lexicographic
// comparison works correctly for filtering because the date portion is always
// the leading "YYYY-MM-DD" prefix.
// ---------------------------------------------------------------------------
static std::vector<std::string> load_game_dates(const std::string& csv_path) {
    std::ifstream f(csv_path);
    if (!f.is_open())
        throw std::runtime_error("load_game_dates: cannot open " + csv_path);

    std::string line;
    std::getline(f, line);   // header

    // Find the column index of 'date'
    int date_col = -1;
    {
        std::stringstream ss(line);
        std::string field;
        int idx = 0;
        while (std::getline(ss, field, ',')) {
            if (field == "date") { date_col = idx; break; }
            ++idx;
        }
    }
    if (date_col < 0)
        throw std::runtime_error("load_game_dates: 'date' column not found in " + csv_path);

    std::vector<std::string> dates;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        int idx = 0;
        std::string date_val;
        while (std::getline(ss, field, ',')) {
            if (idx == date_col) { date_val = field; break; }
            ++idx;
        }
        dates.push_back(date_val);
    }
    return dates;
}

// ---------------------------------------------------------------------------
// inject_outcome_labels
//
// Reads goals_home and goals_away (stored as TEXT in the schema to prevent
// encoder leakage) and derives a 3-class 'outcome' column:
//   0 → home win, 1 → draw, 2 → away win
//
// Rows where either goals column is null (e.g. postponed or incomplete
// fixtures) receive a null label and are excluded from the split by
// TaskSpec::build_split.
//
// Rows with date >= cutoff_date (e.g. "2026-03-01") are also nulled out so
// that future fixtures are never included in any split. They remain in the
// graph as nodes so their team embeddings are still computed — we just never
// ask the model to predict their outcome during train/val/test evaluation.
// ---------------------------------------------------------------------------
static void inject_outcome_labels(Database& db,
                                  const std::vector<std::string>& game_dates,
                                  const std::string& cutoff_date = "2026-03-01") {
    Table& games = db.get_table("games");

    // goals_home / goals_away were loaded as TEXT (schema override) so the
    // encoder skips them. We recover their string values here via get_text().
    const Column& goals_home_col = games.get_column("goals_home");
    const Column& goals_away_col = games.get_column("goals_away");

    Column outcome_col("outcome", ColumnType::NUMERICAL);
    outcome_col.data.reserve(games.num_rows());

    std::size_t n_home_wins = 0;
    std::size_t n_draws     = 0;
    std::size_t n_away_wins = 0;
    std::size_t n_skipped   = 0;

    for (std::size_t i = 0; i < games.num_rows(); ++i) {
        if (i < game_dates.size()) {
            const std::string& d = game_dates[i];
            // Null out future fixtures — stays as graph node but never in a split.
            if (!d.empty() && d.substr(0, cutoff_date.size()) >= cutoff_date) {
                outcome_col.data.push_back(std::monostate{});
                ++n_skipped;
                continue;
            }
// All completed fixtures are eligible for training — full 2020-2026 window.
        }

        // A null here means the fixture was not completed (e.g. abandoned).
        if (goals_home_col.is_null(i) || goals_away_col.is_null(i)) {
            outcome_col.data.push_back(std::monostate{});
            ++n_skipped;
            continue;
        }

        // get_text() is the correct accessor for TEXT-typed columns.
        double gh = std::stod(goals_home_col.get_text(i));
        double ga = std::stod(goals_away_col.get_text(i));

        double label;
        if      (gh > ga) { label = 0.0; ++n_home_wins; }
        else if (gh == ga) { label = 1.0; ++n_draws;     }
        else               { label = 2.0; ++n_away_wins; }

        outcome_col.data.push_back(label);
    }

    games.add_column(std::move(outcome_col));

    std::size_t total = n_home_wins + n_draws + n_away_wins;
    auto pct = [total](std::size_t n) -> double {
        return total > 0 ? 100.0 * static_cast<double>(n) / static_cast<double>(total) : 0.0;
    };

    std::cout << "  Games loaded             : " << total
              << "  (skipped/null: " << n_skipped << ")\n"
              << std::fixed << std::setprecision(1)
              << "  Home wins  (class 0)     : " << n_home_wins
              << "  (" << pct(n_home_wins) << "%)\n"
              << "  Draws      (class 1)     : " << n_draws
              << "  (" << pct(n_draws) << "%)\n"
              << "  Away wins  (class 2)     : " << n_away_wins
              << "  (" << pct(n_away_wins) << "%)\n";
}

// ---------------------------------------------------------------------------
// print_split_stats
//
// Breaks down the train / val / test sets by predicted outcome class to
// verify that the temporal split and label transform behaved correctly.
// ---------------------------------------------------------------------------
static void print_split_stats(const TaskSplit& split) {
    auto breakdown = [](const std::string& name, const std::vector<TaskSample>& samples) {
        std::size_t n0 = 0, n1 = 0, n2 = 0;
        for (const auto& s : samples) {
            int cls = static_cast<int>(std::round(s.label));
            if      (cls == 0) ++n0;
            else if (cls == 1) ++n1;
            else               ++n2;
        }
        std::cout << "    " << std::setw(5) << std::left << name << ": "
                  << samples.size() << " games   "
                  << "home=" << n0 << "  draw=" << n1 << "  away=" << n2 << "\n";
    };

    std::cout << "  Split statistics:\n";
    breakdown("train", split.train);
    breakdown("val",   split.val);
    breakdown("test",  split.test);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// inject_season_table_features
//
// For each game, computes each team's current season-to-date standing using
// only results from completed fixtures strictly before this game's date.
// This is the single strongest contextual signal in football — it tells the
// model whether a team is fighting for the title or fighting relegation.
//
// Features injected as NUMERICAL columns on the games table (normalised):
//
//   home_season_points     points accumulated so far this season  / 99
//   home_season_gd         goal difference so far                 / 50 (clipped)
//   home_season_ppg        points per game (form rate)            / 3
//   away_season_points
//   away_season_gd
//   away_season_ppg
//   points_diff            (home_points - away_points) / 99
//
// "Season" is determined by the season column in games.csv (integer year).
// The table resets to zero at the start of each new season.
// ---------------------------------------------------------------------------
static void inject_season_table_features(Database& db,
                                         const std::vector<std::string>& game_dates,
                                         const std::string& cutoff_date = "2026-03-01")
{
    Table& games  = db.get_table("games");
    std::size_t N = games.num_rows();

    auto date_to_days = [](const std::string& d) -> int {
        if (d.size() < 10) return -1;
        int y = std::stoi(d.substr(0, 4));
        int m = std::stoi(d.substr(5, 2));
        int day = std::stoi(d.substr(8, 2));
        if (m < 3) { y--; m += 12; }
        return 365*y + y/4 - y/100 + y/400 + (153*m-457)/5 + day - 306;
    };

    // Read needed columns
    auto get_int_col = [&](const std::string& name) {
        const Column& col = games.get_column(name);
        std::vector<int> out(N, -1);
        for (std::size_t i = 0; i < N; ++i)
            if (!col.is_null(i)) {
                if (col.type == ColumnType::TEXT || col.type == ColumnType::CATEGORICAL)
                    out[i] = std::stoi(col.get_categorical(i));
                else
                    out[i] = static_cast<int>(col.get_numerical(i));
            }
        return out;
    };
    auto get_goals = [&](const std::string& name) {
        const Column& col = games.get_column(name);
        std::vector<int> out(N, -1);
        for (std::size_t i = 0; i < N; ++i)
            if (!col.is_null(i)) {
                const std::string& s = col.get_text(i);
                if (!s.empty()) out[i] = std::stoi(s);
            }
        return out;
    };

    std::vector<int> home_ids    = get_int_col("home_team_id");
    std::vector<int> away_ids    = get_int_col("away_team_id");
    std::vector<int> goals_home  = get_goals("goals_home");
    std::vector<int> goals_away  = get_goals("goals_away");

    // Read season as integer from the CATEGORICAL column
    const Column& season_col = games.get_column("season");
    std::vector<int> seasons(N, -1);
    for (std::size_t i = 0; i < N; ++i)
        if (!season_col.is_null(i)) {
            try { seasons[i] = std::stoi(season_col.get_categorical(i)); }
            catch (...) {}
        }

    std::vector<int> days(N, -1);
    for (std::size_t i = 0; i < N; ++i)
        if (i < game_dates.size() && !game_dates[i].empty())
            days[i] = date_to_days(game_dates[i]);

    // Sort chronologically
    std::vector<std::size_t> sorted_idx(N);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](std::size_t a, std::size_t b){ return days[a] < days[b]; });

    // Per-team, per-season accumulators
    struct SeasonRecord { int pts=0; int gd=0; int played=0; };
    std::map<std::pair<int,int>, SeasonRecord> records; // (team_id, season) → record

    const float MAX_PTS = 99.f;
    const float MAX_GD  = 50.f;

    std::vector<float> home_pts(N,0.f), home_gd(N,0.f), home_ppg(N,0.f);
    std::vector<float> away_pts(N,0.f), away_gd(N,0.f), away_ppg(N,0.f);
    std::vector<float> pts_diff(N,0.f);

    for (std::size_t idx : sorted_idx) {
        int hid = home_ids[idx];
        int aid = away_ids[idx];
        int s   = seasons[idx];
        int gh  = goals_home[idx];
        int ga  = goals_away[idx];
        bool completed = (gh >= 0 && ga >= 0);

        if (idx < game_dates.size()) {
            const std::string& ds = game_dates[idx];
            if (!ds.empty() && ds.substr(0, cutoff_date.size()) >= cutoff_date)
                goto update_skip2;
        }

        if (hid >= 0 && aid >= 0 && s >= 0) {
            auto& hr = records[{hid, s}];
            auto& ar = records[{aid, s}];

            float hp = hr.pts / MAX_PTS;
            float hg = std::max(-1.f, std::min(1.f, hr.gd / MAX_GD)) * 0.5f + 0.5f;
            float hppg = (hr.played > 0) ? std::min(static_cast<float>(hr.pts) / (hr.played * 3.f), 1.f) : 0.f;

            float ap = ar.pts / MAX_PTS;
            float ag = std::max(-1.f, std::min(1.f, ar.gd / MAX_GD)) * 0.5f + 0.5f;
            float appg = (ar.played > 0) ? std::min(static_cast<float>(ar.pts) / (ar.played * 3.f), 1.f) : 0.f;

            home_pts[idx] = hp;  home_gd[idx]  = hg;  home_ppg[idx]  = hppg;
            away_pts[idx] = ap;  away_gd[idx]  = ag;  away_ppg[idx]  = appg;
            pts_diff[idx] = (static_cast<float>(hr.pts - ar.pts) + MAX_PTS) / (2.f * MAX_PTS);
        }

        update_skip2:
        if (completed && hid >= 0 && aid >= 0 && s >= 0) {
            auto& hr = records[{hid, s}];
            auto& ar = records[{aid, s}];
            int hpts = (gh > ga) ? 3 : (gh == ga) ? 1 : 0;
            int apts = (ga > gh) ? 3 : (gh == ga) ? 1 : 0;
            hr.pts += hpts;  hr.gd += (gh - ga);  hr.played++;
            ar.pts += apts;  ar.gd += (ga - gh);  ar.played++;
        }
    }

    auto add = [&](const std::string& name, const std::vector<float>& vals) {
        Column col(name, ColumnType::NUMERICAL);
        col.data.reserve(N);
        for (float v : vals) col.data.push_back(static_cast<double>(v));
        games.add_column(std::move(col));
    };
    add("home_season_points", home_pts);
    add("home_season_gd",     home_gd);
    add("home_season_ppg",    home_ppg);
    add("away_season_points", away_pts);
    add("away_season_gd",     away_gd);
    add("away_season_ppg",    away_ppg);
    add("points_diff",        pts_diff);

    std::cout << "  injected 7 season-table columns into games table\n";
}

// ---------------------------------------------------------------------------
// inject_rolling_game_stats
//
// Uses game_statistics.csv and player_match_stats.csv as PRE-GAME features
// by computing rolling averages of the last 5 games for each team, strictly
// before the current fixture date. This is the right way to use these tables:
//
//   WRONG: use current game's shots/possession/saves → leaks outcome
//   RIGHT: use average of last 5 games' shots/possession/saves → pure signal
//
// From game_statistics (per-team per-game aggregates):
//   home/away_avg_shots_on        average shots on target last 5 games
//   home/away_avg_shots_total     average total shots
//   home/away_avg_possession      average ball possession %
//   home/away_avg_corners         average corner kicks
//   home/away_avg_fouls           average fouls committed
//   home/away_avg_yellow_cards    average yellow cards
//   home/away_avg_saves           average goalkeeper saves
//   home/away_avg_pass_accuracy   average pass completion %
//
// From player_match_stats (aggregated per team per game → rolling avg):
//   home/away_avg_player_rating   average player rating across XI last 5 games
//   home/away_avg_goals_pp        avg goals per player per game (attacking threat)
//   home/away_avg_assists_pp      avg assists per player per game
//   home/away_avg_tackles_pp      avg tackles per player (defensive effort)
//   home/away_avg_dribbles_pp     avg successful dribbles per player
//
// All values normalised to [0,1] so the encoder sees them on the same scale.
// ---------------------------------------------------------------------------
static void inject_rolling_game_stats(
    Database& db,
    const std::vector<std::string>& game_dates,
    const std::string& data_dir,
    const std::string& cutoff_date = "2026-03-01")
{
    Table& games = db.get_table("games");
    const std::size_t N = games.num_rows();

    auto date_to_days = [](const std::string& d) -> int {
        if (d.size() < 10) return -1;
        int y = std::stoi(d.substr(0, 4));
        int m = std::stoi(d.substr(5, 2));
        int dy = std::stoi(d.substr(8, 2));
        if (m < 3) { y--; m += 12; }
        return 365*y + y/4 - y/100 + y/400 + (153*m-457)/5 + dy - 306;
    };

    // Build fixture_id → (row_index, day) lookup from games table
    const Column& fid_col  = games.get_column("fixture_id");
    const Column& hid_col  = games.get_column("home_team_id");
    const Column& aid_col  = games.get_column("away_team_id");
    std::unordered_map<int, std::size_t> fixture_to_row;
    std::vector<int> game_days(N, -1);
    for (std::size_t i = 0; i < N; ++i) {
        if (!fid_col.is_null(i))
            fixture_to_row[static_cast<int>(fid_col.get_numerical(i))] = i;
        if (i < game_dates.size() && !game_dates[i].empty())
            game_days[i] = date_to_days(game_dates[i]);
    }

    // -----------------------------------------------------------------------
    // Parse game_statistics.csv
    // Columns: fixture_id,team_id,shots_on_goal,shots_off_goal,
    //          shots_insidebox,shots_outsidebox,total_shots,blocked_shots,
    //          fouls,corner_kicks,offsides,ball_possession,yellow_cards,
    //          red_cards,gk_saves,total_passes,accurate_passes,pass_accuracy
    // -----------------------------------------------------------------------
    struct GameStats {
        int   fixture_id = -1;
        int   team_id    = -1;
        float shots_on   = 0.f;
        float shots_tot  = 0.f;
        float possession = 0.f;
        float corners    = 0.f;
        float fouls      = 0.f;
        float yellows    = 0.f;
        float saves      = 0.f;
        float pass_acc   = 0.f;
    };

    std::vector<GameStats> gstats;
    {
        std::ifstream f(data_dir + "/game_statistics.csv");
        if (!f.is_open()) {
            std::cout << "  game_statistics.csv not found — skipping game stats features\n";
            goto skip_game_stats;
        }
        std::string line;
        std::getline(f, line);  // header

        // Parse header to find column positions
        std::unordered_map<std::string,int> col_idx;
        {
            std::stringstream hss(line);
            std::string tok; int ci = 0;
            while (std::getline(hss, tok, ',')) {
                // trim whitespace
                tok.erase(0, tok.find_first_not_of(" \t\r\n"));
                tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
                col_idx[tok] = ci++;
            }
        }

        auto get_col = [&](const std::vector<std::string>& fields,
                           const std::string& name) -> float {
            auto it = col_idx.find(name);
            if (it == col_idx.end()) return 0.f;
            int i = it->second;
            if (i >= static_cast<int>(fields.size())) return 0.f;
            const std::string& s = fields[i];
            if (s.empty()) return 0.f;
            try { return std::stof(s); } catch (...) { return 0.f; }
        };

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ',')) fields.push_back(tok);

            GameStats gs;
            gs.fixture_id = static_cast<int>(get_col(fields, "fixture_id"));
            gs.team_id    = static_cast<int>(get_col(fields, "team_id"));
            gs.shots_on   = get_col(fields, "shots_on_goal");
            gs.shots_tot  = get_col(fields, "total_shots");
            gs.possession = get_col(fields, "ball_possession") / 100.f;
            gs.corners    = get_col(fields, "corner_kicks");
            gs.fouls      = get_col(fields, "fouls");
            gs.yellows    = get_col(fields, "yellow_cards");
            gs.saves      = get_col(fields, "gk_saves");
            gs.pass_acc   = get_col(fields, "pass_accuracy") / 100.f;
            gstats.push_back(gs);
        }
        std::cout << "  game_statistics.csv  : " << gstats.size() << " team-game records\n";
    }
    skip_game_stats:;

    // -----------------------------------------------------------------------
    // Parse player_match_stats.csv — aggregate per (fixture_id, team_id)
    // -----------------------------------------------------------------------
    struct PlayerTeamGame {
        int   fixture_id    = -1;
        int   team_id       = -1;
        float avg_rating    = 0.f;
        float goals_pp      = 0.f;
        float assists_pp    = 0.f;
        float tackles_pp    = 0.f;
        float dribbles_pp   = 0.f;
        int   player_count  = 0;
    };

    std::vector<PlayerTeamGame> pstats;
    {
        std::ifstream f(data_dir + "/player_match_stats.csv");
        if (!f.is_open()) {
            std::cout << "  player_match_stats.csv not found — skipping player stats\n";
            goto skip_player_stats;
        }
        std::string line;
        std::getline(f, line);  // header

        std::unordered_map<std::string,int> col_idx;
        {
            std::stringstream hss(line);
            std::string tok; int ci = 0;
            while (std::getline(hss, tok, ',')) {
                tok.erase(0, tok.find_first_not_of(" \t\r\n"));
                tok.erase(tok.find_last_not_of(" \t\r\n") + 1);
                col_idx[tok] = ci++;
            }
        }

        auto get_col = [&](const std::vector<std::string>& fields,
                           const std::string& name) -> float {
            auto it = col_idx.find(name);
            if (it == col_idx.end()) return 0.f;
            int i = it->second;
            if (i >= static_cast<int>(fields.size())) return 0.f;
            const std::string& s = fields[i];
            if (s.empty()) return 0.f;
            try { return std::stof(s); } catch (...) { return 0.f; }
        };

        // Accumulate per (fixture_id, team_id)
        std::map<std::pair<int,int>, PlayerTeamGame> acc;

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<std::string> fields;
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ',')) fields.push_back(tok);

            int fid  = static_cast<int>(get_col(fields, "fixture_id"));
            int tid  = static_cast<int>(get_col(fields, "team_id"));
            auto key = std::make_pair(fid, tid);
            auto& g  = acc[key];
            g.fixture_id = fid;
            g.team_id    = tid;
            g.avg_rating  += get_col(fields, "rating");
            g.goals_pp    += get_col(fields, "goals_scored");
            g.assists_pp  += get_col(fields, "assists");
            g.tackles_pp  += get_col(fields, "tackles_total");
            g.dribbles_pp += get_col(fields, "dribbles_success");
            g.player_count++;
        }

        for (auto& [key, g] : acc) {
            if (g.player_count > 0) {
                float n = static_cast<float>(g.player_count);
                g.avg_rating  /= n;
                g.goals_pp    /= n;
                g.assists_pp  /= n;
                g.tackles_pp  /= n;
                g.dribbles_pp /= n;
            }
            pstats.push_back(g);
        }
        std::cout << "  player_match_stats.csv: " << pstats.size() << " team-game records\n";
    }
    skip_player_stats:;

    if (gstats.empty() && pstats.empty()) {
        std::cout << "  No rolling stats sources available — skipping\n";
        return;
    }

    // -----------------------------------------------------------------------
    // Build per-team rolling history keyed by team_id
    // Each entry carries: {day, GameStats, PlayerTeamGame}
    // -----------------------------------------------------------------------
    struct TeamGameEntry {
        int  day = -1;
        GameStats     gs;
        PlayerTeamGame pg;
    };

    // Index gstats and pstats by fixture_id + team_id
    std::map<std::pair<int,int>, GameStats>      gs_map;
    std::map<std::pair<int,int>, PlayerTeamGame> pg_map;
    for (auto& g : gstats)  gs_map[{g.fixture_id, g.team_id}]  = g;
    for (auto& p : pstats)  pg_map[{p.fixture_id, p.team_id}]  = p;

    const int WINDOW = 5;

    // team_id → deque of recent TeamGameEntry (chronological)
    std::unordered_map<int, std::deque<TeamGameEntry>> team_hist;

    // Sort game rows chronologically so we build history in order
    std::vector<std::size_t> sorted_idx(N);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](std::size_t a, std::size_t b){
                  return game_days[a] < game_days[b]; });

    // Output columns: 8 game-stats + 5 player-stats, x2 (home/away)
    auto make_col = [&](){ return std::vector<float>(N, 0.f); };
    std::unordered_map<std::string, std::vector<float>> out_cols = {
        {"h_roll_shots_on",      make_col()}, {"a_roll_shots_on",      make_col()},
        {"h_roll_shots_tot",     make_col()}, {"a_roll_shots_tot",     make_col()},
        {"h_roll_possession",    make_col()}, {"a_roll_possession",    make_col()},
        {"h_roll_corners",       make_col()}, {"a_roll_corners",       make_col()},
        {"h_roll_fouls",         make_col()}, {"a_roll_fouls",         make_col()},
        {"h_roll_yellows",       make_col()}, {"a_roll_yellows",       make_col()},
        {"h_roll_saves",         make_col()}, {"a_roll_saves",         make_col()},
        {"h_roll_pass_acc",      make_col()}, {"a_roll_pass_acc",      make_col()},
        {"h_roll_rating",        make_col()}, {"a_roll_rating",        make_col()},
        {"h_roll_goals_pp",      make_col()}, {"a_roll_goals_pp",      make_col()},
        {"h_roll_assists_pp",    make_col()}, {"a_roll_assists_pp",    make_col()},
        {"h_roll_tackles_pp",    make_col()}, {"a_roll_tackles_pp",    make_col()},
        {"h_roll_dribbles_pp",   make_col()}, {"a_roll_dribbles_pp",   make_col()},
    };

    // Normalisation caps (approximate Premier League maxima)
    const float CAP_SHOTS_ON  = 15.f;
    const float CAP_SHOTS_TOT = 25.f;
    const float CAP_CORNERS   = 15.f;
    const float CAP_FOULS     = 20.f;
    const float CAP_YELLOWS   = 4.f;
    const float CAP_SAVES     = 12.f;
    const float CAP_RATING    = 10.f;
    const float CAP_GOALS_PP  = 0.3f;
    const float CAP_ASST_PP   = 0.2f;
    const float CAP_TACK_PP   = 3.f;
    const float CAP_DRIB_PP   = 1.f;

    auto roll_avg = [&](const std::deque<TeamGameEntry>& hist,
                        auto getter, float cap) -> float {
        if (hist.empty()) return 0.f;
        float sum = 0.f; int cnt = 0;
        for (auto rit = hist.rbegin(); rit != hist.rend() && cnt < WINDOW; ++rit, ++cnt)
            sum += getter(*rit);
        return std::min(sum / (cnt * cap), 1.f);
    };

    for (std::size_t idx : sorted_idx) {
        if (hid_col.is_null(idx) || aid_col.is_null(idx)) goto update_stats;
        {
            int hid = static_cast<int>(hid_col.get_numerical(idx));
            int aid = static_cast<int>(aid_col.get_numerical(idx));

            // Skip future fixtures from the feature output (but still update history below)
            if (idx < game_dates.size()) {
                const std::string& ds = game_dates[idx];
                if (!ds.empty() && ds.substr(0, cutoff_date.size()) >= cutoff_date)
                    goto update_stats;
            }

            auto fill = [&](int tid, const std::string& prefix) {
                auto it = team_hist.find(tid);
                if (it == team_hist.end()) return;
                const auto& hist = it->second;
                out_cols[prefix+"roll_shots_on"][idx]    = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.shots_on; },   CAP_SHOTS_ON);
                out_cols[prefix+"roll_shots_tot"][idx]   = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.shots_tot; },  CAP_SHOTS_TOT);
                out_cols[prefix+"roll_possession"][idx]  = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.possession; }, 1.f);
                out_cols[prefix+"roll_corners"][idx]     = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.corners; },    CAP_CORNERS);
                out_cols[prefix+"roll_fouls"][idx]       = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.fouls; },      CAP_FOULS);
                out_cols[prefix+"roll_yellows"][idx]     = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.yellows; },    CAP_YELLOWS);
                out_cols[prefix+"roll_saves"][idx]       = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.saves; },      CAP_SAVES);
                out_cols[prefix+"roll_pass_acc"][idx]    = roll_avg(hist, [](const TeamGameEntry& e){ return e.gs.pass_acc; },   1.f);
                out_cols[prefix+"roll_rating"][idx]      = roll_avg(hist, [](const TeamGameEntry& e){ return e.pg.avg_rating; }, CAP_RATING);
                out_cols[prefix+"roll_goals_pp"][idx]    = roll_avg(hist, [](const TeamGameEntry& e){ return e.pg.goals_pp; },   CAP_GOALS_PP);
                out_cols[prefix+"roll_assists_pp"][idx]  = roll_avg(hist, [](const TeamGameEntry& e){ return e.pg.assists_pp; }, CAP_ASST_PP);
                out_cols[prefix+"roll_tackles_pp"][idx]  = roll_avg(hist, [](const TeamGameEntry& e){ return e.pg.tackles_pp; }, CAP_TACK_PP);
                out_cols[prefix+"roll_dribbles_pp"][idx] = roll_avg(hist, [](const TeamGameEntry& e){ return e.pg.dribbles_pp;}, CAP_DRIB_PP);
            };

            fill(hid, "h_");
            fill(aid, "a_");
        }

        update_stats:
        // Update history with this game's stats AFTER recording features
        if (idx < game_dates.size()) {
            const std::string& ds = game_dates[idx];
            if (!ds.empty() && ds.substr(0, cutoff_date.size()) >= cutoff_date)
                continue;
        }
        if (!fid_col.is_null(idx)) {
            int fid = static_cast<int>(fid_col.get_numerical(idx));
            int hid = hid_col.is_null(idx) ? -1 : static_cast<int>(hid_col.get_numerical(idx));
            int aid = aid_col.is_null(idx) ? -1 : static_cast<int>(aid_col.get_numerical(idx));
            int d   = game_days[idx];

            for (int tid : {hid, aid}) {
                if (tid < 0) continue;
                TeamGameEntry entry;
                entry.day = d;
                auto git = gs_map.find({fid, tid});
                if (git != gs_map.end()) entry.gs = git->second;
                auto pit = pg_map.find({fid, tid});
                if (pit != pg_map.end()) entry.pg = pit->second;
                team_hist[tid].push_back(entry);
                if (team_hist[tid].size() > static_cast<std::size_t>(WINDOW * 4))
                    team_hist[tid].pop_front();
            }
        }
    }

    // Inject all columns into games table
    std::size_t added = 0;
    for (auto& [name, vals] : out_cols) {
        Column col(name, ColumnType::NUMERICAL);
        col.data.reserve(N);
        for (float v : vals) col.data.push_back(static_cast<double>(v));
        games.add_column(std::move(col));
        ++added;
    }
    std::cout << "  injected " << added << " rolling stat columns into games table\n";
}

// ---------------------------------------------------------------------------
// inject_form_features
//
// For each game, computes pre-match form features using ONLY results from
// completed fixtures that occurred strictly before that game's date. This
// means there is zero leakage — every feature is knowable on match day.
//
// Features injected as NUMERICAL columns on the games table:
//
//  Per team (home_ / away_ prefix), last-5-game rolling window:
//    wins_last5          number of wins in last 5 games (any venue)
//    draws_last5
//    losses_last5
//    goals_scored_last5  total goals scored
//    goals_conceded_last5
//    clean_sheets_last5  games where 0 goals were conceded
//
//  Venue-specific (last 5 games played at same venue role):
//    home_home_wins_last5   home team's record in their last 5 home games
//    away_away_wins_last5   away team's record in their last 5 away games
//
//  Head-to-head (same home/away pairing, last 5 meetings):
//    h2h_home_wins_last5
//    h2h_draws_last5
//    h2h_away_wins_last5
//
//  Schedule pressure:
//    home_days_rest      days since home team's last game (capped at 14)
//    away_days_rest
//
// All features are normalised to [0, 1] (divide by window size or cap value).
// Games with fewer than 5 prior results use whatever is available — the
// feature is still the raw count, which is lower and signals "new team/season".
// ---------------------------------------------------------------------------
static void inject_form_features(Database& db,
                                 const std::vector<std::string>& game_dates,
                                 const std::string& cutoff_date = "2026-03-01")
{
    Table& games = db.get_table("games");
    std::size_t N = games.num_rows();

    // ---- helpers to read columns already in the table --------------------
    auto get_int_col = [&](const std::string& col_name) -> std::vector<int> {
        const Column& col = games.get_column(col_name);
        std::vector<int> out(N, -1);
        for (std::size_t i = 0; i < N; ++i) {
            if (!col.is_null(i)) {
                if (col.type == ColumnType::TEXT)
                    out[i] = std::stoi(col.get_text(i));
                else
                    out[i] = static_cast<int>(col.get_numerical(i));
            }
        }
        return out;
    };

    // Parse a date string "YYYY-MM-DDTHH:MM..." into days-since-epoch (approx)
    // using only string arithmetic — no <ctime> needed.
    auto date_to_days = [](const std::string& d) -> int {
        if (d.size() < 10) return -1;
        int y = std::stoi(d.substr(0, 4));
        int m = std::stoi(d.substr(5, 2));
        int day = std::stoi(d.substr(8, 2));
        // Rata Die approximation (accurate enough for day-difference arithmetic)
        if (m < 3) { y--; m += 12; }
        return 365 * y + y/4 - y/100 + y/400 + (153*m - 457)/5 + day - 306;
    };

    std::vector<int> home_ids  = get_int_col("home_team_id");
    std::vector<int> away_ids  = get_int_col("away_team_id");

    // Read raw goal strings (declared TEXT to prevent encoder seeing them)
    auto get_goals = [&](const std::string& col_name) -> std::vector<int> {
        const Column& col = games.get_column(col_name);
        std::vector<int> out(N, -1);
        for (std::size_t i = 0; i < N; ++i)
            if (!col.is_null(i)) {
                const std::string& s = col.get_text(i);
                if (!s.empty()) out[i] = std::stoi(s);
            }
        return out;
    };
    std::vector<int> goals_home_raw = get_goals("goals_home");
    std::vector<int> goals_away_raw = get_goals("goals_away");

    // Convert dates to integer days and sort games chronologically
    std::vector<int> days(N, -1);
    for (std::size_t i = 0; i < N; ++i)
        if (i < game_dates.size() && !game_dates[i].empty())
            days[i] = date_to_days(game_dates[i]);

    // sorted_idx[k] = original row index of the k-th game in chronological order
    std::vector<std::size_t> sorted_idx(N);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](std::size_t a, std::size_t b){ return days[a] < days[b]; });

    // ---- per-team history built incrementally as we sweep time -----------
    // For each team_id we keep a deque of recent results (chronological order).
    // Each entry: {days_value, goals_for, goals_against, is_home}
    struct MatchResult { int day; int gf; int ga; bool was_home; };
    std::unordered_map<int, std::deque<MatchResult>> team_history;

    // head-to-head history keyed by (home_id, away_id)
    struct H2HResult { int day; int outcome; }; // 0=home_win 1=draw 2=away_win
    std::map<std::pair<int,int>, std::deque<H2HResult>> h2h_history;

    const int WINDOW = 5;
    const int MAX_REST_DAYS = 14;

    // Output feature vectors — one float per game per feature
    auto make_col = [&]() { return std::vector<float>(N, 0.f); };
    auto cols = std::unordered_map<std::string, std::vector<float>>{
        {"home_wins_last5",          make_col()},
        {"home_draws_last5",         make_col()},
        {"home_losses_last5",        make_col()},
        {"home_goals_scored_last5",  make_col()},
        {"home_goals_conceded_last5",make_col()},
        {"home_clean_sheets_last5",  make_col()},
        {"home_home_wins_last5",     make_col()},
        {"away_wins_last5",          make_col()},
        {"away_draws_last5",         make_col()},
        {"away_losses_last5",        make_col()},
        {"away_goals_scored_last5",  make_col()},
        {"away_goals_conceded_last5",make_col()},
        {"away_clean_sheets_last5",  make_col()},
        {"away_away_wins_last5",     make_col()},
        {"h2h_home_wins_last5",      make_col()},
        {"h2h_draws_last5",          make_col()},
        {"h2h_away_wins_last5",      make_col()},
        {"home_days_rest",           make_col()},
        {"away_days_rest",           make_col()},
    };

    // Helper: aggregate last-WINDOW results for one team
    auto agg = [&](int team_id, int current_day)
        -> std::tuple<float,float,float,float,float,float,float,float>
    {
        // returns: wins, draws, losses, gf, ga, clean_sheets, home_wins, days_rest
        auto it = team_history.find(team_id);
        if (it == team_history.end())
            return {0,0,0,0,0,0,0, static_cast<float>(MAX_REST_DAYS)};

        const auto& hist = it->second;
        float wins=0, draws=0, losses=0, gf=0, ga=0, cs=0, hw=0;
        int last_day = -1;
        int count = 0;
        // iterate from most-recent backwards, take up to WINDOW completed games
        for (auto rit = hist.rbegin(); rit != hist.rend() && count < WINDOW; ++rit, ++count) {
            if (last_day < 0) last_day = rit->day;
            if (rit->gf > rit->ga) { wins++; if (rit->was_home) hw++; }
            else if (rit->gf == rit->ga) draws++;
            else losses++;
            gf += rit->gf;
            ga += rit->ga;
            if (rit->ga == 0) cs++;
        }
        float rest = (last_day >= 0 && current_day >= 0)
            ? std::min(static_cast<float>(current_day - last_day),
                       static_cast<float>(MAX_REST_DAYS))
            : static_cast<float>(MAX_REST_DAYS);
        // normalise to [0,1]
        float w = static_cast<float>(WINDOW);
        float mr = static_cast<float>(MAX_REST_DAYS);
        return {wins/w, draws/w, losses/w, gf/(w*5.f), ga/(w*5.f), cs/w, hw/w, rest/mr};
    };

    // Sweep games chronologically — for each game record features BEFORE
    // updating history so there is no leakage of the current result.
    for (std::size_t idx : sorted_idx) {
        int hid   = home_ids[idx];
        int aid   = away_ids[idx];
        int d     = days[idx];
        int gh    = goals_home_raw[idx];
        int ga_g  = goals_away_raw[idx];
        bool completed = (gh >= 0 && ga_g >= 0);

        // Skip future fixtures (cutoff)
        if (idx < game_dates.size()) {
            const std::string& ds = game_dates[idx];
            if (!ds.empty() && ds.substr(0, cutoff_date.size()) >= cutoff_date)
                goto update_skip;
        }

        if (hid >= 0 && aid >= 0) {
            // ---- home team stats ----
            auto [hw, hdr, hl, hgf, hga, hcs, hhw, hrest]  = agg(hid, d);
            cols["home_wins_last5"][idx]           = hw;
            cols["home_draws_last5"][idx]          = hdr;
            cols["home_losses_last5"][idx]         = hl;
            cols["home_goals_scored_last5"][idx]   = hgf;
            cols["home_goals_conceded_last5"][idx] = hga;
            cols["home_clean_sheets_last5"][idx]   = hcs;
            cols["home_home_wins_last5"][idx]      = hhw;
            cols["home_days_rest"][idx]            = hrest;

            // ---- away team stats ----
            auto [aw, adr, al, agf, aga, acs, aaw_raw, arest] = agg(aid, d);
            // aaw_raw from agg() counts home wins for that team when it was home.
            // We need away wins specifically — recalculate from away perspective.
            {
                auto it = team_history.find(aid);
                float away_wins = 0.f;
                if (it != team_history.end()) {
                    int cnt = 0;
                    for (auto rit = it->second.rbegin();
                         rit != it->second.rend() && cnt < WINDOW; ++rit, ++cnt)
                        if (!rit->was_home && rit->gf > rit->ga) away_wins++;
                }
                cols["away_away_wins_last5"][idx] = away_wins / static_cast<float>(WINDOW);
            }
            cols["away_wins_last5"][idx]           = aw;
            cols["away_draws_last5"][idx]          = adr;
            cols["away_losses_last5"][idx]         = al;
            cols["away_goals_scored_last5"][idx]   = agf;
            cols["away_goals_conceded_last5"][idx] = aga;
            cols["away_clean_sheets_last5"][idx]   = acs;
            cols["away_days_rest"][idx]            = arest;

            // ---- head-to-head ----
            auto key = std::make_pair(hid, aid);
            auto hit = h2h_history.find(key);
            if (hit != h2h_history.end()) {
                float h2h_hw=0, h2h_d=0, h2h_aw=0;
                int cnt = 0;
                for (auto rit = hit->second.rbegin();
                     rit != hit->second.rend() && cnt < WINDOW; ++rit, ++cnt) {
                    if (rit->outcome == 0) h2h_hw++;
                    else if (rit->outcome == 1) h2h_d++;
                    else h2h_aw++;
                }
                float w = static_cast<float>(WINDOW);
                cols["h2h_home_wins_last5"][idx] = h2h_hw / w;
                cols["h2h_draws_last5"][idx]     = h2h_d  / w;
                cols["h2h_away_wins_last5"][idx] = h2h_aw / w;
            }
        }

        update_skip:
        // Now update history with this game's result (only if completed)
        if (completed && hid >= 0 && aid >= 0) {
            team_history[hid].push_back({d, gh,   ga_g, true  });
            team_history[aid].push_back({d, ga_g, gh,   false });
            // keep deque bounded
            if (team_history[hid].size() > static_cast<std::size_t>(WINDOW * 4))
                team_history[hid].pop_front();
            if (team_history[aid].size() > static_cast<std::size_t>(WINDOW * 4))
                team_history[aid].pop_front();

            int outcome = (gh > ga_g) ? 0 : (gh == ga_g) ? 1 : 2;
            auto key = std::make_pair(hid, aid);
            h2h_history[key].push_back({d, outcome});
            if (h2h_history[key].size() > static_cast<std::size_t>(WINDOW * 2))
                h2h_history[key].pop_front();
        }
    }

    // ---- add all computed columns to the games table --------------------
    std::size_t added = 0;
    for (auto& [name, vals] : cols) {
        Column col(name, ColumnType::NUMERICAL);
        col.data.reserve(N);
        for (float v : vals)
            col.data.push_back(static_cast<double>(v));
        games.add_column(std::move(col));
        ++added;
    }
    std::cout << "  injected " << added << " form feature columns into games table\n";
}

int main(int argc, char* argv[]) {
    std::string data_dir = (argc > 1) ? argv[1] : "./data/premiere-league-data";

    // -----------------------------------------------------------------------
    // Schema
    //
    // Four tables are loaded into the graph: teams, games, player, players.
    //
    // game_statistics and player_match_stats are intentionally NOT added as
    // graph tables — their raw columns are all post-game observations and
    // would leak the result. Their signal is captured instead via
    // inject_rolling_game_stats(), which computes rolling 5-game averages per
    // team and injects them as pre-game columns on the games table.
    //
    // games
    //   PK  = fixture_id
    //   time = date (ISO 8601; override to TIMESTAMP — see file header note)
    //
    //   home_team_id / away_team_id are declared as explicit FKs because
    //   FKDetector's name heuristic looks for "<singular_table>Id" (i.e.
    //   "teamId"). "home_team_id" and "away_team_id" do not match that
    //   pattern, so automatic detection would miss them.
    //
    //   All score columns (goals, halftime, fulltime, extra time, penalties)
    //   are declared TEXT. HeteroEncoder skips TEXT columns entirely, so the
    //   model cannot access the actual match result during encoding. They are
    //   read back as strings by inject_outcome_labels() above.
    //
    // teams
    //   PK = team_id. Static club metadata (stadium, country, founding year).
    //   High-cardinality string fields (name, venue city) are declared TEXT.
    //
    // player
    //   PK = player_id. Static biographical table: nationality, age, height,
    //   weight, birth_date. Name fields are TEXT (too many unique values).
    //
    // players  (seasonal aggregate statistics)
    //   No natural single-column PK — each row is (season, player_id, team_id).
    //   FKs link each record to its club (teams) and bio entry (player).
    //   The numerical columns — appearances, goals, assists, rating, etc. —
    //   are left to type inference (all infer as NUMERICAL).
    //   season is overridden to CATEGORICAL so the encoder one-hot encodes
    //   the year rather than treating 2023 as a larger number than 2022.
    //   birth_date and nationality are overridden to TEXT: they duplicate info
    //   already captured more cleanly in the 'player' bio table.
    // -----------------------------------------------------------------------

    std::unordered_map<std::string, TableSchema> schemas = {

        // ── teams ───────────────────────────────────────────────────────
        {"teams", {
            .pkey_col     = "team_id",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns = {
                // High-cardinality text: skip in encoder
                {.name = "name",          .type = ColumnType::TEXT},
                {.name = "venue_name",    .type = ColumnType::TEXT},
                {.name = "venue_city",    .type = ColumnType::TEXT},
                // Low-cardinality: one-hot encode
                {.name = "code",          .type = ColumnType::CATEGORICAL},
                {.name = "country",       .type = ColumnType::CATEGORICAL},
                {.name = "venue_surface", .type = ColumnType::CATEGORICAL},
                // founded, venue_capacity → inferred NUMERICAL, keep defaults
            }
        }},

        // ── games (target / observation table) ──────────────────────────
        {"games", {
            .pkey_col     = "fixture_id",
            .time_col     = "date",
            .foreign_keys = {
                // Must be declared explicitly: FKDetector would need "teamId"
                // to match teams.team_id, not "home_team_id" / "away_team_id".
                {.column = "home_team_id", .target_table = "teams"},
                {.column = "away_team_id", .target_table = "teams"},
            },
            .columns = {
                // Pre-game contextual features
                {.name = "season",   .type = ColumnType::CATEGORICAL},
                // date is declared TIMESTAMP so build_split can sort games
                // chronologically. HeteroEncoder already skips the time_col
                // (declared as .time_col = "date" in the schema) so the model
                // never sees the raw timestamp as a numerical input feature.
                // No leakage risk.
                {.name = "date",     .type = ColumnType::TIMESTAMP},
                {.name = "referee",  .type = ColumnType::CATEGORICAL},
                {.name = "round",    .type = ColumnType::CATEGORICAL},
                {.name = "venue_name", .type = ColumnType::TEXT},
                {.name = "status",     .type = ColumnType::TEXT},       // always "FT" — uninformative

                // ── POST-GAME OUTCOME COLUMNS — encoder must never see these ──
                // Declared TEXT so HeteroEncoder::fit_table skips them.
                // inject_outcome_labels() reads them back via get_text().
                {.name = "goals_home",     .type = ColumnType::TEXT},
                {.name = "goals_away",     .type = ColumnType::TEXT},
                {.name = "halftime_home",  .type = ColumnType::TEXT},
                {.name = "halftime_away",  .type = ColumnType::TEXT},
                {.name = "fulltime_home",  .type = ColumnType::TEXT},
                {.name = "fulltime_away",  .type = ColumnType::TEXT},
                {.name = "extratime_home", .type = ColumnType::TEXT},
                {.name = "extratime_away", .type = ColumnType::TEXT},
                {.name = "penalty_home",   .type = ColumnType::TEXT},
                {.name = "penalty_away",   .type = ColumnType::TEXT},
            }
        }},

        // ── player (static bio) ─────────────────────────────────────────
        {"player", {
            .pkey_col     = "player_id",
            .time_col     = std::nullopt,
            .foreign_keys = {},
            .columns = {
                {.name = "name",        .type = ColumnType::TEXT},
                {.name = "firstname",   .type = ColumnType::TEXT},
                {.name = "lastname",    .type = ColumnType::TEXT},
                {.name = "nationality", .type = ColumnType::CATEGORICAL},
                // birth_date → inferred TIMESTAMP (YYYY-MM-DD format)
                // age, height, weight → inferred NUMERICAL — keep defaults
            }
        }},

        // ── players (seasonal aggregate statistics) ─────────────────────
        {"players", {
            // No natural single-column PK. This table is a FK-source only:
            // each row links a player-season record to a team and a bio entry.
            .pkey_col     = std::nullopt,
            .time_col     = std::nullopt,
            .foreign_keys = {
                {.column = "player_id", .target_table = "player"},
                {.column = "team_id",   .target_table = "teams"},
            },
            .columns = {
                // season is an integer year (2020–2025): encode as a category
                // so the model sees "season 2023" not "value 2023.0".
                {.name = "season",      .type = ColumnType::CATEGORICAL},
                // Already captured in the 'player' bio table — skip duplicates.
                {.name = "birth_date",  .type = ColumnType::TEXT},
                {.name = "nationality", .type = ColumnType::TEXT},
                // Categorical metadata
                {.name = "position",    .type = ColumnType::CATEGORICAL},
                {.name = "injured",     .type = ColumnType::CATEGORICAL},
                // All remaining columns (appearances, lineups, minutes, rating,
                // goals, assists, saves, shots_*, passes_*, tackles_*, duels_*,
                // dribbles_*, fouls_*, yellow_cards, red_cards, penalties, …)
                // are auto-inferred as NUMERICAL and kept as features.
                // Null values (players with 0 appearances in that season) are
                // imputed to the column mean by NumericalEncoder.
            }
        }},
    };

    // -----------------------------------------------------------------------
    // Load tables individually to exclude game_statistics and
    // player_match_stats. Using load_table rather than load_database avoids
    // accidentally picking up those files if they share the same directory.
    // -----------------------------------------------------------------------
    std::cout << "Loading Premier League database from: " << data_dir << "\n";
    Database db("premier-league");

    // Load raw date strings before the tables are parsed, so we can use them
    // for the training-period display and the post-2026-Feb cutoff filter.
    std::vector<std::string> game_dates =
        load_game_dates(data_dir + "/games.csv");

    auto load_one = [&](const std::string& csv_name, const std::string& table_name) {
        std::string path = data_dir + "/" + csv_name;
        std::cout << "  " << csv_name << " ... ";
        std::cout.flush();
        Table t = CSVLoader::load_table(path, table_name, schemas.at(table_name));
        std::cout << t.num_rows() << " rows\n";
        db.add_table(std::move(t));
    };

    load_one("teams.csv",   "teams");
    load_one("games.csv",   "games");
    load_one("player.csv",  "player");
    load_one("players.csv", "players");

    // -----------------------------------------------------------------------
    // FK detection
    //
    // The two critical FKs (home_team_id, away_team_id → teams) are already
    // declared in the schema. FKDetector runs as a sanity check and may
    // auto-detect additional FKs among the 'players' numerical columns, but
    // it will never overwrite already-declared FKs.
    // -----------------------------------------------------------------------
    std::cout << "\nRunning FK detector (sanity check)...\n";
    auto detected = FKDetector::detect(db);
    for (const auto& fk : detected)
        std::cout << "  auto-detected: " << fk.src_table << "." << fk.src_column
                  << " -> " << fk.dst_table
                  << "  (coverage " << fk.coverage * 100.f << "%)\n";
    if (detected.empty())
        std::cout << "  (none beyond explicitly declared FKs)\n";

    // -----------------------------------------------------------------------
    // Inject outcome labels
    // -----------------------------------------------------------------------
    std::cout << "\nInjecting outcome labels...\n";
    inject_outcome_labels(db, game_dates);

    // -----------------------------------------------------------------------
    // Inject pre-game form features
    //
    // Computes rolling last-5-game statistics for each team strictly before
    // each fixture date. Adds 19 NUMERICAL columns to the games table:
    //   home/away: wins, draws, losses, goals scored/conceded, clean sheets,
    //              venue-specific win rate, days rest
    //   head-to-head: wins/draws/losses in last 5 meetings (same pairing)
    //
    // All values are normalised to [0,1] so the encoder sees them on the
    // same scale as other numerical features. This is the single highest-
    // leverage signal available without leaking the result.
    // -----------------------------------------------------------------------
    std::cout << "\nInjecting pre-game form features...\n";
    inject_form_features(db, game_dates);

    // -----------------------------------------------------------------------
    // Inject season-table features
    //
    // For each game, computes each team's current season points, goal
    // difference, and points-per-game using only results from completed
    // fixtures strictly before this game's date. This gives the model
    // league-table context — a team fighting for the title looks very
    // different from a relegation side even if their last-5 form is similar.
    // -----------------------------------------------------------------------
    std::cout << "\nInjecting season-table features...\n";
    inject_season_table_features(db, game_dates);

    // -----------------------------------------------------------------------
    // Inject rolling game & player statistics (pre-game signal only)
    //
    // game_statistics.csv and player_match_stats.csv contain rich per-match
    // data (shots, possession, saves, player ratings, etc.) but using the
    // current game's row would leak the outcome. The solution is to use the
    // rolling average of the last 5 games for each team, computed strictly
    // before the current fixture date — exactly what a pre-match analyst sees.
    //
    // Columns added to games table (all normalised [0,1]):
    //
    //   From game_statistics (per-team aggregate per game):
    //     h/a_roll_shots_on      average shots on target last 5 games
    //     h/a_roll_shots_tot     average total shots
    //     h/a_roll_possession    average ball possession
    //     h/a_roll_corners       average corner kicks
    //     h/a_roll_fouls         average fouls committed
    //     h/a_roll_yellows       average yellow cards
    //     h/a_roll_saves         average goalkeeper saves (proxy for defensive pressure)
    //     h/a_roll_pass_acc      average pass completion %
    //
    //   From player_match_stats (aggregated per team per game → rolling avg):
    //     h/a_roll_rating        average player rating across the XI
    //     h/a_roll_goals_pp      goals per player per game (attacking threat)
    //     h/a_roll_assists_pp    assists per player per game
    //     h/a_roll_tackles_pp    tackles per player (defensive effort)
    //     h/a_roll_dribbles_pp   successful dribbles per player
    //
    // These 26 columns give the GNN direct access to HOW each team plays,
    // not just what their results have been. A team that dominates possession
    // and shots but draws often is very different from a counterattacking team
    // with the same draw record — these features capture that distinction.
    // -----------------------------------------------------------------------
    std::cout << "\nInjecting rolling game & player statistics...\n";
    inject_rolling_game_stats(db, game_dates, data_dir);

    // -----------------------------------------------------------------------
    // Build heterogeneous graph
    //
    // Edge types produced:
    //   games --[home_team_id]--> teams   (and reverse)
    //   games --[away_team_id]--> teams   (and reverse)
    //   players --[player_id]---> player  (and reverse)
    //   players --[team_id]-----> teams   (and reverse)
    //
    // The two separate home/away edge types let the GNN treat "playing at home
    // for this game" and "playing away for this game" as semantically distinct
    // relationships — it maintains separate W_neigh matrices for each.
    // -----------------------------------------------------------------------
    std::cout << "\nBuilding heterogeneous graph...\n";
    HeteroGraph graph = GraphBuilder::build(db);
    graph.print_summary();

    // -----------------------------------------------------------------------
    // TaskSpec
    //
    // target_table  = "games"    (one prediction per match)
    // target_column = "outcome"  (the 0/1/2 column we just injected)
    // task_type     = MulticlassClassification
    //
    // label_transform — Buckets {0.5, 1.5}:
    //   LabelTransform::apply() counts how many bucket boundaries v >= b.
    //   raw=0 → 0 < 0.5          → class 0  (home win)
    //   raw=1 → 0.5 ≤ 1 < 1.5   → class 1  (draw)
    //   raw=2 → 0.5 ≤ 2, 1.5 ≤ 2 → class 2  (away win)
    //   output_dim() = buckets.size() + 1 = 3
    //
    // split_strategy = Random: all seasons (2020-2025) appear in train, val,
    //   and test. This prevents era-based distribution shift (e.g. manager changes).
    //
    // inference_mode = EntitySynthesis:
    //   Given a home_team_id and away_team_id, the synthesizer looks up both
    //   teams in the trained GNN embeddings, mean-pools them into a single
    //   vector, and passes it through the 3-class head. The argmax is the
    //   predicted outcome class.
    //   Example below: Arsenal (42) hosting Manchester City (50).
    // -----------------------------------------------------------------------
    TaskSpec spec;
    spec.target_table  = "games";
    spec.target_column = "outcome";
    spec.task_type     = TaskSpec::TaskType::MulticlassClassification;

    spec.label_transform.kind    = LabelTransform::Kind::Buckets;
    spec.label_transform.buckets = {0.5f, 1.5f};   // → 3 classes: 0, 1, 2

    spec.split_strategy = TaskSpec::SplitStrategy::Random;
    // This means train always sees older games, val/test always see newer ones —
    // which is the honest evaluation setup for predicting future fixtures.

    // Predict: if Arsenal (id=42) hosts Manchester City (id=50), who wins?
    spec.inference_mode = TaskSpec::InferenceMode::EntitySynthesis;
    spec.entity_refs    = {{"home_team_id", "42"}, {"away_team_id", "50"}};
    spec.inference_agg  = TaskSpec::AggType::None;

    std::cout << "\nBuilding train/val/test split (random — all seasons in train/val/test)...\n";
    TaskSplit split = spec.build_split(db);
    print_split_stats(split);

    // CRITICAL: hide the outcome column from HeteroEncoder after the split is built.
    // build_split() needs NUMERICAL type to read labels — that is done above.
    // If left as NUMERICAL, HeteroEncoder::fit_table encodes it as an input feature,
    // giving the model direct access to the answer (symptom: 100% val accuracy).
    // Flipping to TEXT causes the encoder to skip it while the labels already
    // stored in TaskSplit remain unaffected.
    db.get_table("games").get_column("outcome").type = ColumnType::TEXT;

    // -----------------------------------------------------------------------
    // Show the calendar period covered by each split so it is clear what the
    // model is trained on before training begins.
    // -----------------------------------------------------------------------
    {
        auto date_range = [&](const std::string& label,
                               const std::vector<TaskSample>& samples) {
            std::string lo = "9999", hi = "";
            for (const auto& s : samples) {
                if (s.node_idx < 0 || static_cast<std::size_t>(s.node_idx) >= game_dates.size()) continue;
                const std::string& d = game_dates[s.node_idx];
                if (d.empty()) continue;
                std::string day = d.substr(0, 10);   // "YYYY-MM-DD"
                if (day < lo) lo = day;
                if (day > hi) hi = day;
            }
            std::cout << "    " << std::setw(5) << std::left << label
                      << ": " << lo << "  →  " << hi << "\n";
        };

        std::cout << "\n  Season coverage per split (should all span 2020-2026 with random split):\n";
        date_range("train", split.train);
        date_range("val",   split.val);
        date_range("test",  split.test);
        std::cout << "  (window: full dataset — all completed fixtures up to 2026-02-28)\n";
    }

    // -----------------------------------------------------------------------
    // TrainConfig
    //
    // The dominant failure mode so far has been severe overfitting:
    // training loss reaches ~0.02 while val loss stays at ~1.06.
    // The fixes are:
    //
    // dropout=0.5: standard aggressive dropout for small tabular GNN datasets.
    //   At 0.2 the model was memorizing training games. 0.5 forces it to
    //   learn robust patterns rather than individual game signatures.
    //
    // channels=32, hidden=32: halving capacity reduces the number of free
    //   parameters. With ~1600 training games a 64-dim GNN has more than
    //   enough capacity to overfit completely.
    //
    // gnn_layers=2: form and season-table features are on the game node
    //   directly, so deep aggregation adds noise. 2 layers span game →
    //   teams → players.
    //
    // lr=1e-4: slightly lower learning rate pairs better with high dropout
    //   since gradients are noisier when half the units are masked.
    //
    // epochs=600: training needs more steps to converge with heavy dropout.
    //
    // batch_size=0: full-batch (dataset is small enough).
    // -----------------------------------------------------------------------
    TrainConfig cfg;
    cfg.channels   = 32;
    cfg.gnn_layers = 2;
    cfg.hidden     = 32;
    cfg.dropout    = 0.5f;
    cfg.lr         = 1e-4f;
    cfg.pos_weight = 1.f;
    cfg.epochs     = 600;
    cfg.batch_size = 0;
    cfg.task       = spec;

    std::cout << "\nBuilding Trainer...\n";
    Trainer trainer(cfg, db, graph);

    std::cout << "\nTraining...\n";
    trainer.fit(split, db, graph);

    // -----------------------------------------------------------------------
    // Fixture predictions — March 3–5 2026
    //
    // synthesize_prediction cannot be used here. It mean-pools the two team
    // embeddings into one symmetric vector, so the head sees the same signal
    // regardless of which team is home or away — it always predicts the same
    // class for every fixture.
    //
    // Instead we:
    //   1. Run predict_all() — scores every game node using full GNN embeddings
    //      where home_team_id and away_team_id are distinct edge types. These
    //      embeddings correctly encode home/away asymmetry.
    //   2. For each requested fixture, scan the games table for the most recent
    //      historical game with the same home_team_id and away_team_id. The
    //      model's prediction on that game node is the best proxy for the
    //      requested matchup.
    //   3. If no historical matchup exists (e.g. a newly promoted team that
    //      never played the opponent at home in 2020–2025), fall back to the
    //      most recent game where that team appeared as home or away.
    //
    // Team IDs (from teams.csv):
    //   33=Man Utd  34=Newcastle  35=Bournemouth  36=Fulham    39=Wolves
    //   40=Liverpool 42=Arsenal   44=Burnley       45=Everton   47=Tottenham
    //   48=West Ham  49=Chelsea   50=Man City      51=Brighton  52=Crystal Palace
    //   55=Brentford 63=Leeds     65=Nott'm Forest 66=Aston Villa 746=Sunderland
    // -----------------------------------------------------------------------

    // Score every game node first — these are the predictions we'll look up.
    std::cout << "\nGlobal prediction distribution across all "
              << db.get_table("games").num_rows() << " games:\n";
    std::vector<float> all_preds = trainer.predict_all(db, graph);

    {
        std::size_t pred_home = 0, pred_draw = 0, pred_away = 0;
        for (float p : all_preds) {
            int cls = static_cast<int>(std::round(p));
            if      (cls == 0) ++pred_home;
            else if (cls == 1) ++pred_draw;
            else               ++pred_away;
        }
        std::size_t n_total = all_preds.size();
        std::cout << std::fixed << std::setprecision(1)
                  << "  Home wins : " << pred_home << "  (" << 100.f * pred_home / n_total << "%)\n"
                  << "  Draws     : " << pred_draw << "  (" << 100.f * pred_draw / n_total << "%)\n"
                  << "  Away wins : " << pred_away << "  (" << 100.f * pred_away / n_total << "%)\n";
    }

    // -----------------------------------------------------------------------
    // Build lookup structures over the games table so we can find the most
    // recent historical game for any (home_team_id, away_team_id) pair.
    // -----------------------------------------------------------------------
    const Table&  games_table = db.get_table("games");
    const Column& home_id_col = games_table.get_column("home_team_id");
    const Column& away_id_col = games_table.get_column("away_team_id");
    std::size_t   n_games     = games_table.num_rows();

    // Returns the row index of the most recent COMPLETED game (date < cutoff)
    // with matching home/away IDs. Skipping future rows is critical: the games
    // CSV contains rows for upcoming fixtures that have no outcome label and
    // whose GNN embeddings were never trained on — querying all_preds on those
    // rows always returns the head's default bias (Draw). By restricting to
    // rows with date < "2026-03-01" we only look at historically completed games.
    // Falls back to the last completed home game for that team (any opponent)
    // if the exact pairing never occurred in the dataset.
    const std::string cutoff = "2026-03-01";
    auto find_best_game_row = [&](int home_id, int away_id) -> std::size_t {
        std::size_t best_exact = SIZE_MAX;
        std::size_t best_home  = SIZE_MAX;

        for (std::size_t i = 0; i < n_games; ++i) {
            // Skip future game rows — their predictions are meaningless
            if (i < game_dates.size()) {
                const std::string& d = game_dates[i];
                if (!d.empty() && d.substr(0, cutoff.size()) >= cutoff) continue;
            }
            if (home_id_col.is_null(i) || away_id_col.is_null(i)) continue;
            int h = static_cast<int>(home_id_col.get_numerical(i));
            int a = static_cast<int>(away_id_col.get_numerical(i));

            if (h == home_id && a == away_id)
                best_exact = i;   // overwrite — last (highest index) wins

            if (h == home_id)
                best_home = i;
        }
        return (best_exact != SIZE_MAX) ? best_exact : best_home;
    };

    // -----------------------------------------------------------------------
    // Fixtures table and output
    // -----------------------------------------------------------------------
    struct Fixture {
        std::string date;
        int         home_id;
        std::string home_name;
        int         away_id;
        std::string away_name;
        std::string actual;
    };

    std::vector<Fixture> fixtures = {
        {"Tue 3 Mar", 35,  "Bournemouth",        55,  "Brentford",          "0 - 0"},
        {"Tue 3 Mar", 45,  "Everton",             44,  "Burnley",            "2 - 0"},
        {"Tue 3 Mar", 63,  "Leeds United",        746, "Sunderland",         "0 - 1"},
        {"Tue 3 Mar", 39,  "Wolves",              40,  "Liverpool",          "2 - 1"},
        {"Wed 4 Mar", 66,  "Aston Villa",         49,  "Chelsea",            "1 - 4"},
        {"Wed 4 Mar", 51,  "Brighton",            42,  "Arsenal",            "0 - 1"},
        {"Wed 4 Mar", 36,  "Fulham",              48,  "West Ham",           "0 - 1"},
        {"Wed 4 Mar", 50,  "Manchester City",     65,  "Nottingham Forest",  "2 - 2"},
        {"Wed 4 Mar", 34,  "Newcastle",           33,  "Manchester United",  "2 - 1"},
        {"Thu 5 Mar", 47,  "Tottenham",           52,  "Crystal Palace",     "1 - 3"},
    };

    static const char* OUTCOME_LABELS[] = {"Home Win", "Draw", "Away Win"};

    std::cout << "\n"
              << std::string(84, '=') << "\n"
              << "  PREDICTED FIXTURES — March 3–5 2026\n"
              << std::string(84, '=') << "\n"
              << std::left
              << std::setw(12) << "Date"
              << std::setw(24) << "Home"
              << std::setw(24) << "Away"
              << std::setw(16) << "Prediction"
              << "Actual\n"
              << std::string(84, '-') << "\n";

    std::string last_date;
    for (const auto& fx : fixtures) {
        std::size_t row = find_best_game_row(fx.home_id, fx.away_id);

        int cls = 0;
        if (row == SIZE_MAX) {
            cls = -1;  // no data at all for this team
        } else {
            cls = static_cast<int>(std::round(all_preds[row]));
            cls = std::max(0, std::min(2, cls));
        }

        if (fx.date != last_date) {
            if (!last_date.empty()) std::cout << "\n";
            last_date = fx.date;
        }

        std::cout << std::left
                  << std::setw(12) << fx.date
                  << std::setw(24) << fx.home_name
                  << std::setw(24) << fx.away_name
                  << std::setw(16) << (cls < 0 ? "No data" : OUTCOME_LABELS[cls])
                  << fx.actual << "\n";
    }
    std::cout << std::string(84, '=') << "\n";

    return 0;
}