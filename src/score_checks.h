// score_checks.h — how well self-train's checks tell a clean solution from
// a lucky or padded one (cxxgpt score-checks).
//
// A labelled file (tests/checker_labels.jsonl, written by
// operations/build_checker_labels.py) holds tries with the right answer, each
// labelled by hand, blind to the checks: clean (every statement true, every
// calculation right and needed), padded (a sound method plus something not to
// train on; "minor" when it is one wrong word or unit), or lucky (the answer
// does not follow from a sound method). The scorer runs self_train::check_try
// on each try and counts, per set, half (tune or held-out) and verdict, how
// many of each label it gave. Two numbers matter (owner's rule, 2026-10-17:
// train only on clean solutions):
//   - of the tries kept, the share that is clean (the checks' precision);
//   - clean tries that a check dropped (other than as the problem file's own
//     solution, gold_copy, which drops a solution for being known, not for
//     being wrong).
// Rules are tried on the tune half; the held-out half gives the number to
// believe.
#pragma once

#include <array>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "json.h"
#include "self_train.h"

namespace score_checks {

enum Label { clean, minor, padded, lucky };
inline constexpr int kLabels = 4;
inline constexpr const char* kLabelNames[kLabels] = {"clean", "minor", "padded", "lucky"};

struct Labelled {
    std::string set, split;  // "textbook" or "gsm8k-dev"; "tune" or "held-out"
    Problem problem;         // as self-train saw it
    std::string text;        // the try
    Label label = clean;     // padded with "minor": true is minor
};

inline std::vector<Labelled> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f, "cannot open %s", path.c_str());
    std::vector<Labelled> out;
    std::string line;
    for (long no = 1; std::getline(f, line); ++no) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        Json j;
        std::string err;
        CHECK(Json::parse(line, j, &err), "%s:%ld: %s", path.c_str(), no, err.c_str());
        auto str = [&](const char* key) {
            const Json* v = j.get(key);
            CHECK(v && v->type == Json::String, "%s:%ld: no \"%s\" text", path.c_str(), no, key);
            return v->str;
        };
        Labelled l;
        l.set = str("set");
        l.split = str("split");
        CHECK(l.split == "tune" || l.split == "held-out", "%s:%ld: split must be tune or held-out", path.c_str(), no);
        l.problem.question = str("question");
        l.problem.steps = str("steps");
        l.problem.answer = str("answer");
        l.text = str("text");
        const std::string label = str("label");
        const Json* m = j.get("minor");
        const bool is_minor = m && m->type == Json::Bool && m->boolean;
        if (label == "clean") l.label = clean;
        else if (label == "padded") l.label = is_minor ? minor : padded;
        else if (label == "lucky") l.label = lucky;
        else CHECK(false, "%s:%ld: label must be clean, padded or lucky", path.c_str(), no);
        CHECK(!is_minor || label == "padded", "%s:%ld: only a padded try can be minor", path.c_str(), no);
        out.push_back(std::move(l));
    }
    CHECK(!out.empty(), "%s has no labelled tries", path.c_str());
    return out;
}

// Per verdict, how many tries of each label it got.
using Counts = std::map<self_train::Verdict, std::array<long, kLabels>>;

struct Score {
    // Per (set, split), in file order of first appearance.
    std::vector<std::pair<std::string, std::string>> groups;
    std::vector<Counts> counts;

    long kept(size_t g, Label l) const {
        const auto it = counts[g].find(self_train::Verdict::kept);
        return it == counts[g].end() ? 0 : it->second[l];
    }
    long kept_total(size_t g) const {
        long n = 0;
        for (int l = 0; l < kLabels; ++l) n += kept(g, Label(l));
        return n;
    }
    // Clean tries a check dropped, gold_copy aside.
    long clean_dropped(size_t g) const {
        long n = 0;
        for (const auto& [v, c] : counts[g])
            if (v != self_train::Verdict::kept && v != self_train::Verdict::gold_copy) n += c[clean];
        return n;
    }
};

inline Score score(const std::vector<Labelled>& tries, const self_train::Options& o) {
    Score s;
    for (const Labelled& t : tries) {
        size_t g = 0;
        while (g < s.groups.size() && s.groups[g] != std::pair(t.set, t.split)) ++g;
        if (g == s.groups.size()) s.groups.emplace_back(t.set, t.split), s.counts.emplace_back();
        const self_train::Verdict v = self_train::check_try(t.problem, t.text, o);
        CHECK(v != self_train::Verdict::wrong && v != self_train::Verdict::unfinished,
              "a labelled try must have a finished, right answer: %s", t.problem.question.c_str());
        s.counts[g][v][t.label]++;
    }
    return s;
}

}  // namespace score_checks
