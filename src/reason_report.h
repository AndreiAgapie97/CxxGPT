// reason_report.h — optional per-evaluation answers and changes during training.
#pragma once

#include "failure_labels.h"
#include "generate.h"

// One instance per training invocation, with the same problems and ids every
// time. Resuming starts a new comparison baseline; existing files are not read.
struct ReasonReport {
    long previous_step = -1;
    std::vector<bool> previous_correct;

    template <class Encode>
    void write(const std::string& prefix, long step, const std::string& settings,
               const std::vector<Problem>& problems, const std::vector<std::string>& completions,
               const std::vector<size_t>& ids, Encode encode) {
        const std::string path = prefix + ".step-" + std::to_string(step) + ".jsonl";
        const labels::LabelSummary summary = labels::write_labels(path, settings, problems, completions, ids, encode);
        CHECK(previous_step < 0 || previous_correct.size() == problems.size(), "reasoning report sample changed");
        std::vector<bool> correct(problems.size());
        std::string gained, lost;
        size_t n_gained = 0, n_lost = 0;
        for (size_t i = 0; i < problems.size(); ++i) {
            correct[i] = completion_done(completions[i]) && answer_matches(completions[i], problems[i].answer);
            if (previous_step < 0 || correct[i] == previous_correct[i]) continue;
            std::string& list = correct[i] ? gained : lost;
            list += (list.empty() ? "" : ",") + std::to_string(ids[i]);
            ++(correct[i] ? n_gained : n_lost);
        }
        std::string out = "{\"type\":\"summary\",\"previous_step\":" +
                          (previous_step < 0 ? std::string("null") : std::to_string(previous_step)) +
                          ",\"became_correct\":[" + gained + "],\"regressed\":[" + lost + "],\"labels\":{";
        for (size_t i = 0; i < std::size(labels::kAllLabels); ++i) {
            const auto label = labels::kAllLabels[i];
            out += std::string(i ? "," : "") + "\"" + labels::label_name(label) + "\":" +
                   std::to_string(summary.count[size_t(label)]);
        }
        out += "},\"loops_unfinished\":" + std::to_string(summary.loops_unfinished) +
               ",\"runs_unfinished\":" + std::to_string(summary.runs_unfinished) + "}\n";
        FILE* f = create_file(path, "ab");
        CHECK(f, "cannot append %s", path.c_str());
        const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
        CHECK(std::fclose(f) == 0 && ok, "cannot append %s", path.c_str());
        summary.print(labels::kRunTokens);
        if (previous_step >= 0)
            std::printf("since reasoning step %ld: %zu became correct, %zu regressed\n", previous_step, n_gained, n_lost);
        std::printf("reasoning report: %s\n", path.c_str());
        previous_step = step;
        previous_correct = std::move(correct);
    }
};
