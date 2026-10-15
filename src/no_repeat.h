// no_repeat.h — which tokens would repeat a run of N tokens already written.
// Used at decode time by the no-repeat block (--no-repeat N, generate.h,
// #33) and in training by the unlikelihood term (--unlikelihood, #34), so
// both work against the same definition of a repeat.
#pragma once

#include <algorithm>
#include <map>
#include <vector>

#include "common.h"

// For one sequence: the tokens that would complete a run of N tokens the
// sequence has already written once. Small models greedily repeat a
// sentence until the token budget runs out; the block forces something new
// instead. Only the generated text counts, not the prompt: an answer may
// copy a whole phrase of its question. A run shorter than N may repeat
// ("9 * 2" after "16 - 3 - 4 = 9"), so N must be longer than the numbers and
// phrases a solution reuses on purpose. A BOS starts a new document and
// forgets what came before.
class NoRepeat {
public:
    explicit NoRepeat(int n) : n_(n) { CHECK(n >= 2, "--no-repeat must be 0 (off) or at least 2, not %d", n); }

    void add(int t) {
        if (t == kBos) {
            text_.clear();
            next_.clear();
            return;
        }
        text_.push_back(t);
        if (int(text_.size()) < n_) return;
        // The run of N tokens just completed: its first N - 1 tokens, then t.
        std::vector<int>& after = next_[std::vector<int>(text_.end() - n_, text_.end() - 1)];
        if (std::find(after.begin(), after.end(), t) == after.end()) after.push_back(t);
    }

    // A token that is not part of the text (training: a masked token, such
    // as the question or a planted mistake): no run continues across it, but
    // the runs written before it still count.
    void skip() { text_.clear(); }

    // The tokens that may not come next.
    const std::vector<int>& blocked() const {
        if (int(text_.size()) < n_ - 1) return none_;
        const auto it = next_.find(std::vector<int>(text_.end() - (n_ - 1), text_.end()));
        return it == next_.end() ? none_ : it->second;
    }

private:
    int n_;
    std::vector<int> text_;  // since the BOS or the last skip()
    // Every N - 1 tokens seen so far -> the tokens that followed them. Exact
    // keys, not hashes: a collision would block a token wrongly.
    std::map<std::vector<int>, std::vector<int>> next_;
    std::vector<int> none_;
};
