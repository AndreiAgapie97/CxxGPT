// Preserve byte-token corpora and create final-answer-digit supervision plus
// a fresh question holdout, excluding train, validation and the old test set.
#include <filesystem>
#include <set>
#include "../src/args.h"
#include "../src/reasoning.h"
#include "../src/train.h"

TokenData final_digit_mask(const TokenData& original, std::set<std::string>& seen) {
    CHECK(original.vocab == 256 && !original.tokens.empty() && original.tokens[0] == kBos,
          "requires a native byte reasoning corpus starting with BOS");
    TokenData out = original;
    out.mask.assign(out.tokens.size(), 0);
    for (size_t bos = 0; bos < original.tokens.size();) {
        size_t end = bos + 1;
        while (end < original.tokens.size() && original.tokens[end] != kBos) ++end;
        std::string doc;
        for (size_t i = bos + 1; i < end; ++i) doc += char(original.tokens[i]);
        const size_t prompt = doc.find("\nA: "), mark = doc.rfind("\n#### ");
        CHECK(doc.starts_with("Q: ") && prompt != std::string::npos && mark > prompt &&
              mark != std::string::npos, "malformed reasoning document");
        seen.insert(doc.substr(3, prompt - 3));
        const size_t answer = mark + 6, stop = doc.find('\n', answer);
        CHECK(stop != std::string::npos, "missing final-answer newline");
        size_t valid = 0;
        for (size_t i = answer; i < stop; ++i) {
            const size_t pos = bos + 1 + i;
            const bool digit = doc[i] >= '0' && doc[i] <= '9';
            out.mask[pos] = digit && (original.mask.empty() || original.mask[pos]);
            valid += out.mask[pos] != 0;
        }
        CHECK(valid > 0, "reasoning document has no supervised final digits");
        bos = end;
    }
    return out;
}
int main(int argc, char** argv) {
    Args args(argc, argv, 1);
    const auto train = args.str("train"), val = args.str("val"), old_test = args.str("exclude-test");
    const auto dir = args.str("out");
    const int per_cell = int(args.integer("per-cell", 100));
    const uint64_t seed = uint64_t(args.integer("seed", 20261014));
    args.done();
    CHECK(!train.empty() && !val.empty() && !old_test.empty() && !dir.empty() && per_cell > 0,
          "provide train, val, exclude-test, a new out directory and positive per-cell");
    const auto a = TokenData::load(train), b = TokenData::load(val);
    std::set<std::string> seen;
    const auto fa = final_digit_mask(a, seen), fb = final_digit_mask(b, seen);
    for (const auto& p : load_problems(old_test)) seen.insert(p.question);
    Rng rng(seed);
    std::vector<Problem> probe;
    for (const std::string task : {"add", "cmp"})
        for (int difficulty = 1; difficulty <= 3; ++difficulty) {
            int accepted = 0;
            for (int tries = 0; accepted < per_cell && tries < 1000000; ++tries) {
                auto p = make_problem(task, difficulty, rng);
                if (!seen.insert(p.question).second) continue;
                probe.push_back(std::move(p)); ++accepted;
            }
            CHECK(accepted == per_cell, "holdout question domain exhausted");
        }
    CHECK(std::filesystem::create_directory(dir), "out directory already exists or cannot be created");
    fa.save(dir + "/train-final.tok"); fb.save(dir + "/val-final.tok");
    save_problems(dir + "/probe.txt", probe);
    std::printf("tokens preserved: %zu train, %zu val; final digits: %zu train, %zu val; fresh questions: %zu\n",
                a.tokens.size(), b.tokens.size(),
                size_t(std::count(fa.mask.begin(), fa.mask.end(), uint8_t(1))),
                size_t(std::count(fb.mask.begin(), fb.mask.end(), uint8_t(1))), probe.size());
}
