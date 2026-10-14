// Record generated-answer accuracy alongside strict add/cmp trace checks.
#include <filesystem>
#include <fstream>
#include <map>
#include "../src/args.h"
#include "../src/generate.h"
#include "../src/reasoning_traces.h"
#include "../src/train.h"

struct Cell {
    size_t problems = 0, exact = 0, full_trace = 0, exact_trace = 0;
    size_t expected = 0, generated = 0, local = 0, grounded = 0;
    void add(const Problem& p, const std::string& text) {
        const auto s = reasoning_traces::score(p, text);
        const bool ok = completion_done(text) && answer_matches(text, p.answer);
        ++problems; exact += ok; full_trace += s.all_grounded(); exact_trace += ok && s.all_grounded();
        expected += s.expected; generated += s.generated; local += s.local_correct; grounded += s.grounded_correct;
    }
};
int main(int argc, char** argv) {
    Args a(argc, argv, 1);
    const auto ckpt = a.str("ckpt"), path = a.str("problems"), transcript = a.str("transcripts");
    const int threads = a.integer_int("threads", 2), batch = a.integer_int("batch", 64);
    a.done();
    CHECK(!ckpt.empty() && !path.empty() && threads > 0 && batch > 0, "invalid trace audit arguments");
    ThreadPool pool_(threads); g_pool = &pool_;
    Model<float> model(peek_checkpoint_config(ckpt));
    CHECK(model.cfg.vocab == 256, "trace audit requires a byte-vocabulary checkpoint");
    load_checkpoint(ckpt, model, nullptr, nullptr);
    const auto problems = load_problems(path);
    const auto result = eval_reasoning(model, problems, Tokenizer{}, batch, 200, 0, true);
    CHECK(result.completions.size() == problems.size(), "missing generated completions");
    std::map<std::pair<std::string, int>, Cell> cells;
    Cell total;
    std::ofstream output;
    if (!transcript.empty()) {
        output.open(transcript);
        CHECK(output, "cannot open transcript output");
        output << "task\tdifficulty\tquestion\tcompletion_hex\n";
    }
    constexpr char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < problems.size(); ++i) {
        const auto& p = problems[i]; const auto& text = result.completions[i];
        cells[{p.task, p.difficulty}].add(p, text); total.add(p, text);
        if (output.is_open()) {
            output << p.task << '\t' << p.difficulty << '\t' << p.question << '\t';
            for (unsigned char c : text) output << hex[c >> 4] << hex[c & 15];
            output << '\n';
        }
    }
    CHECK(total.exact == size_t(result.correct), "trace audit disagrees with production answer metric");
    if (output.is_open()) { output.close(); CHECK(output, "transcript write failed"); }
    const auto id = std::filesystem::path(ckpt).stem().string();
    std::printf("checkpoint,task,difficulty,problems,exact,full_grounded_trace,exact_and_grounded,expected_statements,generated_statements,local_correct,grounded_correct\n");
    auto print = [&](const std::string& task, int difficulty, const Cell& c) {
        std::printf("%s,%s,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n", id.c_str(), task.c_str(), difficulty,
                    c.problems, c.exact, c.full_trace, c.exact_trace, c.expected, c.generated, c.local, c.grounded);
    };
    for (const auto& [key, cell] : cells) print(key.first, key.second, cell);
    print("overall", 0, total);
}
