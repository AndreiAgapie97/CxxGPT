// Fresh paired add/cmp curricula with explicit question exclusions. Training
// and model execution remain native; this also replays the exact data budget.
#include <filesystem>
#include <fstream>
#include <set>
#include "../src/args.h"
#include "../src/reasoning.h"
#include "../src/train.h"

std::vector<std::string> files(const std::string& list) {
    std::vector<std::string> out;
    for (size_t at = 0; at < list.size();) {
        const auto end = list.find(',', at); out.push_back(list.substr(at, end - at));
        if (end == list.npos) break;
        at = end + 1;
    }
    return out;
}
void exclude_corpus(const TokenData& data, std::set<std::string>& seen) {
    CHECK(data.vocab == 256 && !data.tokens.empty() && data.tokens[0] == kBos,
          "requires byte reasoning corpora starting with BOS");
    for (size_t bos = 0; bos < data.tokens.size();) {
        size_t end = bos + 1; std::string doc;
        while (end < data.tokens.size() && data.tokens[end] != kBos) {
            CHECK(data.tokens[end] < 256, "corpus contains a non-byte token");
            doc += char(data.tokens[end++]);
        }
        const size_t prompt = doc.find("\nA: ");
        CHECK(doc.starts_with("Q: ") && prompt != doc.npos, "malformed prior reasoning corpus");
        seen.insert(doc.substr(3, prompt - 3)); bos = end;
    }
}
void save_tokens(const std::string& path, const std::vector<Problem>& problems) {
    TokenData data; data.vocab = 256;
    for (const auto& p : problems) {
        const std::string prompt = std::string(1, char(kBos)) + p.prompt(), text = prompt + p.completion();
        for (size_t i = 0; i < text.size(); ++i) {
            data.tokens.push_back(uint8_t(text[i])); data.mask.push_back(i >= prompt.size());
        }
    }
    data.save(path);
}
int main(int argc, char** argv) {
    Args a(argc, argv, 1);
    const std::string mode = a.str("mode", "prepare");
    if (mode == "budget") {
        const auto path = a.str("data"), ckpt = a.str("ckpt"), log_path = a.str("log");
        const int64_t steps = a.integer("steps", 1000), seed = a.integer("seed", 101);
        const int batch = a.integer_int("batch", 2), seq = a.integer_int("seq", 128);
        a.done(); CHECK(steps > 0 && batch > 0 && seq > 0 && batch <= INT32_MAX / seq, "invalid budget");
        const auto data = TokenData::load(path); Rng rng(uint64_t(seed) + 1);
        size_t hi = data.tokens.size();
        if (!ckpt.empty()) {
            CHECK(!log_path.empty(), "checkpoint budget validation requires --log runtime sampling evidence");
            std::ifstream log(log_path); CHECK(log, "cannot read sampling log");
            std::string source; bool shape = false;
            for (std::string line; std::getline(log,line);) {
                if (line.starts_with("data sources: ")) source = line.substr(14);
                size_t train=0,val=0,tokens=0; int b=0,t=0,accum=0;
                if (std::sscanf(line.c_str(), "data: %zu train tokens, %zu val tokens; %d x %d x %d = %zu tokens per step",
                                &train,&val,&b,&t,&accum,&tokens)==6) {
                    CHECK(b==batch && t==seq && accum==1 && tokens==size_t(batch)*seq,
                          "runtime sampling shape differs from replay (accum must be 1)");
                    CHECK(train<=data.tokens.size(), "runtime training range exceeds replay data");
                    hi=train; shape=true;
                }
            }
            CHECK(!log.bad() && shape && source==path, "missing or mismatched runtime sampling source/range evidence");
        }
        std::vector<int> inputs(size_t(batch) * seq), targets(inputs.size());
        size_t valid = 0, minimum = inputs.size();
        for (int64_t i = 0; i < steps; ++i) {
            data.sample_batch(rng, batch, seq, 0, hi, inputs.data(), targets.data());
            const size_t n = count_targets(targets.data(), targets.size());
            valid += n; minimum = std::min(minimum, n);
        }
        CHECK(minimum > 0, "empty sampled group changes the successful-update budget");
        if (!ckpt.empty()) {
            Model<float> m(peek_checkpoint_config(ckpt)); Rng actual(0);
            load_checkpoint(ckpt, m, nullptr, &actual);
            FILE* f=std::fopen(ckpt.c_str(),"rb"); CHECK(f, "cannot read checkpoint step");
            CHECK(std::fseek(f,8,SEEK_SET)==0, "cannot seek checkpoint config");
            read_config(f); int64_t saved=0;
            CHECK(std::fread(&saved,8,1,f)==1 && saved==steps, "checkpoint successful-update budget mismatch");
            std::fclose(f);
            for (int i = 0; i < 4; ++i) CHECK(actual.state()[i] == rng.state()[i], "checkpoint sampler budget mismatch");
        }
        std::printf("steps,sampled_tokens,valid_targets,min_valid_per_step\n%lld,%zu,%zu,%zu\n",
                    (long long)steps,size_t(steps) * inputs.size(), valid, minimum);
        return 0;
    }
    CHECK(mode == "prepare", "unknown curriculum mode");
    const auto dir = a.str("out"), corpora = a.str("exclude-corpora"), old = a.str("exclude-problems");
    const int train = a.integer_int("train", 2400), per_cell = a.integer_int("per-cell", 100);
    const int val_cell = a.integer_int("val-per-cell", 60);
    Rng rng(uint64_t(a.integer("seed", 20261017)));
    a.done(); CHECK(!dir.empty() && !corpora.empty() && !old.empty() && train > 0 && train % 12 == 0 &&
                    per_cell > 0 && val_cell > 0, "invalid curriculum arguments");
    std::set<std::string> seen;
    for (const auto& file : files(corpora)) exclude_corpus(TokenData::load(file), seen);
    for (const auto& file : files(old)) for (const auto& p : load_problems(file)) seen.insert(p.question);
    const auto prior = seen;
    auto fresh = [&](const std::string& task, int difficulty) {
        for (int tries = 0; tries < 1000000; ++tries) {
            auto p = make_problem(task, difficulty, rng);
            if (seen.insert(p.question).second) return p;
        }
        CHECK(false, "question domain exhausted"); return Problem{};
    };
    std::vector<Problem> probe, validation, control, expanded;
    for (const std::string task : {"add", "cmp"}) {
        for (int d = 1; d <= 4; ++d)
            for (int i = 0; i < per_cell; ++i) probe.push_back(fresh(task, d));
        for (int d = 1; d <= 3; ++d)
            for (int i = 0; i < val_cell; ++i) validation.push_back(fresh(task, d));
    }
    for (int i = 0; i < train; ++i) {
        const std::string task = i % 2 == 0 ? "add" : "cmp";
        if (i % 3 != 2) {
            const auto p = fresh(task, i % 3 + 1); control.push_back(p); expanded.push_back(p);
        } else {
            control.push_back(fresh(task, (i / 6) % 2 + 1)); expanded.push_back(fresh(task, 3));
        }
    }
    CHECK(std::filesystem::create_directory(dir), "output directory exists or cannot be created");
    std::ofstream exclusion(dir + "/excluded-questions.txt");
    for (const auto& q : prior) exclusion << q << '\n';
    exclusion.close(); CHECK(exclusion, "exclusion archive failed");
    for (const auto& [name, problems] : std::vector<std::pair<std::string, std::vector<Problem>>>{
             {"control", control}, {"expanded", expanded}, {"validation", validation}}) {
        save_problems(dir + "/" + name + ".txt", problems); save_tokens(dir + "/" + name + ".tok", problems);
    }
    save_problems(dir + "/probe.txt", probe);
    std::printf("prior excluded=%zu; final=%zu; validation=%zu; control=%zu; expanded=%zu; shared=%d\n",
                prior.size(), probe.size(), validation.size(), control.size(), expanded.size(), train * 2 / 3);
}
