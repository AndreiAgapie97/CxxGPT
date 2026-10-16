// The control file of `cxxgpt train` (--control FILE, default OUT.control):
// a way to steer a run while it trains, without stopping it. The trainer
// looks at the file before every step and acts when it has changed:
//
//   lr-scale 0.5   the learning rate is the schedule's times 0.5 from now on
//                  (a setting: it holds while the line is in the file, also
//                  after a restart; remove the line to go back to 1)
//   save           save a checkpoint now
//   eval           measure the validation loss now
//   reason         run the reasoning exam now (needs --reason-test)
//   stop           save and stop, like Ctrl-C
//
// save, eval, reason and stop run once each time the file changes (write it
// again to repeat one), and never from a file that was already there when
// training started. Blank lines and lines starting with # are skipped.
#pragma once

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct ControlCommands {
    double lr_scale = 1;  // multiplies the schedule's learning rate
    bool save = false, eval = false, reason = false, stop = false;
    std::vector<std::string> errors;  // lines that are not commands, ignored
};

inline ControlCommands parse_control(std::string text) {
    // Windows PowerShell 5's `>` writes UTF-16 with a byte-order mark; the
    // commands are ASCII, so keeping every other byte reads them.
    if (text.size() >= 2 && (unsigned char)text[0] == 0xFF && (unsigned char)text[1] == 0xFE) {
        std::string ascii;
        for (size_t i = 2; i < text.size(); i += 2) ascii += text[i];
        text = ascii;
    }
    if (text.compare(0, 3, "\xEF\xBB\xBF") == 0) text.erase(0, 3);  // UTF-8 byte-order mark
    ControlCommands c;
    std::istringstream lines(text);
    std::string line;
    for (int n = 1; std::getline(lines, line); ++n) {
        std::istringstream words(line);
        std::string command, value, extra;
        words >> command;
        if (command.empty() || command[0] == '#') continue;
        words >> value >> extra;
        const auto error = [&](const std::string& why) {
            c.errors.push_back("line " + std::to_string(n) + " (" + command + "): " + why);
        };
        if (!extra.empty()) {
            error("too many words");
        } else if (command == "lr-scale") {
            char* end = nullptr;
            const double x = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !std::isfinite(x) || x <= 0)
                error("needs a positive number, as in lr-scale 0.5");
            else
                c.lr_scale = x;
        } else if (!value.empty()) {
            error("takes no value");
        } else if (command == "save") {
            c.save = true;
        } else if (command == "eval") {
            c.eval = true;
        } else if (command == "reason") {
            c.reason = true;
        } else if (command == "stop") {
            c.stop = true;
        } else {
            error("unknown command (lr-scale X, save, eval, reason, stop)");
        }
    }
    return c;
}

class ControlFile {
  public:
    explicit ControlFile(std::string path) : path_(std::move(path)) {}
    const std::string& path() const { return path_; }

    // The file's commands when it changed since the last look (its time or
    // its text); a deleted file reads as empty, so lr-scale goes back to 1.
    // A file that cannot be read now (an editor holding it on Windows) is
    // tried again next time.
    std::optional<ControlCommands> poll() {
        std::error_code ec;
        std::optional<State> now;
        if (std::filesystem::exists(path_, ec)) {
            State s;
            s.time = std::filesystem::last_write_time(path_, ec);
            if (ec) return std::nullopt;
            std::ifstream f(path_, std::ios::binary);
            if (!f) return std::nullopt;
            s.text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            if (f.bad()) return std::nullopt;
            now = std::move(s);
        }
        if (now == last_) return std::nullopt;
        last_ = now;
        return parse_control(now ? now->text : std::string());
    }

  private:
    struct State {
        std::filesystem::file_time_type time;
        std::string text;
        bool operator==(const State&) const = default;
    };
    std::string path_;
    std::optional<State> last_;
};
