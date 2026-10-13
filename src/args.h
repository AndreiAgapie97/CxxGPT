// args.h — a tiny "--key value" command-line parser that rejects typos.
#pragma once

#include <cstdlib>
#include <map>
#include <set>
#include <string>

#include "common.h"

class Args {
public:
    // argv[first..] as --key value pairs; a --key followed by another --key
    // (or nothing) is a flag with value "1".
    Args(int argc, char** argv, int first) {
        for (int i = first; i < argc; ++i) {
            std::string k = argv[i];
            CHECK(k.size() > 2 && k.rfind("--", 0) == 0, "expected --option, got '%s'", argv[i]);
            k = k.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv_[k] = argv[++i];
            else kv_[k] = "1";
        }
    }

    bool has(const std::string& k) {
        used_.insert(k);
        return kv_.count(k) > 0;
    }
    std::string str(const std::string& k, const std::string& def = "") {
        return has(k) ? kv_[k] : def;
    }
    long integer(const std::string& k, long def) {
        if (!has(k)) return def;
        char* end = nullptr;
        const long v = std::strtol(kv_[k].c_str(), &end, 10);
        CHECK(end && *end == 0, "--%s expects an integer, got '%s'", k.c_str(), kv_[k].c_str());
        return v;
    }
    double real(const std::string& k, double def) {
        if (!has(k)) return def;
        char* end = nullptr;
        const double v = std::strtod(kv_[k].c_str(), &end);
        CHECK(end && *end == 0, "--%s expects a number, got '%s'", k.c_str(), kv_[k].c_str());
        return v;
    }
    bool flag(const std::string& k) { return has(k) && kv_[k] != "0"; }

    // Call after reading every option: any option nobody asked for is a typo.
    void done() const {
        for (const auto& [k, v] : kv_) CHECK(used_.count(k), "unknown option --%s", k.c_str());
    }

private:
    std::map<std::string, std::string> kv_;
    std::set<std::string> used_;
};
