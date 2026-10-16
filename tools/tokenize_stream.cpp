// Length-prefixed document encoder for large, provenance-preserving corpora.
// Input: uint64 UTF-8 byte length + bytes. Output: uint64 token count + uint16 ids.
// Each document starts with kBos; CXTOKEN1 headers are written by the caller.
#include <iostream>
#include "../src/tokenizer.h"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    Tokenizer tok;
    tok.load(argv[1]);
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    uint64_t size;
    while (std::cin.read(reinterpret_cast<char*>(&size), sizeof(size))) {
        if (size > (16u << 20)) return 3;
        std::string text(size, '\0');
        if (!std::cin.read(text.data(), std::streamsize(size))) return 4;
        text.erase(std::remove(text.begin(), text.end(), char(kBos)), text.end());
        const auto ids = tok.encode(text);
        std::vector<uint16_t> out;
        out.reserve(ids.size() + 1);
        out.push_back(kBos);
        for (int id : ids) out.push_back(uint16_t(id));
        const uint64_t count = out.size();
        std::cout.write(reinterpret_cast<const char*>(&count), sizeof(count));
        std::cout.write(reinterpret_cast<const char*>(out.data()), std::streamsize(count * 2));
        std::cout.flush();
        if (!std::cout) return 5;
    }
    return std::cin.eof() ? 0 : 6;
}
