// loss_mask.h — text annotations for context that must not be predicted.
#pragma once

#include <string_view>
#include <vector>

#include "tokenizer.h"

inline constexpr std::string_view kNoLossBegin = "[[no-loss]]";
inline constexpr std::string_view kNoLossEnd = "[[/no-loss]]";

// Each annotation boundary is also a tokenization boundary. Thus even BPE
// tokens belong wholly to one loss region; no token straddles a wrong step.
inline void encode_loss_marked(std::string_view text, const Tokenizer& tok,
                               std::vector<uint16_t>& tokens, std::vector<uint8_t>& mask) {
    bool ignored = false;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t open = text.find(kNoLossBegin, pos), close = text.find(kNoLossEnd, pos);
        const size_t next = std::min(open, close);
        const size_t end = next == text.npos ? text.size() : next;
        for (int id : tok.encode(text.substr(pos, end - pos))) {
            tokens.push_back(uint16_t(id));
            mask.push_back(ignored ? 0 : 1);
        }
        if (next == text.npos) break;
        CHECK((next == open && !ignored) || (next == close && ignored),
              "nested or unmatched no-loss marker");
        ignored = next == open;
        pos = next + (ignored ? kNoLossBegin.size() : kNoLossEnd.size());
    }
    CHECK(!ignored, "unclosed no-loss marker");
}

// Keep complete annotations in a streaming block. A cut may move forward to
// the closing marker, which is itself a safe tokenization boundary. npos
// means the reader needs more input, subject to its existing buffer limit.
inline size_t loss_marker_safe_split(std::string_view text, size_t end, size_t start = 0) {
    // Search only the block [start, end), plus enough bytes past end to see a
    // marker that starts inside it: with many threads, each block searching
    // the rest of the buffer would read it many times over. Only a span still
    // open at the end makes us look further, for its closing marker.
    bool ignored = false;
    size_t pos = start;
    while (pos < end) {
        const std::string_view block = text.substr(0, std::min(text.size(), end + kNoLossEnd.size() - 1));
        const size_t open = block.find(kNoLossBegin, pos), close = block.find(kNoLossEnd, pos);
        const size_t next = std::min(open, close);
        if (next == block.npos || next >= end) break;
        CHECK((next == open && !ignored) || (next == close && ignored),
              "nested or unmatched no-loss marker");
        ignored = next == open;
        pos = next + (ignored ? kNoLossBegin.size() : kNoLossEnd.size());
        end = std::max(end, pos);  // a cut must not split the marker itself
    }
    if (ignored) {
        const size_t close = text.find(kNoLossEnd, pos);
        if (close == text.npos) return text.npos;
        return close + kNoLossEnd.size();
    }
    return end;
}
