// Vector payload accounting, separate from allocator/process memory.
#pragma once
#include <cstddef>
#include <initializer_list>
#include <vector>

struct ArrayBytes {
    size_t used = 0, reserved = 0;
    template <class T> void add(const std::vector<T>& v) {
        used += v.size() * sizeof(T);
        reserved += v.capacity() * sizeof(T);
    }
    ArrayBytes& operator+=(const ArrayBytes& b) {
        used += b.used; reserved += b.reserved; return *this;
    }
};

struct ModelMemory {
    ArrayBytes parameters, gradients, quantized, routing, rope;
    ArrayBytes activations, backward, scratch, kv;
    ArrayBytes total() const {
        ArrayBytes b;
        for (const auto& x : {parameters, gradients, quantized, routing, rope,
                              activations, backward, scratch, kv}) b += x;
        return b;
    }
};

struct PackedMemory {
    // weights includes inactive placeholder bitplanes; logical weight_bytes()
    // deliberately counts only active model tensors and their scalar scales.
    ArrayBytes weights, rope, kv, scratch;
    ArrayBytes total() const {
        ArrayBytes b;
        for (const auto& x : {weights, rope, kv, scratch}) b += x;
        return b;
    }
};
