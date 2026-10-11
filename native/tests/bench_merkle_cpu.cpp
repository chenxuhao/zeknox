// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Multicore CPU benchmark for Merkle tree building (Plonky2-compatible layout).
// Leaf hashing and each tree level are parallelized with OpenMP.
//
// Usage:
//   ./bench_merkle_cpu.exe [--hash poseidon|poseidon2|keccak|monolith|poseidon_bn128]
//       [--log-min 13] [--log-max 20] [--leaf-size 135] [--cap-height 0]
//       [--threads 1,2,4,8] [--iters 10] [--warmup 2]
// --threads defaults to omp_get_max_threads() (honours OMP_NUM_THREADS).

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "types/int_types.h"
#include "merkle/merkle.h"

struct Args
{
    u64 hash_type = HashPoseidon;
    const char *hash_name = "poseidon";
    int log_min = 13;
    int log_max = 20;
    u64 leaf_size = 135;
    u64 cap_height = 0;
    std::vector<int> threads;
    int iters = 10;
    int warmup = 2;
};

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--hash poseidon|poseidon2|keccak|monolith|poseidon_bn128] [--log-min N] [--log-max N]\n"
            "          [--leaf-size N] [--cap-height N] [--threads 1,2,4,...] [--iters N] [--warmup N]\n",
            prog);
    exit(1);
}

static Args parse_args(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; i++)
    {
        if (i + 1 >= argc)
            usage(argv[0]);
        const char *flag = argv[i];
        const char *val = argv[++i];
        if (!strcmp(flag, "--hash"))
        {
            a.hash_name = val;
            if (!strcmp(val, "poseidon"))
                a.hash_type = HashPoseidon;
            else if (!strcmp(val, "poseidon2"))
                a.hash_type = HashPoseidon2;
            else if (!strcmp(val, "keccak"))
                a.hash_type = HashKeccak;
            else if (!strcmp(val, "monolith"))
                a.hash_type = HashMonolith;
            else if (!strcmp(val, "poseidon_bn128"))
                a.hash_type = HashPoseidonBN128;
            else
                usage(argv[0]);
        }
        else if (!strcmp(flag, "--log-min"))
            a.log_min = atoi(val);
        else if (!strcmp(flag, "--log-max"))
            a.log_max = atoi(val);
        else if (!strcmp(flag, "--leaf-size"))
            a.leaf_size = strtoull(val, NULL, 10);
        else if (!strcmp(flag, "--cap-height"))
            a.cap_height = strtoull(val, NULL, 10);
        else if (!strcmp(flag, "--iters"))
            a.iters = atoi(val);
        else if (!strcmp(flag, "--warmup"))
            a.warmup = atoi(val);
        else if (!strcmp(flag, "--threads"))
        {
            std::string s(val);
            size_t pos = 0;
            while (pos <= s.size())
            {
                size_t comma = s.find(',', pos);
                if (comma == std::string::npos)
                    comma = s.size();
                a.threads.push_back(atoi(s.substr(pos, comma - pos).c_str()));
                pos = comma + 1;
            }
        }
        else
            usage(argv[0]);
    }
    if (a.threads.empty())
        a.threads.push_back(omp_get_max_threads());
    if (a.log_min < 1 || a.log_min > a.log_max || (int)a.cap_height > a.log_min || a.iters < 1 || a.leaf_size < 1)
        usage(argv[0]);
    for (int t : a.threads)
        if (t < 1)
            usage(argv[0]);
    return a;
}

typedef void (*fill_fn)(void *, void *, const void *, u64, u64, u64, u64, u64, u64);

// Runs warmup + iters builds; returns {median, min} in milliseconds.
static std::pair<double, double> time_ms(fill_fn fn, const Args &a, u64 *digests, u64 *caps, const u64 *leaves,
                                         u64 n_digests, u64 n_caps, u64 n_leaves)
{
    std::vector<double> samples;
    for (int i = 0; i < a.warmup + a.iters; i++)
    {
        auto t0 = std::chrono::steady_clock::now();
        fn(digests, caps, leaves, n_digests, n_caps, n_leaves, a.leaf_size, a.cap_height, a.hash_type);
        auto t1 = std::chrono::steady_clock::now();
        if (i >= a.warmup)
            samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples[0]};
}

int main(int argc, char **argv)
{
    Args a = parse_args(argc, argv);

    struct Variant
    {
        const char *name;
        fill_fn fn;
    };
    std::vector<Variant> variants = {{"scalar", fill_digests_buf_linear_cpu}};
#ifdef __USE_AVX__
    if (a.hash_type == HashPoseidon)
        variants.push_back({"avx2", fill_digests_buf_linear_cpu_avx});
#endif
#ifdef __AVX512__
    if (a.hash_type == HashPoseidon)
        variants.push_back({"avx512", fill_digests_buf_linear_cpu_avx512});
#endif

    printf("CPU Merkle tree: hash=%s leaf_size=%lu cap_height=%lu iters=%d warmup=%d max_threads=%d\n",
           a.hash_name, (unsigned long)a.leaf_size, (unsigned long)a.cap_height, a.iters, a.warmup,
           omp_get_max_threads());
    printf("%6s %10s %8s %8s %12s %12s %14s\n", "log_n", "leaves", "variant", "threads", "median_ms", "min_ms",
           "Kleaves/s");

    std::mt19937_64 rng(42);
    for (int log_n = a.log_min; log_n <= a.log_max; log_n++)
    {
        u64 n_leaves = 1ULL << log_n;
        u64 n_caps = 1ULL << a.cap_height;
        u64 n_digests = 2 * (n_leaves - n_caps);

        // Goldilocks elements must be < p = 2^64 - 2^32 + 1.
        std::vector<u64> leaves(n_leaves * a.leaf_size);
        for (auto &x : leaves)
            x = rng() % 0xFFFFFFFF00000001ULL;
        std::vector<u64> digests(std::max<u64>(n_digests, 1) * HASH_SIZE_U64);
        std::vector<u64> caps(n_caps * HASH_SIZE_U64);
        std::vector<u64> ref_caps;

        for (const Variant &v : variants)
        {
            for (int t : a.threads)
            {
                omp_set_num_threads(t);
                auto r = time_ms(v.fn, a, digests.data(), caps.data(), leaves.data(), n_digests, n_caps, n_leaves);
                printf("%6d %10lu %8s %8d %12.3f %12.3f %14.2f\n", log_n, (unsigned long)n_leaves, v.name, t,
                       r.first, r.second, (double)n_leaves / r.first);
                fflush(stdout);

                // All variants and thread counts must produce the same Merkle cap.
                if (ref_caps.empty())
                    ref_caps = caps;
                else if (ref_caps != caps)
                {
                    fprintf(stderr, "ERROR: cap mismatch for variant=%s threads=%d log_n=%d\n", v.name, t, log_n);
                    return 1;
                }
            }
        }
    }
    return 0;
}
