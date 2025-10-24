// SPDX-License-Identifier: GPL-2.0 OR MIT
/*
 * Copyright (C) 2015-2019 Jason A. Donenfeld <Jason@zx2c4.com>. All Rights Reserved.
 *
 * This is an implementation of the BLAKE2s hash and PRF functions.
 *
 * Information: https://blake2.net/
 *
 */

#include <stdint.h>
#include <string.h>
#include <stdalign.h>
#include <assert.h>

#include "blake2s.h"
#include "util.h"

#define SZ_4K 0x00001000

static inline void
blake2s_set_lastblock(struct blake2s_state *state) {
    state->f[0] = -1;
}

#if defined(__SSE3__)
extern void
blake2s_compress_ssse3(struct blake2s_state* state, const uint8_t* block,
                       const size_t nblocks, const uint32_t inc);

#if defined(__AVX512_F__)
extern void
blake2s_compress_avx512(struct blake2s_state* state, const uint8_t* block,
                        const size_t nblocks, const uint32_t inc);
#endif

void
blake2s_compress_arch(struct blake2s_state* state, const uint8_t* block,
                      size_t nblocks, const uint32_t inc) {
    do {
        const size_t blocks = (nblocks > ((size_t) SZ_4K / BLAKE2S_BLOCK_SIZE)) ?
                            ((size_t) SZ_4K / BLAKE2S_BLOCK_SIZE) : nblocks;

#if defined(__AVX512_F__)
        blake2s_compress_avx512(state, block, blocks, inc);
#else
        blake2s_compress_ssse3(state, block, blocks, inc);
#endif

        nblocks -= blocks;
        block += blocks * BLAKE2S_BLOCK_SIZE;
    } while (nblocks);
}
#endif

void
blake2s_compress_generic(struct blake2s_state* state, const uint8_t* block,
                         size_t nblocks, const uint32_t inc);

void
blake2s_update(struct blake2s_state *state, const uint8_t *in, size_t inlen) {
    const size_t fill = BLAKE2S_BLOCK_SIZE - state->buflen;
    
    if (unlikely(!inlen))
        return;
    if (inlen > fill) {
        memcpy(state->buf + state->buflen, in, fill);
#if defined(__SSE3__)
        blake2s_compress_arch(state, state->buf, 1, BLAKE2S_BLOCK_SIZE);
#else
        blake2s_compress_generic(state, state->buf, 1, BLAKE2S_BLOCK_SIZE);
#endif
        state->buflen = 0;
        in += fill;
        inlen -= fill;
    }
    if (inlen > BLAKE2S_BLOCK_SIZE) {
        size_t nblocks = (inlen + BLAKE2S_BLOCK_SIZE - 1) / BLAKE2S_BLOCK_SIZE;
        /* Hash one less (full) block than strictly possible */
#if defined(__SSE3__)
        blake2s_compress_arch(state, in, nblocks - 1, BLAKE2S_BLOCK_SIZE);
#else
        blake2s_compress_generic(state, in, nblocks - 1, BLAKE2S_BLOCK_SIZE);
#endif
        in += BLAKE2S_BLOCK_SIZE * (nblocks - 1);
        inlen -= BLAKE2S_BLOCK_SIZE * (nblocks - 1);
    }
    memcpy(state->buf + state->buflen, in, inlen);
    state->buflen += inlen;
}

void
blake2s_final(struct blake2s_state *state, uint8_t *out) {
    assert(out);
    blake2s_set_lastblock(state);
    memset(state->buf + state->buflen, 0,
           BLAKE2S_BLOCK_SIZE - state->buflen); /* Padding */
#if defined(__SSE3__)
    blake2s_compress_arch(state, state->buf, 1, state->buflen);
#else
    blake2s_compress_generic(state, state->buf, 1, state->buflen);
#endif
    memcpy(out, state->h, state->outlen);
#if !defined(BLK_LANCZOS_BLAKE2S_NO_CLEANUP)
    memset(state, 0x0, sizeof(*state));
#endif
}

void
blake2s256_hmac(uint8_t *out, const uint8_t *in, const uint8_t *key,
                const size_t inlen, const size_t keylen) {
    struct blake2s_state state;
    alignas(alignof(uint32_t)) uint8_t x_key[BLAKE2S_BLOCK_SIZE] = { 0 };
    alignas(alignof(uint32_t)) uint8_t i_hash[BLAKE2S_HASH_SIZE];

    if (keylen > BLAKE2S_BLOCK_SIZE) {
        blake2s_init(&state, BLAKE2S_HASH_SIZE);
        blake2s_update(&state, key, keylen);
        blake2s_final(&state, x_key);
    } else
        memcpy(x_key, key, keylen);

    for (int i = 0; i < BLAKE2S_BLOCK_SIZE; ++i)
            x_key[i] ^= 0x36;

    blake2s_init(&state, BLAKE2S_HASH_SIZE);
    blake2s_update(&state, x_key, BLAKE2S_BLOCK_SIZE);
    blake2s_update(&state, in, inlen);
    blake2s_final(&state, i_hash);

    for (int i = 0; i < BLAKE2S_BLOCK_SIZE; ++i)
            x_key[i] ^= 0x5c ^ 0x36;

    blake2s_init(&state, BLAKE2S_HASH_SIZE);
    blake2s_update(&state, x_key, BLAKE2S_BLOCK_SIZE);
    blake2s_update(&state, i_hash, BLAKE2S_HASH_SIZE);
    blake2s_final(&state, i_hash);

    memcpy(out, i_hash, BLAKE2S_HASH_SIZE);
#if !defined(BLK_LANCZOS_BLAKE2S_NO_CLEANUP)
    memset(x_key, 0x0, BLAKE2S_BLOCK_SIZE);
    memset(i_hash, 0x0, BLAKE2S_HASH_SIZE);
#endif
}
