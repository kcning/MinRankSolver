/* SPDX-License-Identifier: GPL-2.0 OR MIT */
/*
 * Copyright (C) 2015-2019 Jason A. Donenfeld <Jason@zx2c4.com>. All Rights Reserved.
 */

#ifndef __BLK_LANCZOS_BLAKE2S_H__
#define __BLK_LANCZOS_BLAKE2S_H__


#include <stdint.h>
#include <string.h>
#include <stdalign.h>
#include <assert.h>


enum blake2s_lengths {
    BLAKE2S_BLOCK_SIZE = 64,
    BLAKE2S_HASH_SIZE = 32,
    BLAKE2S_KEY_SIZE = 32,

    BLAKE2S_128_HASH_SIZE = 16,
    BLAKE2S_160_HASH_SIZE = 20,
    BLAKE2S_224_HASH_SIZE = 28,
    BLAKE2S_256_HASH_SIZE = 32,
};

struct blake2s_state {
    uint32_t h[8];
    uint32_t t[2];
    uint32_t f[2];
    uint8_t buf[BLAKE2S_BLOCK_SIZE];
    uint32_t buflen;
    uint32_t outlen;
};

#define BLAKE2S_IV0     0x6A09E667UL
#define BLAKE2S_IV1     0xBB67AE85UL
#define BLAKE2S_IV2     0x3C6EF372UL
#define BLAKE2S_IV3     0xA54FF53AUL
#define BLAKE2S_IV4     0x510E527FUL
#define BLAKE2S_IV5     0x9B05688CUL
#define BLAKE2S_IV6     0x1F83D9ABUL
#define BLAKE2S_IV7     0x5BE0CD19UL

void
blake2s_update(struct blake2s_state *state, const uint8_t *in, size_t inlen);

void
blake2s_final(struct blake2s_state *state, uint8_t *out);

static inline void
blake2s_init_param(struct blake2s_state *state, const uint32_t param) {
    memset(state, 0x0, sizeof(struct blake2s_state));
    state->h[0] = BLAKE2S_IV0 ^ param;
    state->h[1] = BLAKE2S_IV1;
    state->h[2] = BLAKE2S_IV2;
    state->h[3] = BLAKE2S_IV3;
    state->h[4] = BLAKE2S_IV4;
    state->h[5] = BLAKE2S_IV5;
    state->h[6] = BLAKE2S_IV6;
    state->h[7] = BLAKE2S_IV7;
}

static inline void
blake2s_init(struct blake2s_state *state, const size_t outlen) {
    blake2s_init_param(state, 0x01010000 | outlen);
    state->outlen = outlen;
}

static inline void
blake2s_init_key(struct blake2s_state *state, const size_t outlen,
                 const void *key, const size_t keylen) {
    assert(outlen && outlen <= BLAKE2S_HASH_SIZE);
    assert(key && keylen && keylen <= BLAKE2S_KEY_SIZE);
    blake2s_init_param(state, 0x01010000 | keylen << 8 | outlen);
    memcpy(state->buf, key, keylen);
    state->buflen = BLAKE2S_BLOCK_SIZE;
    state->outlen = outlen;
}

static inline void
blake2s(uint8_t *out, const uint8_t *in, const uint8_t *key,
        const size_t outlen, const size_t inlen, const size_t keylen) {
    struct blake2s_state state;
    assert(in || !inlen);
    assert(out && outlen && outlen <= BLAKE2S_HASH_SIZE);
    assert(keylen <= BLAKE2S_KEY_SIZE);
    assert(key || !keylen);

    if (keylen)
        blake2s_init_key(&state, outlen, key, keylen);
    else
        blake2s_init(&state, outlen);

    blake2s_update(&state, in, inlen);
    blake2s_final(&state, out);
}

void
blake2s256_hmac(uint8_t *out, const uint8_t *in, const uint8_t *key,
                const size_t inlen, const size_t keylen);

#endif /* __BLK_LANCZOS_BLAKE2S_H__ */
