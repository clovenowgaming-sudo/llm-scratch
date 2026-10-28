// run.c - mesin inferensi LLM buatan sendiri (GPT byte-level, format model.bin)
// Compile: clang -O3 -o run run.c -lm
// Pakai:   ./run model.bin -p "Indonesia adalah" -n 300 -t 0.8 -k 40 -s 42
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int D, L, H, T, V;
static float *wte, *wpe, *lnfw, *lnfb;
typedef struct { float *ln1w, *ln1b, *qkv, *proj, *ln2w, *ln2b, *fc, *out; } Layer;
static Layer *ly;
static float *x, *xb, *qkvb, *att, *hb, *kc, *vc, *logits;

static void load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tidak bisa buka %s\n", path); exit(1); }
    int h[6];
    if (fread(h, 4, 6, f) != 6 || h[0] != 0x4C4C4D31) { fprintf(stderr, "format model salah\n"); exit(1); }
    D = h[1]; L = h[2]; H = h[3]; T = h[4]; V = h[5];
    size_t d = D;
    size_t per = 2 * d + 3 * d * d + d * d + 2 * d + 4 * d * d + 4 * d * d;
    size_t n = (size_t)V * d + (size_t)T * d + (size_t)L * per + 2 * d;
    float *buf = malloc(n * sizeof(float));
    if (!buf) { fprintf(stderr, "RAM tidak cukup\n"); exit(1); }
    if (fread(buf, sizeof(float), n, f) != n) { fprintf(stderr, "file model kurang lengkap\n"); exit(1); }
    fclose(f);
    float *p = buf;
    wte = p; p += (size_t)V * d;
    wpe = p; p += (size_t)T * d;
    ly = malloc(sizeof(Layer) * L);
    for (int l = 0; l < L; l++) {
        ly[l].ln1w = p; p += d;
        ly[l].ln1b = p; p += d;
        ly[l].qkv = p;  p += 3 * d * d;
        ly[l].proj = p; p += d * d;
        ly[l].ln2w = p; p += d;
        ly[l].ln2b = p; p += d;
        ly[l].fc = p;   p += 4 * d * d;
        ly[l].out = p;  p += 4 * d * d;
    }
    lnfw = p; p += d;
    lnfb = p;
}

static void lnorm(float *o, const float *in, const float *w, const float *b) {
    float m = 0;
    for (int i = 0; i < D; i++) m += in[i];
    m /= D;
    float v = 0;
    for (int i = 0; i < D; i++) { float d = in[i] - m; v += d * d; }
    v /= D;
    float r = 1.0f / sqrtf(v + 1e-5f);
    for (int i = 0; i < D; i++) o[i] = (in[i] - m) * r * w[i] + b[i];
}

static void mv(float *o, const float *W, const float *in, int ni, int no) {
    for (int r = 0; r < no; r++) {
        const float *w = W + (size_t)r * ni;
        float s = 0;
        for (int i = 0; i < ni; i++) s += w[i] * in[i];
        o[r] = s;
    }
}

static void forward(int tok, int pos) {
    int hs = D / H;
    for (int i = 0; i < D; i++) x[i] = wte[(size_t)tok * D + i] + wpe[(size_t)pos * D + i];
    for (int l = 0; l < L; l++) {
        Layer *w = &ly[l];
        lnorm(xb, x, w->ln1w, w->ln1b);
        mv(qkvb, w->qkv, xb, D, 3 * D);
        memcpy(kc + ((size_t)l * T + pos) * D, qkvb + D, D * sizeof(float));
        memcpy(vc + ((size_t)l * T + pos) * D, qkvb + 2 * D, D * sizeof(float));
        float *ao = hb;
        for (int h = 0; h < H; h++) {
            float *q = qkvb + h * hs;
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                float *k = kc + ((size_t)l * T + t) * D + h * hs;
                float s = 0;
                for (int i = 0; i < hs; i++) s += q[i] * k[i];
                s /= sqrtf((float)hs);
                att[t] = s;
                if (s > mx) mx = s;
            }
            float sum = 0;
            for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - mx); sum += att[t]; }
            float *o = ao + h * hs;
            for (int i = 0; i < hs; i++) o[i] = 0;
            for (int t = 0; t <= pos; t++) {
                float *v = vc + ((size_t)l * T + t) * D + h * hs;
                float p = att[t] / sum;
                for (int i = 0; i < hs; i++) o[i] += p * v[i];
            }
        }
        mv(xb, w->proj, ao, D, D);
        for (int i = 0; i < D; i++) x[i] += xb[i];
        lnorm(xb, x, w->ln2w, w->ln2b);
        mv(hb, w->fc, xb, D, 4 * D);
        for (int i = 0; i < 4 * D; i++) {
            float v = hb[i];
            hb[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        }
        mv(xb, w->out, hb, 4 * D, D);
        for (int i = 0; i < D; i++) x[i] += xb[i];
    }
    lnorm(xb, x, lnfw, lnfb);
    mv(logits, wte, xb, D, V);
}

static uint64_t rng = 42;
static float rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) & 0xFFFFFF) / 16777216.0f;
}

static int cmp(const void *a, const void *b) {
    float p = logits[*(const int *)a], q = logits[*(const int *)b];
    return (p < q) - (p > q);
}

static int sample(float temp, int topk) {
    int *idx = malloc(sizeof(int) * V);
    for (int i = 0; i < V; i++) idx[i] = i;
    qsort(idx, V, sizeof(int), cmp);
    if (temp <= 0) { int r = idx[0]; free(idx); return r; }
    int k = topk < V ? topk : V;
    if (k < 1) k = 1;
    float *pr = malloc(sizeof(float) * k);
    float mx = logits[idx[0]], sum = 0;
    for (int i = 0; i < k; i++) { pr[i] = expf((logits[idx[i]] - mx) / temp); sum += pr[i]; }
    float r = rnd() * sum, c = 0;
    int out = idx[k - 1];
    for (int i = 0; i < k; i++) { c += pr[i]; if (r < c) { out = idx[i]; break; } }
    free(idx); free(pr);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "pakai: %s model.bin [-p prompt] [-n token] [-t temp] [-k topk] [-s seed]\n", argv[0]); return 1; }
    const char *prompt = "Indonesia ";
    int n = 300, topk = 40;
    float temp = 0.8f;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-p")) prompt = argv[i + 1];
        else if (!strcmp(argv[i], "-n")) n = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-t")) temp = (float)atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-k")) topk = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) rng = (uint64_t)atoll(argv[i + 1]) * 2654435761ULL + 88172645463325252ULL;
    }
    load(argv[1]);
    x = calloc(D, 4); xb = calloc(D, 4); qkvb = calloc(3 * D, 4); att = calloc(T, 4);
    hb = calloc(4 * D, 4); logits = calloc(V, 4);
    kc = calloc((size_t)L * T * D, 4); vc = calloc((size_t)L * T * D, 4);

    int plen = (int)strlen(prompt);
    if (plen == 0) { prompt = "\n"; plen = 1; }
    fwrite(prompt, 1, plen, stdout);
    int start = plen > T / 2 ? plen - T / 2 : 0;
    int *hist = malloc(sizeof(int) * (plen + n + 8));
    int nh = 0, pos = 0;
    for (int i = start; i < plen; i++) hist[nh++] = (unsigned char)prompt[i];
    for (int i = 0; i < nh; i++) forward(hist[i], pos++);
    for (int s = 0; s < n; s++) {
        int next = sample(temp, topk);
        putchar(next);
        fflush(stdout);
        hist[nh++] = next;
        if (pos >= T) {
            int keep = T / 2;
            pos = 0;
            for (int j = nh - keep; j < nh; j++) forward(hist[j], pos++);
        } else {
            forward(next, pos++);
        }
    }
    putchar('\n');
    return 0;
}
