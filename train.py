import os, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# Arsitektur GPT byte-level (~28.6M parameter, fp32 => ~114 MB)
D, L, H, T, V = 512, 9, 8, 256, 256
BS = int(os.environ.get("BS", "16"))
MINUTES = float(os.environ.get("TRAIN_MINUTES", "290"))
TARGET = int(os.environ.get("DATA_MB", "40")) * 1024 * 1024
MAXLR = 6e-4
OUT = "out"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1337)
np.random.seed(1337)
torch.set_num_threads(os.cpu_count() or 4)


def load_data():
    from datasets import load_dataset
    print("Download dataset Wikipedia Indonesia...", flush=True)
    ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10000)
    buf, n = [], 0
    for ex in ds:
        t = ex["text"].strip()
        if len(t) < 300:
            continue
        b = (t + "\n\n").encode("utf-8")
        buf.append(b)
        n += len(b)
        if n >= TARGET:
            break
    data = np.frombuffer(b"".join(buf), dtype=np.uint8).copy()
    print("Data: %.1f MB" % (len(data) / 1e6), flush=True)
    return data


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1 = nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D, bias=False)
        s.proj = nn.Linear(D, D, bias=False)
        s.ln2 = nn.LayerNorm(D)
        s.fc = nn.Linear(D, 4 * D, bias=False)
        s.out = nn.Linear(4 * D, D, bias=False)

    def forward(s, x):
        B, Tt, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, dim=2)
        q = q.view(B, Tt, H, C // H).transpose(1, 2)
        k = k.view(B, Tt, H, C // H).transpose(1, 2)
        v = v.view(B, Tt, H, C // H).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, Tt, C)
        x = x + s.proj(y)
        x = x + s.out(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))
        return x


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.wte = nn.Embedding(V, D)
        s.wpe = nn.Embedding(T, D)
        s.blocks = nn.ModuleList([Block() for _ in range(L)])
        s.lnf = nn.LayerNorm(D)
        s.apply(s._init)
        for b in s.blocks:
            nn.init.normal_(b.proj.weight, 0, 0.02 / math.sqrt(2 * L))
            nn.init.normal_(b.out.weight, 0, 0.02 / math.sqrt(2 * L))

    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(s, idx, targets=None):
        B, Tt = idx.shape
        x = s.wte(idx) + s.wpe(torch.arange(Tt))
        for b in s.blocks:
            x = b(x)
        x = s.lnf(x)
        logits = x @ s.wte.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        return logits, loss


def export(model, path):
    with open(path, "wb") as f:
        f.write(np.array([0x4C4C4D31, D, L, H, T, V], dtype=np.int32).tobytes())

        def w(t):
            f.write(t.detach().cpu().float().contiguous().numpy().tobytes())

        w(model.wte.weight)
        w(model.wpe.weight)
        for b in model.blocks:
            for t in (b.ln1.weight, b.ln1.bias, b.qkv.weight, b.proj.weight,
                      b.ln2.weight, b.ln2.bias, b.fc.weight, b.out.weight):
                w(t)
        w(model.lnf.weight)
        w(model.lnf.bias)
    print("Export %s: %.1f MB" % (path, os.path.getsize(path) / 1e6), flush=True)


@torch.no_grad()
def greedy(model, prompt, n):
    ids = list(prompt.encode())
    for _ in range(n):
        x = torch.tensor([ids[-T:]])
        logits, _ = model(x)
        ids.append(int(logits[0, -1].argmax()))
    return bytes(ids)


def main():
    data = load_data()
    cut = int(len(data) * 0.99)
    tr, va = data[:cut], data[cut:]

    def batch(d):
        ix = np.random.randint(0, len(d) - T - 1, BS)
        x = torch.from_numpy(np.stack([d[i:i + T] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([d[i + 1:i + T + 1] for i in ix]).astype(np.int64))
        return x, y

    model = GPT()
    n_par = sum(p.numel() for p in model.parameters())
    print("Parameter: %.2fM (%.1f MB fp32)" % (n_par / 1e6, n_par * 4 / 1e6), flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=MAXLR, betas=(0.9, 0.95))

    total = MINUTES * 60
    t0 = time.time()
    last_save = t0
    step = 0
    while True:
        el = time.time() - t0
        if el >= total:
            break
        frac = el / total
        lr = MAXLR * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(tr)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print("step %d | loss %.4f | lr %.2e | %.1f mnt" % (step, loss.item(), lr, el / 60), flush=True)
        if step % 200 == 0 and step > 0:
            with torch.no_grad():
                vl = sum(model(*batch(va))[1].item() for _ in range(10)) / 10
            print(">>> val loss %.4f" % vl, flush=True)
        if time.time() - last_save > 1800:
            export(model, os.path.join(OUT, "model.bin"))
            last_save = time.time()
        step += 1

    export(model, os.path.join(OUT, "model.bin"))
    model.eval()
    ref = greedy(model, "Indonesia", 40)
    with open(os.path.join(OUT, "ref.bin"), "wb") as f:
        f.write(ref)
    print("Contoh greedy:", ref.decode("utf-8", "replace"), flush=True)


if __name__ == "__main__":
    main()
