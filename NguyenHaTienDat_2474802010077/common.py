# Module dùng chung: SV2, SV3 chỉ cần `from common import *`
import copy, time, numpy as np, torch, torch.nn as nn, yaml
from torchvision import datasets

CFG = yaml.safe_load(open('config.yaml'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline: cấu hình mặc định hợp lý, không tuning
BASELINE = dict(lr=1e-3, weight_decay=0.0, hidden=128, dropout=0.0, batch_size=128)

def load_data():
    # Tải Fashion-MNIST; chia cố định train/val từ tập train gốc, test = tập test gốc
    tr = datasets.FashionMNIST('/content/data', train=True, download=True)
    te = datasets.FashionMNIST('/content/data', train=False, download=True)
    prep = lambda d: ((d.data.float() / 255 - 0.2860) / 0.3530).reshape(len(d.data), -1)
    X, y, Xte, yte = prep(tr), tr.targets, prep(te), te.targets
    idx = np.random.RandomState(CFG['seed']).permutation(len(X))
    a, b = CFG['n_train'], CFG['n_train'] + CFG['n_val']
    d = dict(Xtr=X[idx[:a]], ytr=y[idx[:a]], Xva=X[idx[a:b]], yva=y[idx[a:b]], Xte=Xte, yte=yte)
    return {k: v.to(DEVICE) for k, v in d.items()}

def make_mlp(hidden, dropout):
    return nn.Sequential(
        nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 10)).to(DEVICE)

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def latency_ms(model, X, n_rep=30):
    # Độ trễ suy diễn trên CPU: median thời gian / mẫu (ms), batch 256
    m = copy.deepcopy(model).cpu().eval()
    xb, ts = X[:256].cpu(), []
    with torch.no_grad():
        for _ in range(3):
            m(xb)
        for _ in range(n_rep):
            t = time.perf_counter(); m(xb); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) / len(xb) * 1000

def train_eval(params, data, train_seed=None, epochs=None):
    # f(x): huấn luyện MLP với siêu tham số x, trả về metric trên tập VALIDATION
    seed = CFG['seed'] if train_seed is None else train_seed
    epochs = epochs or CFG['epochs']
    torch.manual_seed(seed)
    model = make_mlp(int(params['hidden']), float(params['dropout']))
    opt = torch.optim.Adam(model.parameters(), lr=float(params['lr']),
                           weight_decay=float(params['weight_decay']))
    lossf, bs = nn.CrossEntropyLoss(), int(params['batch_size'])
    X, y = data['Xtr'], data['ytr']
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(X[b]), y[b]).backward()
            opt.step()
    train_time = time.perf_counter() - t0
    metrics = dict(val_acc=accuracy(model, data['Xva'], data['yva']),
                   n_params=sum(p.numel() for p in model.parameters()),
                   latency_ms=latency_ms(model, data['Xva']),
                   train_time=train_time)
    return metrics, model

def test_accuracy(model, data):
    # CHỈ gọi một lần ở bước báo cáo cuối, không dùng để chọn tham số
    return accuracy(model, data['Xte'], data['yte'])
