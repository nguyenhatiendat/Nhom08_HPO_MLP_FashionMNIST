<div align="center">

# 🧠 Nhom08_HPO_MLP_FashionMNIST

### Tối ưu siêu tham số mạng nơ-ron MLP trên Fashion-MNIST
**Grid Search · Random Search · TPE (Optuna) — cùng ngân sách đánh giá**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-TPE-1F77B4)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab&logoColor=white)

🎓 Đồ án môn **Nhập môn Tối ưu hóa** · Trường Đại học Văn Lang · Khoa CNTT (chuyên ngành AI)

[👥 Nhóm](#-thành-viên-và-phân-công) · [🎯 Bài toán](#-phát-biểu-bài-toán) · [🧪 Thực nghiệm](#-thiết-kế-thực-nghiệm-so-sánh-công-bằng) · [📁 Cấu trúc](#-cấu-trúc-repo) · [🚀 Cách chạy](#-cách-chạy-google-colab) · [✅ Tiến độ](#-tiến-độ)

</div>

---

## 👥 Thành viên và phân công

| STT | Họ và tên | MSSV | Thư mục | Vai trò |
|:---:|---|---|---|---|
| 1 | Nguyễn Hà Tiến Đạt | 2474802010077 | `NguyenHaTienDat_2474802010077/` | 📐 Mô hình hóa · Baseline · Dữ liệu |
| 2 | Nguyễn Bảo Lâm | [247480201.... | `NguyenBaoLam_247480201..../` | ⚙️ Thuật toán chính (Grid, Random, TPE) · Code |
| 3 | [Họ tên SV3] | [MSSV] | `[Thư mục SV3]/` | 📊 Thiết kế thực nghiệm · Phân tích · Trực quan hóa |

> 💡 Mỗi thành viên làm việc trong thư mục riêng. Đóng góp được ghi nhận qua **commit history**.

---

## 🎯 Phát biểu bài toán

Đây là bài toán **tối ưu hộp đen**: tìm bộ siêu tham số cho MLP phân loại ảnh Fashion-MNIST sao cho độ chính xác trên tập validation cao nhất, trong một **ngân sách đánh giá cố định**.

| Thành phần | Nội dung |
|---|---|
| 🎛️ **Biến quyết định x** | `(lr, weight_decay, hidden, dropout, batch_size)` |
| 📏 **Miền giá trị** | lr ∈ [1e-4, 1e-1] (log) · weight_decay ∈ [1e-6, 1e-2] (log) · hidden ∈ [32, 1024] (log, số nguyên) · dropout ∈ [0, 0.5] · batch_size ∈ {64, 128, 256} |
| 🎯 **Objective chính** | Maximize `val_acc` (accuracy trên validation sau 5 epoch) |
| ⚖️ **Objective phụ (Pareto)** | Minimize `latency_ms` (CPU, median, batch 256) song song với maximize `val_acc` |
| 🔒 **Ràng buộc** | Mỗi phương pháp tối đa **B = 32** lần đánh giá |

---

## 🧪 Thiết kế thực nghiệm (so sánh công bằng)

| | Nội dung |
|---|---|
| 🗂️ **Dữ liệu** | Fashion-MNIST (tải tự động bằng `torchvision`): 10.000 train · 5.000 validation (tách cố định từ train gốc) · 10.000 test gốc |
| 📌 **Baseline** | MLP cấu hình mặc định (lr = 1e-3, weight_decay = 0, hidden = 128, dropout = 0, batch_size = 128), chạy 10 seed |
| 🔍 **Phương pháp** | Grid Search (2⁵ = 32 điểm) · Random Search · TPE (Optuna) |
| 🎲 **Seed** | 10 seed cho Random và TPE; báo cáo mean ± std, best, worst |
| 🚫 **Test set** | Chỉ dùng **một lần** ở bước báo cáo cuối, không dùng để chọn siêu tham số |
| 📝 **Lưu vết** | Mọi trial ghi ra CSV trong `results/` để vẽ convergence và tái lập |

---

## 🚀 Cách chạy (Google Colab)

1. Mở Colab → **Runtime → Change runtime type → T4 GPU**.
2. Chạy ô sau để lấy code dùng chung:

```python
!git clone https://github.com/nguyenhatiendat/Nhom08_HPO_MLP_FashionMNIST.git
%cd Nhom08_HPO_MLP_FashionMNIST/NguyenHaTienDat_2474802010077
!pip -q install optuna pyyaml
from common import *
data = load_data()
```

3. Chạy các notebook theo thứ tự: `01` (dữ liệu, baseline) ➜ `02` (Grid, Random, TPE) ➜ `03` (phân tích, hình).

---

## 🔁 Tái lập kết quả

- ⚙️ Toàn bộ tham số trong `config.yaml` (seed = 42, B = 32, 10 seed, epochs = 5).
- 💻 Môi trường: Google Colab, GPU T4, Python 3, PyTorch, Optuna *(ghi rõ phiên bản khi hoàn tất thí nghiệm)*.
- 📊 Mọi bảng và hình trong báo cáo sinh từ các file CSV trong `results/`.

---

## ✅ Tiến độ

- [x] Chốt đề tài và formulation
- [x] Notebook 01: dữ liệu, split, hàm huấn luyện, baseline
- [ ] Notebook 02: Grid, Random, TPE (10 seed), lưu từng trial
- [ ] Notebook 03: convergence, boxplot, sensitivity, ablation, Pareto
- [ ] Báo cáo (5 chương), slide, demo

---
