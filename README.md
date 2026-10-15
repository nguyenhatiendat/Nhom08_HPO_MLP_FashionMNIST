Nhom08_HPO_MLP_FashionMNIST

Đồ án môn Nhập môn Tối ưu hóa – Trường Đại học Văn Lang, Khoa Công nghệ thông tin (chuyên ngành Trí tuệ nhân tạo)

Đề tài: Tối ưu siêu tham số mạng nơ-ron MLP trên Fashion-MNIST bằng Grid Search, Random Search và TPE (Optuna) dưới cùng ngân sách đánh giá, kèm phân tích Pareto accuracy – latency.

1. Thành viên và phân công
STT	Họ và tên	MSSV	Thư mục	Vai trò
1	Nguyễn Hà Tiến Đạt	2474802010077	NguyenHaTienDat_2474802010077/	Mô hình hóa, baseline, dữ liệu
2	[Họ tên SV2]	[MSSV]	[Thư mục SV2]/	Thuật toán chính (Grid, Random, TPE), code
3	[Họ tên SV3]	[MSSV]	[Thư mục SV3]/	Thiết kế thực nghiệm, phân tích, trực quan hóa

Mỗi thành viên làm việc trong thư mục của mình. Mọi đóng góp được ghi nhận qua commit history của repo.

2. Phát biểu bài toán

Bài toán tối ưu hộp đen: tìm bộ siêu tham số cho MLP phân loại ảnh Fashion-MNIST sao cho độ chính xác trên tập validation cao nhất, trong một ngân sách đánh giá cố định.

Thành phần	Nội dung
Biến quyết định x	(lr, weight_decay, hidden, dropout, batch_size)
Miền	lr ∈ [1e-4, 1e-1] (log), weight_decay ∈ [1e-6, 1e-2] (log), hidden ∈ [32, 1024] (log, số nguyên), dropout ∈ [0, 0.5], batch_size ∈ {64, 128, 256}
Objective chính	Maximize val_acc (accuracy trên tập validation sau 5 epoch)
Objective phụ (Pareto)	Minimize latency_ms (CPU, median, batch 256) cùng với maximize val_acc
Ràng buộc	Mỗi phương pháp tối đa B = 32 lần đánh giá
3. Thiết kế thực nghiệm (so sánh công bằng)
Dữ liệu: Fashion-MNIST (tải tự động bằng torchvision). 10.000 mẫu train, 5.000 mẫu validation (tách cố định từ tập train gốc), 10.000 mẫu test gốc.
Baseline: MLP với cấu hình mặc định (lr = 1e-3, weight_decay = 0, hidden = 128, dropout = 0, batch_size = 128), chạy 10 seed.
Phương pháp so sánh: Grid Search (2⁵ = 32 điểm), Random Search, TPE (Optuna). Cùng dữ liệu, cùng split, cùng metric, cùng ngân sách B.
Seed: 10 seed cho thuật toán ngẫu nhiên (Random, TPE); báo cáo mean ± std, best, worst.
Test set: chỉ dùng một lần ở bước báo cáo cuối, không dùng để chọn siêu tham số.
Lưu vết: mọi trial được ghi ra CSV trong thư mục results/ để vẽ convergence và tái lập kết quả.
4. Cấu trúc repo
Nhom08_HPO_MLP_FashionMNIST/
├── README.md
├── NguyenHaTienDat_2474802010077/      # SV1
│   ├── common.py                       # hàm dùng chung: load_data, train_eval, ...
│   ├── config.yaml                     # seed, ngân sách, search space, lưới Grid
│   ├── Nhom08_..._01_data_baseline.ipynb
│   └── results/
│       └── baseline.csv
├── [Thư mục SV2]/                      # thuật toán chính, kết quả từng trial
└── [Thư mục SV3]/                      # phân tích, hình, báo cáo
5. Cách chạy (Google Colab)
Mở Colab, chọn Runtime → Change runtime type → T4 GPU.
Chạy ô sau để lấy code dùng chung:
python
!git clone https://github.com/nguyenhatiendat/Nhom08_HPO_MLP_FashionMNIST.git
%cd Nhom08_HPO_MLP_FashionMNIST/NguyenHaTienDat_2474802010077
!pip -q install optuna pyyaml
from common import *
data = load_data()
Chạy lần lượt các notebook theo thứ tự 01 (dữ liệu, baseline) → 02 (Grid, Random, TPE) → 03 (phân tích, hình).
6. Tái lập kết quả
Toàn bộ tham số nằm trong config.yaml (seed = 42, B = 32, 10 seed, epochs = 5).
Môi trường: Google Colab, GPU T4, Python 3, PyTorch, Optuna (ghi rõ phiên bản khi hoàn tất thí nghiệm).
Mọi bảng và hình trong báo cáo được sinh từ các file CSV trong results/.
7. Tiến độ
 Chốt đề tài và formulation
 Notebook 01: dữ liệu, split, hàm huấn luyện, baseline
 Notebook 02: Grid, Random, TPE (10 seed), lưu từng trial
 Notebook 03: convergence, boxplot, sensitivity, ablation, Pareto
 Báo cáo (5 chương), slide, demo
8. Tài liệu và báo cáo
Báo cáo: (thêm link sau khi hoàn thành)
Slide: (thêm link sau khi hoàn thành)
