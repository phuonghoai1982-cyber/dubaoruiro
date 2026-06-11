# 🛡️ Hệ thống Phát hiện Giao dịch Gian lận (Fraud Detection Web App)

Ứng dụng Web tương tác được xây dựng dựa trên Streamlit, giúp chuyển đổi toàn bộ quy trình phân tích và huấn luyện mô hình từ Jupyter Notebook (`phat_hien_giao_dich_gian_lan.ipynb`) thành một sản phẩm công nghệ có khả năng dự báo trực quan và xử lý dữ liệu thực tế.

## 📌 Tính năng chính

- **Cấu hình động (Sidebar):** Cho phép tùy biến trực tiếp siêu tham số mô hình học máy `RandomForestClassifier` (Số cây, độ sâu, tỷ lệ test split) và tải tệp dữ liệu lên nhanh chóng.
- **Tổng quan dữ liệu (Tab 1):** Tóm tắt thống kê, kiểm tra nhanh cấu trúc hàng, cột, dung lượng tệp tin.
- **Trực quan hóa (Tab 2):** Biểu đồ tương tác cao (Plotly) thể hiện chi tiết mật độ phân phối dữ liệu các biến đầu vào liên tục và biến mục tiêu.
- **Đánh giá mô hình (Tab 3):** Tái lập chính xác năng lực đo lường qua Ma trận nhầm lẫn (Confusion Matrix) và định lượng hóa độ quan trọng của các thuộc tính (Feature Importance).
- **Vận hành Dự báo (Tab 4):** Hỗ trợ 2 chế độ: Nhập trực tiếp các chỉ số qua biểu mẫu trực quan hoặc Tải file danh sách lớn để chấm điểm rủi ro hàng loạt tiện lợi.

## 📊 Quy chuẩn cấu trúc dữ liệu đầu vào

Ứng dụng yêu cầu tệp dữ liệu huấn luyện hoặc tệp chạy hàng loạt có định dạng `.csv` hoặc `.xlsx` bao gồm tối thiểu các cột thuộc tính định danh sau đây:
- **Biến đặc trưng đặc tả (X):** Gồm 14 cột từ `X_1`, `X_2`, `X_3`, ..., `X_14` (Kiểu số liên tục).
- **Biến mục tiêu phân lớp (y):** Cột mang tên `default` chứa giá trị nhị phân (`0`: Giao dịch bình thường, `1`: Giao dịch rủi ro/gian lận).

## 🛠️ Hướng dẫn cài đặt và vận hành

### Bước 1: Khởi tạo môi trường và cài đặt thư viện phụ thuộc
Đảm bảo bạn đã cài đặt Python (Khuyến nghị phiên bản từ 3.9 đến 3.12). Sử dụng terminal tại thư mục chứa mã nguồn để thực thi:
```bash
pip install -r requirements.txt
