# 💖 Đồ Án Chuyên Ngành 2 - Dự Đoán Bệnh Tim Sử Dụng KNN

### **Thông Tin Sinh Viên**
- **Họ và tên**: Bùi Đổ Hạnh Nguyên  
- **Lớp**: 21CNTTC  
- **Mã sinh viên**: 3120421014  

---

## **Mục Lục**
1. [Tổng Quan](#1-tổng-quan)  
2. [Bộ Dữ Liệu](#2-bộ-dữ-liệu)  
3. [Cấu Trúc Dự Án](#3-cấu-trúc-dự-án)  
4. [Kết Quả](#4-kết-quả)  

---

## **1. Tổng Quan**  
Dự án này tập trung vào việc dự đoán khả năng mắc bệnh tim của bệnh nhân dựa trên dữ liệu y tế, sử dụng thuật toán **K-Nearest Neighbors (KNN)**.  
Quá trình bao gồm:
- Tiền xử lý dữ liệu
- Trực quan hóa
- Huấn luyện mô hình
- Đánh giá hiệu suất  

**Mục tiêu**: Phân loại bệnh nhân thành hai nhóm:  
- **1**: Có nguy cơ mắc bệnh tim.  
- **0**: Không có nguy cơ mắc bệnh tim.  

---

## **2. Bộ Dữ Liệu**  
- **Tệp dữ liệu**: `heart.csv`  
- **Nguồn**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- **Thông tin chi tiết**:
  - **Số dòng**: 303  
  - **Số cột**: 14  

### **Các đặc trưng chính**:
| **Cột**       | **Ý Nghĩa**                          |
|----------------|--------------------------------------|
| **age**        | Tuổi của bệnh nhân                  |
| **sex**        | Giới tính (1 = nam, 0 = nữ)         |
| **cp**         | Loại đau ngực (0-3)                 |
| **trestbps**   | Huyết áp khi nghỉ ngơi (mm Hg)      |
| **chol**       | Cholesterol trong máu (mg/dl)       |
| **fbs**        | Đường huyết đói (>120 mg/dl, 1 = đúng, 0 = sai) |
| **thalach**    | Nhịp tim tối đa đạt được            |
| **target**     | Chẩn đoán bệnh tim (1 = có bệnh, 0 = không bệnh) |

---

## **3. Cấu Trúc Dự Án**  

### **3.1. Khám Phá Dữ Liệu**  
- Xem trước dữ liệu (`head`) để hiểu cấu trúc.  
- Phân tích thống kê (`info()`, `describe()`).  
- Kiểm tra giá trị thiếu và phân phối dữ liệu bằng IQR.  

### **3.2. Tiền Xử Lý Dữ Liệu**  
- **Xử lý dữ liệu thiếu**: Kiểm tra và thay thế giá trị bị thiếu.  
- **Mã hóa dữ liệu**: Chuyển đổi các cột dạng phân loại sang dạng số.  
- **Chuẩn hóa**: Đảm bảo các đặc trưng có cùng tỷ lệ, giúp tăng hiệu quả của KNN.  
- **Chia tập dữ liệu**:
  - Tập huấn luyện: Dùng để huấn luyện mô hình.  
  - Tập kiểm tra: Đánh giá hiệu suất mô hình.  
- **Transform dữ liệu**: Bằng phương pháp Box-Cox
  
### **3.3. Xây Dựng Mô Hình**  
- Áp dụng thuật toán **K-Nearest Neighbors (KNN)** từ thư viện `sklearn`.  
- Sử dụng **Grid Search** để chọn siêu tham số tối ưu.  

### **3.4. Đánh Giá Hiệu Suất**  
- **Ma trận nhầm lẫn (Confusion Matrix)**: Hiển thị kết quả phân loại.  
- **Các chỉ số đánh giá**:
  - **Accuracy**: Độ chính xác tổng quát.  
  - **Precision**: Tỷ lệ dự đoán đúng trong số dự đoán dương.  
  - **Recall**: Tỷ lệ phát hiện đúng các giá trị thực sự dương.  
  - **F1-Score**: Trung bình điều hòa giữa Precision và Recall.  
- So sánh hiệu suất trên tập huấn luyện và kiểm tra.  

### **3.5. Dự Đoán Kết Quả**  
- Dự đoán trên dữ liệu mới hoặc tập kiểm tra.  
- **Kết quả**:
  - **1**: Có nguy cơ mắc bệnh tim.  
  - **0**: Không có nguy cơ mắc bệnh tim.  

---

## **4. Kết Quả**  
- **Mục tiêu đạt được**: Xây dựng mô hình dự đoán bệnh tim chính xác.  
- **Kết quả đầu ra**:  
  - **1**: Có khả năng mắc bệnh tim.  
  - **0**: Không có khả năng mắc bệnh tim.  

---

### 🌟 **Liên Hệ**
📧 Email: nguyen25102003n@gmail.com  
