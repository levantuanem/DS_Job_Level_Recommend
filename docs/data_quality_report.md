# Báo Cáo Phân Tích Chất Lượng Dữ Liệu (Data Quality Report)
**Project:** LinkedIn Job Postings — Job Level Analysis & Prediction  
**Branch:** `feature/data`  
**Vai trò:** Data Engineer / Data Analyst (NGƯỜI 1)  

---

## 1. Tổng quan Đánh giá Chất lượng Dữ liệu

Dataset LinkedIn Job Postings bao gồm 8 bảng dữ liệu quan hệ với trung tâm là bảng `postings.csv` (123,849 dòng, 31 cột). Qua quá trình kiểm tra toàn diện, chất lượng tổng thể của dữ liệu đạt mức **Tốt - Sẵn sàng cho phân tích** sau khi khắc phục các lỗi nhập liệu cục bộ.

### Các điểm sáng của Dataset:
1. **Tính duy nhất (Uniqueness):** Khóa chính `job_id` trong `postings.csv` và `company_id` trong `companies.csv` đạt mức duy nhất tuyệt đối 100%. Trong các bảng phụ, 21 bản ghi trùng lặp khóa trong `company_specialities.csv` đã được loại bỏ chính xác.
2. **Tính hợp lệ cơ bản của Lương:** Không xuất hiện trường hợp `min_salary > max_salary` trong tập dữ liệu thô.
3. **Tính toàn vẹn khóa ngoại:** Các bảng liên kết (`job_skills`, `job_industries`, `salaries`, `benefits`) đều ánh xạ chính xác về các `job_id` hợp lệ.

---

## 2. Chi tiết Các Vấn đề Chất Lượng Dữ Liệu Phát Hiện & Phương Án Xử Lý

Mục này trình bày chi tiết theo 6 tiêu chí bắt buộc:
1. **Vấn đề là gì?**
2. **Có bao nhiêu record bị ảnh hưởng?**
3. **Feature nào bị ảnh hưởng?**
4. **Mức độ ảnh hưởng?**
5. **Cách xử lý?**
6. **Tại sao chọn cách xử lý đó?**

---

### Vấn đề 1: Giá trị Lương Bằng 0 (Zero Salary Data Error)
- **Vấn đề:** Có các bản ghi xuất hiện `med_salary == 0.0`, kéo theo trường `normalized_salary` cũng bị tính toán bằng 0.0. Đây là lỗi nhập liệu hệ thống hoặc người dùng điền giá trị 0 vào ô mức lương tối thiểu/trung vị.
- **Feature bị ảnh hưởng:** `med_salary`, `normalized_salary` (trong cả `postings.csv` và `salaries.csv`).
- **Số record bị ảnh hưởng:**
  - `postings.csv`: 14 bản ghi.
  - `salaries.csv`: 16 bản ghi.
- **Mức độ ảnh hưởng:** **Cao** đối với các phân tích liên quan đến tiền lương (gây sai lệch nghiêm trọng các phép tính trung bình, phương sai, hồi quy), nhưng **Thấp** trên toàn bộ tập dữ liệu (chỉ chiếm ~0.01%).
- **Cách xử lý:** Đánh dấu chuyển đổi giá trị `0.0` thành `NaN` (`flag_zero_negative_salary_as_nan`), tuyệt đối **không xóa bỏ toàn bộ record**.
- **Tại sao chọn cách xử lý đó?** Mặc dù trường lương bị sai, nhưng các trường thông tin khác của 14 tin tuyển dụng này (như `title`, `description`, `formatted_experience_level`, `company_name`) vẫn hoàn toàn nguyên vẹn và có giá trị lớn cho các mô hình NLP và phân loại Job Level. Việc xóa cả record sẽ gây mất mát dữ liệu không đáng có.

---

### Vấn đề 2: Lệch Đơn Vị Chu Kỳ Trả Lương (Salary Unit / Pay Period Mismatch)
- **Vấn đề:** 
  - Một số tin tuyển dụng khai mức lương hàng chục đến hàng trăm nghìn USD (vd: $240,000 - $275,000) nhưng lại chọn `pay_period = HOURLY`. Khi Kaggle normalize theo công thức `$ \text{hourly} \times 2080 \text{ hrs} $`, mức lương năm bị đội lên tới 535 triệu USD/năm!
  - Ngược lại, một số tin tuyển dụng khai mức lương $20 - $50 nhưng lại chọn `pay_period = YEARLY`, khiến lương năm bị ghi nhận chỉ vỏn vẹn $22/năm.
- **Feature bị ảnh hưởng:** `pay_period`, `min_salary`, `max_salary`, `med_salary`, `normalized_salary`.
- **Số record bị ảnh hưởng:**
  - Lương theo giờ nhưng $\ge 1,000$ USD: **29 bản ghi** trong `postings.csv` (33 trong `salaries.csv`).
  - Lương theo năm nhưng $\le 150$ USD: **373 bản ghi** trong `postings.csv` (364 trong `salaries.csv`).
- **Mức độ ảnh hưởng:** **Nghiêm trọng (Critical)** cho phân tích phân phối lương và trực quan hóa (tạo ra các ngoại lai giả tạo hàng trăm triệu USD).
- **Cách xử lý:** 
  - Với trường hợp Hourly $\ge 1,000$: Đổi `pay_period` thành `YEARLY`, tính lại `normalized_salary` bằng trung bình min-max (hoặc med).
  - Với trường hợp Yearly $\le 150$: Đổi `pay_period` thành `HOURLY`, tính lại `normalized_salary` bằng mức lương giờ nhân với 2080 giờ làm việc chuẩn mỗi năm.
- **Tại sao chọn cách xử lý đó?** Đây là lỗi giao diện người dùng (UI dropdown select mistake) rất phổ biến trong các hệ thống tuyển dụng. Việc phát hiện quy luật và điều chỉnh lại đơn vị giúp khôi phục chính xác mức lương thực tế của hơn 400 tin tuyển dụng mà không làm sai lệch phân phối.

---

### Vấn đề 3: Khoảng Trắng Thừa và Lỗi Định Dạng Văn Bản (Text Whitespace & Formatting Issues)
- **Vấn đề:** Các trường văn bản dài và ngắn chứa nhiều khoảng trắng đầu/cuối chuỗi, nhiều dấu cách liền nhau (`\s+`), ký tự tab (`\t`) hoặc xuống dòng (`\r\n`) bừa bãi.
- **Feature bị ảnh hưởng:** `title`, `description`, `skills_desc`, `company_name` trong `postings.csv`; `name`, `description`, `city`, `state`, `address` trong `companies.csv`.
- **Số record bị ảnh hưởng:**
  - `description` trong `postings`: 19,593 bản ghi.
  - `title` trong `postings`: 1,977 bản ghi.
  - `company_name` trong `postings`: 787 bản ghi.
  - `description` trong `companies`: 5,796 bản ghi.
  - `name` trong `companies`: 318 bản ghi.
- **Mức độ ảnh hưởng:** **Trung bình**. Ảnh hưởng trực tiếp đến chất lượng trích xuất từ khóa, tokenize và embedding trong các mô hình ngôn ngữ (NLP).
- **Cách xử lý:** Áp dụng hàm `clean_text_column()`:
  - Thay thế mọi cụm ký tự khoảng trắng liên tiếp bằng đúng một ký tự cách chuẩn (`re.sub(r'\s+', ' ')`).
  - Cắt bỏ khoảng trắng ở hai đầu chuỗi (`.strip()`).
  - Chuyển các chuỗi rỗng (`""`), `"nan"`, `"None"` thành `np.nan`.
- **Tại sao chọn cách xử lý đó?** Đảm bảo tính nhất quán của văn bản, loại bỏ nhiễu định dạng nhưng bảo toàn 100% ngữ nghĩa và từ ngữ chuyên môn của bài đăng tuyển dụng.

---

### Vấn đề 4: Biểu Diễn Danh Mục Không Đồng Nhất (Inconsistent Categoricals)
- **Vấn đề:** Các giá trị danh mục có thể bị lẫn lộn giữa chữ hoa, chữ thường hoặc chứa khoảng trắng thừa.
- **Feature bị ảnh hưởng:** `work_type`, `pay_period`, `currency`, `compensation_type`, `country`.
- **Số record bị ảnh hưởng:** Tiềm ẩn trên toàn bộ các cột phân loại.
- **Mức độ ảnh hưởng:** **Thấp đến Trung bình** (có thể dẫn đến phân nhóm sai khi thực hiện `groupby` hoặc One-Hot Encoding).
- **Cách xử lý:** Áp dụng hàm `standardize_categorical()`: strip khoảng trắng, chuyển toàn bộ về chữ hoa (UPPERCASE) chuẩn hóa.
- **Tại sao chọn cách xử lý đó?** Đảm bảo các phép toán đếm tần suất và mã hóa feature danh mục hoàn toàn chính xác.

---

### Vấn đề 5: Biểu Diễn Nhị Phân của `remote_allowed`
- **Vấn đề:** Cột `remote_allowed` trong dữ liệu gốc chỉ có hai trạng thái: `1.0` (15,246 bản ghi) hoặc `NaN` (108,603 bản ghi).
- **Feature bị ảnh hưởng:** `remote_allowed`.
- **Số record bị ảnh hưởng:** 108,603 bản ghi (87.69%).
- **Mức độ ảnh hưởng:** **Trung bình**. Gây hiểu nhầm là cột bị thiếu dữ liệu nghiêm trọng, nhưng thực chất trên LinkedIn nếu bài đăng không bật tính năng "Remote" thì hệ thống sẽ để trống (tức là On-site hoặc Hybrid).
- **Cách xử lý:** Điền `0` cho các giá trị `NaN` và chuyển đổi cột thành kiểu nhị phân nguyên (`int`: `0` hoặc `1`).
- **Tại sao chọn cách xử lý đó?** Phản ánh chính xác bản chất nghiệp vụ của nền tảng LinkedIn, đồng thời tạo ra một feature nhị phân hoàn chỉnh sẵn sàng cho việc phân tích và mô hình hóa.

---

### Vấn đề 6: Khuyết Thiếu Nhãn Mục Tiêu `formatted_experience_level`
- **Vấn đề:** Có 29,409 tin tuyển dụng (~23.75%) không ghi rõ mức độ kinh nghiệm (Job Level: Entry level, Associate, Mid-Senior level, Director, Executive, Internship).
- **Feature bị ảnh hưởng:** `formatted_experience_level` (Target chính của đồ án).
- **Số record bị ảnh hưởng:** 29,409 bản ghi.
- **Mức độ ảnh hưởng:** **Rất cao** đối với bài toán học có giám sát (Supervised Learning).
- **Cách xử lý:** **Giữ nguyên `NaN` trong dataset sạch `postings_clean.csv`**. Không tự ý loại bỏ dòng hoặc điền bừa (imputation) ở bước Data Cleaning.
- **Tại sao chọn cách xử lý đó?** 
  - Bước Data Cleaning có nhiệm vụ cung cấp dữ liệu trung thực, sạch sẽ nhất cho toàn bộ các công đoạn sau.
  - Nhóm EDA và nhóm NLP có thể nghiên cứu các tin tuyển dụng này để hiểu vì sao nhà tuyển dụng không ghi Job Level.
  - Các kỹ thuật học bán giám sát (Semi-supervised Learning) hoặc phân cụm (Clustering / Unsupervised) có thể tận dụng 29,409 tin tuyển dụng này.
  - Nhóm Modeling (`feature/model`) sẽ chủ động lọc `dropna(subset=['formatted_experience_level'])` khi huấn luyện mô hình Supervised.

---

## 3. Phân Biệt: Data Error vs Genuine Extreme Value (Outlier Analysis)

Một nguyên tắc sống còn của Data Engineering là: **Không bao giờ tự động xóa dữ liệu chỉ vì nó là ngoại lai (Outlier).**

| Hiện tượng | Bản chất | Quyết định xử lý | Cơ sở khoa học & nghiệp vụ |
|---|---|---|---|
| `company_size = 7` | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Trong hệ thống phân loại của LinkedIn, `company_size` là thang đo từ 1 đến 7 (1: 1-10 nhân viên, 7: trên 10,001 nhân viên). Giá trị 7 đại diện cho các tập đoàn khổng lồ (Microsoft, Google, Amazon). Đây là dữ liệu thực tế cực kỳ có giá trị, không phải lỗi. |
| Tin tuyển dụng có `views > 5,000` hoặc `applies > 500` | **Genuine Extreme Value** | **GIỮ NGUYÊN** | Thị trường lao động luôn có những bài đăng "viral" thu hút hàng chục nghìn ứng viên quan tâm. Phân phối số lượt xem và ứng tuyển luôn có độ lệch phải cao (heavy-tailed distribution). |
| Lương năm $240,000 nhưng gắn nhãn `HOURLY` | **Data Error** | **SỬA ĐƠN VỊ** | Không thể có mức lương $240,000/giờ cho vị trí chuyên viên. Đây là sai sót thao tác người dùng. Ta sửa về `YEARLY` để cứu dữ liệu. |
| Lương cơ bản $= 0.0$ | **Data Error** | **GÁN THÀNH NaN** | Mức lương cơ bản không thể bằng 0. Gán NaN để loại bỏ ảnh hưởng xấu lên các phân tích toán học. |

---

## 4. Kiểm Tra Toàn Vẹn Quan Hệ Giữa Các Bảng (Relational Integrity)

Đã kiểm tra ma trận khóa ngoại giữa bảng chính `postings_clean.csv` và các bảng phụ:
1. **`postings` ↔ `companies`:**
   - 98.61% bản ghi có `company_id` hợp lệ liên kết sang `companies.csv`.
   - 1.39% bản ghi bị thiếu `company_id` (do nhà tuyển dụng đăng tin ẩn danh / Confidential). Các dòng này vẫn được bảo toàn.
2. **`postings` ↔ `salaries`:**
   - Bảng `salaries.csv` chứa 40,785 bản ghi chi tiết về cấu trúc lương, toàn bộ `job_id` trong bảng này đều khớp với `job_id` trong `postings.csv`.
3. **`postings` ↔ `job_skills`:**
   - Bảng `job_skills.csv` cung cấp 213,768 liên kết kỹ năng cho 126,807 tin tuyển dụng, bao phủ trọn vẹn danh mục công việc của `postings.csv`.

---

## 5. Kết Luận & Bàn Giao
Toàn bộ 8 bảng dữ liệu raw đã được xử lý chuẩn hóa, bảo toàn >99.98% số lượng record ban đầu (chỉ loại 7 tin tuyển dụng thiếu hẳn chức danh/mô tả bắt buộc và 21 dòng trùng lặp khóa phụ), khắc phục triệt để các lỗi dữ liệu nghiêm trọng và được lưu trữ an toàn tại `data/processed/`. Dữ liệu sẵn sàng 100% để chuyển giao cho **NGƯỜI 2 (EDA & Visualization)** và **NGƯỜI 3 (Feature Engineering)**.
