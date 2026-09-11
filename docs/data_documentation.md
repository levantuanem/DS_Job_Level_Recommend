# Data Documentation — Branch `feature/data`

## 1. Dataset ban đầu (Data Collection)

- **Nguồn dữ liệu:** Kaggle — LinkedIn Job Postings (2023 - 2024).
- **Tên dataset:** LinkedIn Job Postings Dataset.
- **Dữ liệu raw:** Được lưu trữ nguyên vẹn tại thư mục `data/raw/` (không sửa đổi trực tiếp dữ liệu raw).
- **Phạm vi thời gian:**
  - `original_listed_time`: Từ `05/12/2023 02:28:53 UTC` đến `20/04/2024 03:06:43 UTC`.
  - `listed_time`: Từ `24/03/2024 16:30:14 UTC` đến `20/04/2024 03:06:56 UTC`.
  - `expiry`: Đến `16/10/2024 23:06:36 UTC`.

### Tổng quan số lượng bảng và thuộc tính:

| Bảng dữ liệu | Số dòng (records) | Số cột (features) | Khóa chính / Liên kết chính | Mô tả vai trò |
|---|---|---|---|---|
| `postings.csv` | 123,849 | 31 | `job_id` | Bảng sự kiện trung tâm (Job postings) |
| `companies.csv` | 24,473 | 10 | `company_id` | Bảng thông tin công ty |
| `company_industries.csv` | 24,375 | 2 | `company_id` | Ngành nghề của từng công ty |
| `company_specialities.csv` | 169,387 | 2 | `company_id` | Lĩnh vực chuyên môn của công ty |
| `job_industries.csv` | 164,808 | 2 | `job_id` | Ngành nghề được gán cho job posting |
| `job_skills.csv` | 213,768 | 2 | `job_id` | Danh mục kỹ năng được chuẩn hóa |
| `salaries.csv` | 40,785 | 8 | `salary_id`, `job_id` | Dữ liệu chi tiết về mức lương |
| `benefits.csv` | 67,943 | 3 | `job_id` | Phúc lợi kèm theo công việc |

---

## 2. Phân loại Feature (Data Understanding)

Các feature trong bảng trung tâm `postings.csv` được phân loại thành 7 nhóm nghiệp vụ chính:

1. **JOB INFORMATION:**
   - `title`: Chức danh công việc (Text)
   - `description`: Mô tả chi tiết trách nhiệm, yêu cầu công việc (Long Text)
   - `skills_desc`: Mô tả kỹ năng dưới dạng văn bản tự do (Text)
   - `formatted_experience_level`: Mức độ kinh nghiệm (Target chính của bài toán: Entry level, Mid-Senior level, Associate, Director, Executive, Internship)
2. **SALARY:**
   - `min_salary`: Mức lương tối thiểu
   - `med_salary`: Mức lương trung vị
   - `max_salary`: Mức lương tối đa
   - `normalized_salary`: Lương quy đổi chuẩn hóa theo năm (USD)
   - `currency`: Đơn vị tiền tệ (USD, EUR, CAD, AUD, GBP, BBD)
   - `pay_period`: Chu kỳ trả lương (HOURLY, YEARLY, MONTHLY, WEEKLY, BIWEEKLY)
   - `compensation_type`: Loại thù lao (BASE_SALARY)
3. **LOCATION:**
   - `location`: Địa điểm làm việc (thành phố, bang, quốc gia)
   - `zip_code`: Mã bưu chính
   - `fips`: Mã định danh địa lý FIPS
4. **WORK:**
   - `formatted_work_type`: Hình thức làm việc hiển thị (Full-time, Part-time, Contract, Internship, Temporary, Volunteer, Other)
   - `work_type`: Mã hình thức làm việc chuẩn hóa (FULL_TIME, PART_TIME, CONTRACT...)
   - `remote_allowed`: Cờ cho phép làm việc từ xa (1 = Remote, 0 = Onsite/Hybrid)
5. **ENGAGEMENT:**
   - `views`: Số lượt xem tin tuyển dụng
   - `applies`: Số lượt ứng tuyển nộp qua hệ thống
6. **TIME:**
   - `listed_time`: Thời điểm tin được đăng (Epoch millisecond)
   - `original_listed_time`: Thời điểm đăng tin gốc ban đầu
   - `expiry`: Thời điểm tin hết hạn
   - `closed_time`: Thời điểm tin chính thức đóng
7. **COMPANY:**
   - `company_id`: ID định danh công ty (khóa ngoại sang `companies.csv`)
   - `company_name`: Tên công ty tuyển dụng
8. **METADATA / OTHER:**
   - `job_id`, `job_posting_url`, `application_url`, `application_type`, `posting_domain`, `sponsored`

---

## 3. Data Quality Issues phát hiện & Cách xử lý

| # | Vấn đề phát hiện | Feature ảnh hưởng | Số record ảnh hưởng | Mức độ | Cách xử lý | Lý do xử lý |
|---|---|---|---|---|---|---|
| 1 | **Lương = 0 hoặc âm** | `med_salary`, `normalized_salary` | 14 trong `postings`<br>16 trong `salaries` | Cao (cho lương) | Gán giá trị về `NaN` (không xóa record) | Đây là Data Error do nhập liệu sai giá trị 0 vào ô mức lương cơ bản. Giữ record để phục vụ mô hình NLP và phân loại Job Level. |
| 2 | **Lệch đơn vị lương (Salary Unit Mismatch)** | `pay_period`, `normalized_salary` | 29 dòng Hourly $\ge 1000$<br>373 dòng Yearly $\le 150$ | Nghiêm trọng (gây méo mó phân phối) | Đổi `pay_period` sang đơn vị thực và tính lại `normalized_salary` | Dòng khai lương $240,000 nhưng chọn nhầm HOURLY (làm lương năm thành 535 triệu USD) hoặc $25 nhưng chọn nhầm YEARLY. Cần sửa về đúng đơn vị. |
| 3 | **Đảo vị trí lương min > max** | `min_salary`, `max_salary` | 0 dòng trong đợt này (đã có rule bảo vệ) | Cao | Hoán đổi lại hai giá trị | Lỗi đảo cột nhập liệu; đảo lại thay vì loại bỏ tin tuyển dụng. |
| 4 | **Khoảng trắng thừa & ký tự đặc biệt trong text** | `title`, `description`, `skills_desc`, `company_name`, `name`, `city`... | 19,593 trong `description`<br>1,977 trong `title`<br>787 trong `company_name` | Trung bình | `\s+` $\rightarrow$ `' '`, strip khoảng trắng đầu cuối, chuyển chuỗi rỗng thành `NaN` | Dọn dẹp văn bản sạch sẽ cho bước trích xuất đặc trưng và embedding NLP. |
| 5 | **Missing giá trị trong `remote_allowed`** | `remote_allowed` | 108,603 (~87.7%) | Trung bình | Gán `NaN` $\rightarrow$ `0`, `1.0` $\rightarrow$ `1` | Trên LinkedIn, chỉ công việc 100% remote mới có cờ 1.0; không có cờ nghĩa là Onsite hoặc Hybrid. |
| 6 | **Missing Target `formatted_experience_level`** | `formatted_experience_level` | 29,409 (~23.75%) | Nghiêm trọng cho Supervised | Giữ nguyên `NaN` trong clean dataset | Cung cấp dữ liệu đầy đủ cho Semi-supervised / Unsupervised; các nhóm Supervised sẽ chủ động lọc ở bước Feature/Modeling. |
| 7 | **Outlier trong `company_size` (giá trị = 7)** | `companies.company_size` | 1,953 công ty | Không phải lỗi | **Giữ nguyên 100%** | Đây là mã quy mô tập đoàn lớn trên LinkedIn (Genuine Extreme Value), không phải lỗi dữ liệu. |
| 8 | **Lượt views & applies cực cao** | `views`, `applies` | Hàng nghìn lượt | Không phải lỗi | **Giữ nguyên 100%** | Độ lệch phải tự nhiên của thị trường tuyển dụng khi có các tin tuyển dụng viral (Genuine Extreme Value). |

---

## 4. Số record trước / sau cleaning

| Bảng dữ liệu | Số dòng Trước | Số dòng Sau | Số dòng Bị loại | Tỷ lệ dữ liệu bảo toàn |
|---|---|---|---|---|
| `postings.csv` | 123,849 | 123,842 | 7 | 99.99% |
| `companies.csv` | 24,473 | 24,473 | 0 | 100% |
| `company_industries.csv` | 24,375 | 24,375 | 0 | 100% |
| `company_specialities.csv` | 169,387 | 169,366 | 21 | 99.99% |
| `job_industries.csv` | 164,808 | 164,808 | 0 | 100% |
| `job_skills.csv` | 213,768 | 213,768 | 0 | 100% |
| `salaries.csv` | 40,785 | 40,785 | 0 | 100% |
| `benefits.csv` | 67,943 | 67,943 | 0 | 100% |

> **Nhận xét:** Tuân thủ nguyên tắc không tự ý xóa bỏ record nếu thông tin cốt lõi vẫn có giá trị sử dụng. Chỉ có 7 bản ghi tin tuyển dụng bị loại bỏ do thiếu hoàn toàn thông tin bắt buộc (`title` hoặc `description`), và 21 bản ghi trùng lặp khóa trong bảng liên kết `company_specialities` được deduplicate. Mọi lỗi dữ liệu (Data Error) khác đều được sửa chữa tận gốc (chuyển về NaN, hoán đổi thứ tự, sửa đơn vị) thay vì drop row.

---

## 5. Column được giữ lại / Khuyến nghị xử lý tiếp theo

- **Giữ toàn bộ các cột nghiệp vụ:**
  - Giữ nguyên toàn bộ 31 cột của `postings` và các cột của 7 bảng phụ trong thư mục `data/processed/`.
- **Khuyến nghị cho nhóm Feature Engineering (`feature/features`):**
  - Cân nhắc loại bỏ hoặc không đưa vào model các cột định danh URL / metadata thuần túy: `job_posting_url`, `application_url`, `posting_domain`, `sponsored` (vì `sponsored` có giá trị đơn nhất = 0 trên 100% bản ghi).
  - Cột `closed_time`: Có tỷ lệ khuyết thiếu 99.13%, chỉ nên sử dụng làm feature cờ nhị phân `is_closed` thay vì tính toán khoảng thời gian chi tiết.
  - Cột `skills_desc`: 98.03% khuyết thiếu trong `postings.csv`. Khuyến nghị kết hợp bảng `job_skills.csv` (213,768 bản ghi đã chuẩn hóa 35 mã kỹ năng) để xây dựng feature kỹ năng toàn diện hơn.

---

## 6. Output bàn giao

1. **Clean Datasets:** `data/processed/*_clean.csv` (8 file tương ứng 8 bảng).
2. **Cleaning Log:** `data/processed/cleaning_log.csv` (ghi lại đầy đủ 33 bước làm sạch).
3. **Mã nguồn hàm làm sạch:** `src/data/cleaning_functions.py` và `src/data/__init__.py`.
4. **Pipeline tự động:** `src/data/clean_data.py`.
5. **Bộ kiểm thử tự động:** `tests/test_data.py` (15/15 tests PASSED).
6. **Notebooks phân tích:**
   - `notebooks/01_data_understanding.ipynb`
   - `notebooks/02_data_cleaning.ipynb`
7. **Báo cáo chuyên sâu:**
   - `docs/data_documentation.md` (tài liệu này)
   - `docs/data_quality_report.md` (báo cáo phân tích chất lượng chi tiết)
