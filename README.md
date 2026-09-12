============================================================
TOPIC: JOB LEVEL ANALYSIS AND PREDICTION BASED ON JOB POSTING CONTENT AND FEATURES
============================================================
PHÂN CHIA NHIỆM VỤ PROJECT DS_JOB_RECOMMEND_PROJECTS
============================================================
MỤC TIÊU PROJECT
---
Phân tích dữ liệu tuyển dụng từ LinkedIn Job Postings để:
1. Hiểu đặc điểm của dữ liệu và thị trường tuyển dụng.
2. Phân tích các yếu tố liên quan đến Job Level.
3. Phân tích mối quan hệ giữa Feature với Feature.
4. Phân tích mối quan hệ giữa Feature với Target.
5. Phân tích các kỹ năng và yêu cầu trong từng Job Level.
6. Xây dựng Feature Engineering từ dữ liệu numerical, categorical và text.
7. Sử dụng NLP để khai thác thông tin từ nội dung Job Posting.
8. Xây dựng Machine Learning Model để dự đoán Job Level.
9. Phân tích Loss Function, Bias và Variance.
10. Phát hiện Underfitting và Overfitting.
11. Đánh giá và so sánh các Machine Learning Model.
12. Giải thích Model và xác định các Feature quan trọng.
13. Xây dựng Pipeline có thể chạy lại từ đầu đến cuối.
14. Đưa ra Business Insights từ dữ liệu và Model.
============================================================
DATA SCIENCE WORKFLOW
============================================================
Business Understanding
```
    ↓
```
Data Collection
```
    ↓
```
Data Understanding
```
    ↓
```
Data Quality Analysis
```
    ↓
```
Data Cleaning
```
    ↓
```
EDA
```
    ↓
```
Feature ↔ Feature Analysis
```
    ↓
```
Feature ↔ Target Analysis
```
    ↓
```
Statistical Analysis
```
    ↓
```
Feature Engineering
```
    ↓
```
NLP / Text Processing
```
    ↓
```
Skill Extraction
```
    ↓
```
Feature Selection
```
    ↓
```
Preprocessing
```
    ↓
```
Modeling
```
    ↓
```
Loss Function
```
    ↓
```
Cross Validation
```
    ↓
```
Bias / Variance Analysis
```
    ↓
```
Hyperparameter Tuning
```
    ↓
```
Evaluation
```
    ↓
```
Model Interpretation
```
    ↓
```
Prediction
```
    ↓
```
Business Insights
============================================================
NGƯỜI 1 — DATA COLLECTION + DATA UNDERSTANDING + DATA CLEANING
============================================================
git clone https://github.com/levantuanem/DS_Job_Recommend_Projects.git
cd DS_Job_Recommend_Projects
git checkout feature/data

* Branch: feature/data
* VAI TRÒ: Data Engineer / Data Analyst
* MỤC TIÊU: Thu thập, kiểm tra, hiểu và làm sạch dữ liệu tuyển dụng để tạo ra dataset có chất lượng, sẵn sàng cho EDA, Feature Engineering và Modeling.
---
1. DATA COLLECTION
---
* Kiểm tra nguồn dataset.
* Đưa dataset gốc vào: data/raw/
* Không chỉnh sửa trực tiếp dữ liệu raw.
* Ghi nhận:
  * Nguồn dữ liệu.
  * Tên dataset.
  * Số lượng record.
  * Số lượng feature.
  * Thời gian / phạm vi dữ liệu nếu có.
---
2. DATA UNDERSTANDING
---
* Kiểm tra:
  * Shape.
  * Số dòng / số cột.
  * Column names.
  * Data types.
  * Missing values.
  * Unique values.
  * Cardinality.
  * Các giá trị bất thường.
* Phân loại feature:
  * JOB INFORMATION:
    * title
    * description
    * skills_desc
    * formatted_experience_level
  * SALARY:
    * min_salary
    * med_salary
    * max_salary
    * normalized_salary
    * currency
    * pay_period
    * compensation_type
  * LOCATION:
    * location
    * zip_code
    * fips
  * WORK:
    * formatted_work_type
    * work_type
    * remote_allowed
  * ENGAGEMENT:
    * views
    * applies
  * TIME:
    * listed_time
    * original_listed_time
    * expiry
    * closed_time
  * COMPANY:
    * company information
---
3. DATA QUALITY ANALYSIS
---
* Kiểm tra:
  * Missing values.
  * Duplicate.
  * Sai datatype.
  * Giá trị không hợp lệ.
  * Category không đồng nhất.
  * Text bị lỗi.
  * Giá trị âm / bằng 0 không hợp lý.
  * Giá trị bất thường.
* Phải ghi rõ:
  * Vấn đề là gì?
  * Có bao nhiêu record bị ảnh hưởng?
  * Feature nào bị ảnh hưởng?
  * Mức độ ảnh hưởng?
  * Cách xử lý?
  * Tại sao chọn cách xử lý đó?
---
4. DATA CLEANING
---
* Xử lý:
  * Missing values.
  * Duplicate.
  * Sai datatype.
  * Giá trị không hợp lệ.
  * Category không đồng nhất.
  * Text bị lỗi.
  * Giá trị không hợp lý.
* Phân biệt:
  * Data Error.
  * Genuine Extreme Value.
* Không tự động xóa dữ liệu chỉ vì đó là outlier.
* Output: data/processed/
---
5. DATA DOCUMENTATION
---
* Ghi lại:
  * Dataset ban đầu.
  * Data Quality Issues.
  * Cleaning methods.
  * Số record trước / sau cleaning.
  * Column được giữ.
  * Column bị loại bỏ.
  * Lý do xử lý.
---
6. OUTPUT
---
* Clean Dataset.
* Data Quality Report.
* Data Understanding Notebook.
* Data Cleaning Notebook.
* Cleaning Functions.
* Dataset Documentation.
---
FOLDER LIÊN QUAN
---
data/raw/
data/processed/
notebooks/
src/data/
tests/test_data.py
============================================================
NGƯỜI 2 — EDA + STATISTICAL ANALYSIS + FEATURE RELATIONSHIP + VISUALIZATION
============================================================
git clone https://github.com/levantuanem/DS_Job_Recommend_Projects.git
cd DS_Job_Recommend_Projects
git checkout feature/visualization

* Branch: feature/visualization
* VAI TRÒ: Data Analyst / Visualization Analyst
* MỤC TIÊU: Phân tích dữ liệu đã clean, tìm ra các mối quan hệ giữa Feature với Feature và Feature với Target, trực quan hóa dữ liệu và đưa ra Business Insights.
---
1. UNIVARIATE ANALYSIS
---
Phân tích từng Feature riêng lẻ.
* NUMERICAL:
  * Salary.
  * Views.
  * Applies.
  * Normalized Salary.
  * Các numerical feature khác.
* Phân tích:
  * Mean.
  * Median.
  * Standard Deviation.
  * Min / Max.
  * Quartile.
  * IQR.
  * Distribution.
  * Skewness.
  * Outlier.
* Visualization:
  * Histogram.
  * KDE.
  * Boxplot.
* CATEGORICAL:
  * Job Level.
  * Work Type.
  * Employment Type.
  * Location.
  * Remote.
  * Pay Period.
* Phân tích:
  * Frequency.
  * Percentage.
  * Cardinality.
  * Category imbalance.
* Visualization:
  * Bar Chart.
  * Count Plot.
  * Percentage Chart.
---
2. FEATURE ↔ FEATURE ANALYSIS
---
Phân tích mối quan hệ giữa các Feature.
* NUMERICAL ↔ NUMERICAL:
  * Salary ↔ Views.
  * Salary ↔ Applies.
  * Views ↔ Applies.
  * Salary ↔ các numerical features.
  * Views ↔ các numerical features.
  * Applies ↔ các numerical features.
* Sử dụng:
  * Pearson Correlation.
  * Spearman Correlation.
* Xác định:
  * Positive Correlation.
  * Negative Correlation.
  * Weak Correlation.
  * Moderate Correlation.
  * Strong Correlation.
* Kiểm tra Multicollinearity.
* Xác định các Feature có thông tin trùng lặp hoặc tương quan quá cao.
* Visualization:
  * Correlation Matrix.
  * Correlation Heatmap.
  * Scatter Plot.
  * Pair Plot nếu phù hợp.
---
3. FEATURE ↔ TARGET ANALYSIS
---
Target: formatted_experience_level
Phân tích mối quan hệ giữa Feature và Job Level.
* NUMERICAL FEATURE ↔ TARGET:
  * Salary ↔ Job Level.
  * Views ↔ Job Level.
  * Applies ↔ Job Level.
  * Normalized Salary ↔ Job Level.
  * Các numerical feature khác ↔ Job Level.
* Phân tích:
  * Mean theo từng Job Level.
  * Median theo từng Job Level.
  * Distribution theo từng Job Level.
  * Difference giữa các nhóm.
* CATEGORICAL FEATURE ↔ TARGET:
  * Work Type ↔ Job Level.
  * Remote ↔ Job Level.
  * Location ↔ Job Level.
  * Employment Type ↔ Job Level.
* TEXT FEATURE ↔ TARGET:
  * Description Length ↔ Job Level.
  * Title Length ↔ Job Level.
  * Skill Count ↔ Job Level.
  * Keyword Frequency ↔ Job Level.
* Có thể sử dụng:
  * ANOVA.
  * Kruskal-Wallis.
  * Chi-square.
  * Effect Size.
  * Group Comparison.
* Mục tiêu:
  * Xác định Feature nào có mối quan hệ với Target.
  * Xác định mức độ khác biệt giữa các Job Level.
  * Xác định Feature nào có khả năng hữu ích cho Modeling.
* Lưu ý:
  Correlation / Association không đồng nghĩa với Causal Relationship.
---
4. CATEGORICAL ↔ NUMERICAL ANALYSIS
---
* Phân tích:
  * Job Level ↔ Salary.
  * Work Type ↔ Salary.
  * Remote ↔ Salary.
  * Location ↔ Salary.
  * Job Level ↔ Views.
  * Job Level ↔ Applies.
* Visualization:
  * Boxplot.
  * Violin Plot.
  * Bar Chart.
  * Grouped Bar Chart.
---
5. CATEGORICAL ↔ CATEGORICAL ANALYSIS
---
* Phân tích:
  * Job Level ↔ Work Type.
  * Job Level ↔ Remote.
  * Job Level ↔ Employment Type.
  * Job Level ↔ Location.
* Sử dụng:
  * Crosstab.
  * Percentage.
  * Chi-square Test nếu phù hợp.
* Visualization:
  * Stacked Bar Chart.
  * Grouped Bar Chart.
---
6. MULTIVARIATE ANALYSIS
---
* Phân tích nhiều Feature cùng lúc.
* Phân tích:
  * Interaction giữa các Feature.
  * Correlation giữa nhiều Numerical Feature.
  * Multicollinearity.
  * Relationship giữa nhiều Feature với Target.
  * Sự khác biệt giữa nhiều nhóm Job Level.
* Visualization:
  * Correlation Heatmap.
  * Pair Plot nếu phù hợp.
  * Grouped Visualization.
---
7. OUTLIER ANALYSIS
---
* Phân tích Outlier trong:
  * Salary.
  * Views.
  * Applies.
  * Các Numerical Feature khác.
* Xác định:
  * Outlier là lỗi dữ liệu hay giá trị thực tế?
  * Có nên Remove không?
  * Có nên Transform không?
* Có thể sử dụng:
  * IQR.
  * Z-score.
  * Log Transformation.
---
8. CLASS IMBALANCE ANALYSIS
---
Target:
formatted_experience_level
* Kiểm tra:
  * Entry Level.
  * Associate.
  * Mid-Senior Level.
  * Director.
  * Executive.
  * Internship.
* Phân tích:
  * Số lượng từng Class.
  * Tỷ lệ từng Class.
  * Mức độ Class Imbalance.
* Visualization:
  * Class Distribution.
  * Percentage Chart.
* Kết quả được bàn giao cho Người 4 để xử lý trong Modeling.
---
9. SKILL ANALYSIS
---
* Phân tích:
  * Skill phổ biến nhất.
  * Skill theo Job Level.
  * Skill theo Location.
  * Skill theo Work Type.
  * Skill nào xuất hiện nhiều ở Senior?
  * Skill nào xuất hiện nhiều ở Director?
  * Skill nào đặc trưng cho Entry Level?
* Nguồn:
  * skills_desc.
  * description.
  * title.
* Người 2 chỉ thực hiện Skill Analysis.
* Người 3 thực hiện Skill Extraction để tạo Feature cho Machine Learning.
---
10. TEXT ANALYSIS
---
* Phân tích:
  * Title Length.
  * Description Length.
  * Word Count.
  * Skill Count.
  * Text Length theo Job Level.
  * Keyword Frequency.
  * Keyword theo Job Level.
* Mục tiêu:
  * Hiểu đặc điểm nội dung Job Posting.
  * Tìm sự khác biệt về Text giữa các Job Level.
---
11. BUSINESS INSIGHTS
---
* Trả lời:
  * Job Level nào tuyển nhiều nhất?
  * Job Level nào có Salary cao nhất?
  * Salary có quan hệ với Job Level không?
  * Feature nào tương quan mạnh với Feature khác?
  * Có Multicollinearity không?
  * Feature nào có quan hệ với Target?
  * Skill nào phổ biến nhất?
  * Skill nào đặc trưng cho từng Job Level?
  * Location nào tuyển nhiều?
  * Location nào có Salary cao?
  * Remote Job tập trung ở Job Level nào?
---
12. OUTPUT
---
* EDA Notebooks.
* Statistical Analysis.
* Feature ↔ Feature Analysis.
* Feature ↔ Target Analysis.
* Correlation Analysis.
* Multicollinearity Analysis.
* Skill Analysis.
* Text Analysis.
* Visualization.
* Business Insights.
---
FOLDER LIÊN QUAN
---
notebooks/
src/visualization/
tests/
============================================================
NGƯỜI 3 — FEATURE ENGINEERING + NLP + SKILL EXTRACTION
============================================================
git clone https://github.com/levantuanem/DS_Job_Recommend_Projects.git
cd DS_Job_Recommend_Projects
git checkout feature/features

* Branch: feature/features
* VAI TRÒ: Feature Engineer / NLP Engineer
* MỤC TIÊU: Chuyển dữ liệu đã clean thành các Feature có thể sử dụng cho Machine Learning.
---
1. NUMERICAL FEATURE ENGINEERING
---
* Xử lý:
  * min_salary.
  * med_salary.
  * max_salary.
  * normalized_salary.
  * views.
  * applies.
* Có thể tạo:
  * salary_range.
  * log_salary.
  * application_rate.
  * view_to_apply_ratio.
* Phải kiểm tra Data Leakage trước khi sử dụng Views / Applies.
---
2. CATEGORICAL FEATURE ENGINEERING
---
* Xử lý:
  * work_type.
  * formatted_work_type.
  * remote_allowed.
  * pay_period.
  * currency.
  * location.
  * application_type.
* Có thể sử dụng:
  * One-Hot Encoding.
  * Frequency Encoding.
  * Ordinal Encoding nếu phù hợp.
---
3. TEXT FEATURE ENGINEERING
---
* Nguồn:
  * title.
  * description.
  * skills_desc.
* Tạo:
  * text_length.
  * word_count.
  * sentence_count.
  * skill_count.
  * keyword_count.
  * technical_keyword_count.
  * management_keyword_count.
  * leadership_keyword_count.
---
4. NLP
---
* Thực hiện:
  * Text Cleaning.
  * Lowercase.
  * Remove Noise.
  * Tokenization.
  * Stopword Removal nếu phù hợp.
  * N-gram.
  * TF-IDF.
* Có thể thử thêm phương pháp NLP khác nếu phù hợp.
* LƯU Ý:
  TF-IDF chỉ là một kỹ thuật NLP.
  NLP không đồng nghĩa với TF-IDF.
---
5. SKILL EXTRACTION
---
* Xác định và trích xuất Skill từ:
  * title.
  * description.
  * skills_desc.
* Có thể xây dựng:
  has_python
  has_sql
  has_java
  has_aws
  has_excel
  has_machine_learning
  has_cloud
  has_leadership
  has_management
  ...
* Skill Extraction phải tạo ra Feature có thể đưa vào Model.
---
6. FEATURE SELECTION
---
* Dựa trên kết quả EDA và Statistical Analysis của Người 2.
* Kiểm tra:
  * Feature hữu ích.
  * Feature dư thừa.
  * Feature tương quan quá cao với nhau.
  * Multicollinearity.
  * Feature gây Data Leakage.
* Có thể sử dụng:
  * Correlation.
  * Variance.
  * Mutual Information.
  * SelectKBest.
  * Feature Importance.
---
7. DATA LEAKAGE PREVENTION
---
* Kiểm tra:
  * Target Leakage.
  * Train/Test Leakage.
  * Information Leakage.
  * Feature được tạo từ thông tin chỉ có sau thời điểm dự đoán.
* Đặc biệt kiểm tra:
  * views.
  * applies.
  * các biến phát sinh sau khi Job Posting được đăng.
---
8. PREPROCESSING
---
* Xử lý:
  * Scaling.
  * Encoding.
  * Imputation nếu cần.
  * Text Vectorization.
* Xây dựng Preprocessing Pipeline.
* Đảm bảo:
  Training Data và Test Data
  sử dụng cùng preprocessing logic nhưng chỉ Fit trên Training Data.
---
9. OUTPUT
---
* Numerical Features.
* Categorical Features.
* Text Features.
* Skill Features.
* NLP Features.
* Feature Selection Results.
* Preprocessing Pipeline.
* X.
* y.
* Feature Specification.
---
FOLDER LIÊN QUAN
---
src/features/
configs/features.yaml
tests/test_features.py
notebooks/
============================================================
NGƯỜI 4 — MODELING + LOSS FUNCTION + BIAS / VARIANCE + EVALUATION
============================================================
git clone https://github.com/levantuanem/DS_Job_Recommend_Projects.git
cd DS_Job_Recommend_Projects
git checkout feature/model

* Branch: feature/model
* VAI TRÒ: Machine Learning Engineer
* MỤC TIÊU: Xây dựng, tối ưu, đánh giá và giải thích Machine Learning Model dùng để dự đoán Job Level.
---
1. TARGET
---
Target:
formatted_experience_level
Các Class:
```
- Entry Level.
- Associate.
- Mid-Senior Level.
- Director.
- Executive.
- Internship.
```
---
2. BASELINE MODEL
---
* Xây dựng Baseline Model.
* Có thể sử dụng:
  * Logistic Regression.
* Mục đích:
  * Tạo baseline performance.
  * Làm mốc so sánh với các Model phức tạp hơn.
---
3. MACHINE LEARNING MODELS
---
* Thử nghiệm:
  * Logistic Regression.
  * Random Forest.
  * Linear SVM.
  * Gradient Boosting.
  * XGBoost / LightGBM nếu phù hợp.
* So sánh các Model trên cùng Evaluation Strategy.
---
4. LOSS FUNCTION
---
* Phân tích Loss Function phù hợp với bài toán Multiclass Classification.
* Xem xét:
  * Cross-Entropy Loss / Log Loss.
  * Weighted Cross-Entropy nếu sử dụng Class Weight.
* Phân tích:
  * Loss dùng để tối ưu Model.
  * Ý nghĩa của Loss.
  * Training Loss.
  * Validation Loss nếu Model hỗ trợ.
  * Sự thay đổi của Loss trong quá trình Training
* Sử dụng Loss để hỗ trợ phân tích:
  * Underfitting.
  * Overfitting.
---
5. TRAINING
---
* Train/Test Split.
* Cross Validation.
* Training.
* Model Comparison.
* Đảm bảo Preprocessing chỉ được Fit trên Training Data.
* Tránh Data Leakage.
---
6. CLASS IMBALANCE
---
* Dựa trên kết quả Class Imbalance Analysis của Người 2.
* Có thể thử:
  * class_weight.
  * Random Over Sampling.
  * SMOTE nếu phù hợp.
* Đảm bảo Sampling chỉ được thực hiện trên Training Data.
---
7. CROSS VALIDATION
---
* Sử dụng Cross Validation để:
  * Đánh giá độ ổn định của Model.
  * So sánh các Model.
  * Phát hiện Model có Variance cao.
  * Hỗ trợ Hyperparameter Tuning.
* Theo dõi:
  * Mean score.
  * Standard deviation.
  * Score giữa các Fold.
---
8. HYPERPARAMETER TUNING
---
* Sử dụng:
  * GridSearchCV.
  * RandomizedSearchCV.
* Tối ưu:
  * Regularization.
  * Tree Depth.
  * Number of Estimators.
  * Learning Rate.
  * Các Hyperparameter phù hợp với Model.
---
9. BIAS ANALYSIS
---
Phân tích High Bias / Low Bias.
* So sánh:
  * Training Performance.
  * Validation Performance.
  * Test Performance.
* Dấu hiệu High Bias:
  * Training Performance thấp.
  * Validation Performance thấp.
  * Test Performance thấp.
* Kết luận:
  High Bias → Underfitting.
---
10. VARIANCE ANALYSIS
---
Phân tích High Variance / Low Variance.
* So sánh:
  * Training Performance.
  * Validation Performance.
  * Test Performance.
  * Cross Validation Scores.
  * Training Loss.
  * Validation Loss nếu có.
* Dấu hiệu High Variance:
  * Training Performance rất cao.
  * Validation / Test Performance thấp.
  * Training Loss thấp nhưng Validation Loss cao.
* Kết luận:
  High Variance → Overfitting.
---
11. UNDERFITTING / OVERFITTING
---
* Phân tích:
  * Underfitting.
  * Good Fit.
  * Overfitting.
* Dựa trên:
  * Training Score.
  * Validation Score.
  * Test Score.
  * Training Loss.
  * Validation Loss.
  * Cross Validation.
* Đề xuất hướng xử lý:
  Underfitting:
  ```
    - Model phức tạp hơn.
    - Feature tốt hơn.
    - Giảm Regularization nếu phù hợp.
  ```
  Overfitting:
  ```
    - Regularization.
    - Giảm Model Complexity.
    - Feature Selection.
    - Cross Validation.
    - More Data nếu có thể.
  ```
---
12. EVALUATION
---
* Không chỉ sử dụng Accuracy.
* Đánh giá:
  * Accuracy.
  * Precision.
  * Recall.
  * F1-score.
  * Macro-F1.
  * Weighted-F1.
  * Confusion Matrix.
* Với Class Imbalance:
  Ưu tiên Macro-F1.
---
13. MODEL COMPARISON
---
Tạo bảng:
```
Model
Accuracy
Precision
Recall
Macro-F1
Weighted-F1
Training Time
CV Mean
CV Std
```
* Chọn Best Model dựa trên Metric phù hợp với Business Problem.
---
14. MODEL INTERPRETATION
---
Trả lời:
"Model dựa vào những Feature nào để xác định Job Level?"
* Có thể sử dụng:
  * Feature Importance.
  * Permutation Importance.
  * SHAP.
  * Coefficient Analysis.
* Phân tích:
  * Numerical Feature quan trọng.
  * Categorical Feature quan trọng.
  * Skill quan trọng.
  * Keyword quan trọng.
  * Text Feature quan trọng.
---
15. MODEL ARTIFACT
---
* Lưu:
  * Best Model.
  * Vectorizer.
  * Encoder.
  * Scaler nếu cần.
  * Label Mapping.
* Đảm bảo các Artifact có thể sử dụng lại cho Prediction.
---
16. OUTPUT
---
* Baseline Results.
* Model Comparison.
* Best Model.
* Loss Analysis.
* Bias Analysis.
* Variance Analysis.
* Underfitting / Overfitting Analysis.
* Evaluation Report.
* Confusion Matrix.
* Hyperparameter Results.
* Feature Importance.
* Model Interpretation.
* Model Artifacts.
---
FOLDER LIÊN QUAN
---
src/models/
models/
configs/model.yaml
tests/test_model.py
notebooks/
============================================================
NGƯỜI 5 — INTEGRATION + PIPELINE + TESTING + DOCUMENTATION
============================================================
git clone https://github.com/levantuanem/DS_Job_Recommend_Projects.git
cd DS_Job_Recommend_Projects
git checkout feature/pipeline

* Branch: feature/pipeline
* VAI TRÒ: Data Science / ML Engineer / Project Integration
* MỤC TIÊU: Kết nối toàn bộ phần việc của Team thành một Project hoàn chỉnh, có cấu trúc rõ ràng, có thể chạy lại và kiểm thử.
---
1. PROJECT ARCHITECTURE
---
Thiết kế và duy trì cấu trúc:
```
src/
├── data/
├── features/
├── visualization/
├── models/
├── pipeline/
└── utils/
```
---
2. INTEGRATION
---
Kết nối:
```
Data
↓
Cleaning
↓
EDA
↓
Feature Engineering
↓
NLP
↓
Preprocessing
↓
Modeling
↓
Evaluation
↓
Prediction
```
* Đảm bảo module của Người 1, Người 2, Người 3 và Người 4 hoạt động thống nhất.
---
3. PIPELINE
---
Xây dựng Pipeline:
```
Raw Data
    ↓
Validation
    ↓
Cleaning
    ↓
Feature Engineering
    ↓
NLP
    ↓
Preprocessing
    ↓
Train/Test Split
    ↓
Training
    ↓
Evaluation
    ↓
Save Model
    ↓
Prediction
```
---
4. CONFIGURATION
---
configs/
```
├── data.yaml
├── features.yaml
├── model.yaml
└── pipeline.yaml
```
* data.yaml:
  * Data Path.
  * Cleaning Parameters.
  * Split Parameters.
* features.yaml:
  * Feature Parameters.
  * NLP Parameters.
  * TF-IDF Parameters.
* model.yaml:
  * Model Parameters.
  * Hyperparameters.
  * Evaluation Parameters.
* pipeline.yaml:
  * Pipeline Settings.
  * Output Paths.
  * Experiment Settings.
* Người 5 chịu trách nhiệm tích hợp Config.
* Các thành viên khác chịu trách nhiệm cung cấp Parameter liên quan đến module của mình.
---
5. UTILS
---
src/utils/
```
- config_loader.py
- logger.py
- random_seed.py
- file_utils.py
```
* Chứa Utility dùng chung cho toàn Project.
---
6. TESTING
---
Kiểm tra:
```
- Data Module.
- Feature Module.
- Visualization Module nếu cần.
- Model Module.
- Pipeline.
```
* Cấu trúc:
  tests/
  ├── test_data.py
  ├── test_features.py
  ├── test_model.py
  └── test_pipeline.py
* Mỗi thành viên chịu trách nhiệm viết Unit Test cho phần code của mình.
* Người 5 chịu trách nhiệm Integration Test.
---
7. GIT / TEAM INTEGRATION
---
* Không commit trực tiếp vào main.
* Mỗi thành viên làm việc trên Branch riêng.
* Branch được phát triển từ develop.
* Quy trình:
  feature/xxx
  ```
  ↓
  ```
  Pull Request
  ```
  ↓
  ```
  develop
  ```
  ↓
  ```
  Test
  ```
  ↓
  ```
  main
* Kiểm tra:
  * Merge Conflict.
  * Dependency.
  * Code Quality.
  * Project chạy được sau Merge.
---
8. EXPERIMENT MANAGEMENT
---
* Quản lý:
  * Dataset Version.
  * Feature Version.
  * Model Version.
  * Configuration.
  * Evaluation Results.
* Đảm bảo có thể biết:
  * Model nào.
  * Feature nào.
  * Config nào.
  * Dataset nào.
  tạo ra kết quả nào.
---
9. PREDICTION
---
Tạo:
```
scripts/
├── train.py
└── predict.py
```
* Input:
  Job Posting mới.
* Pipeline:
  Text Preprocessing
  ```
    ↓
  ```
  Feature Engineering
  ```
    ↓
  ```
  NLP
  ```
    ↓
  ```
  Preprocessing
  ```
    ↓
  ```
  Model
  ```
    ↓
  ```
  Prediction
* Output:
  Predicted Job Level
  *
  Probability nếu Model hỗ trợ.
---
10. DOCUMENTATION
---
README.md phải mô tả:
```
- Business Problem.
- Dataset.
- Data Understanding.
- Data Cleaning.
- EDA.
- Feature ↔ Feature Analysis.
- Feature ↔ Target Analysis.
- Statistical Analysis.
- Feature Engineering.
- NLP.
- Skill Extraction.
- Modeling.
- Loss Function.
- Bias.
- Variance.
- Underfitting.
- Overfitting.
- Evaluation.
- Model Interpretation.
- Business Insights.
- How to Run Project.
- How to Train Model.
- How to Make Prediction.
```
---
11. FINAL INTEGRATION TEST
---
Kiểm tra Project từ đầu đến cuối:
```
Raw Dataset
    ↓
Data Cleaning
    ↓
EDA
    ↓
Feature ↔ Feature Analysis
    ↓
Feature ↔ Target Analysis
    ↓
Feature Engineering
    ↓
NLP
    ↓
Training
    ↓
Loss Analysis
    ↓
Evaluation
    ↓
Bias / Variance Analysis
    ↓
Model Interpretation
    ↓
Prediction
```
* Đảm bảo Project có thể chạy lại từ đầu.
* Kiểm tra:
  * Dependency.
  * File Path.
  * Config.
  * Model Artifact.
  * Preprocessing.
  * Prediction.
  * Test.
============================================================
Ý NGHĨA CÁC FOLDER
============================================================
.vscode/
```
→ Cấu hình VS Code của Project.
```
configs/
```
→ Configuration của Project.
```
data/
```
→ Dữ liệu.
```
data/raw/
```
→ Dữ liệu gốc.
```
data/interim/
```
→ Dữ liệu trung gian.
```
data/processed/
```
→ Dữ liệu đã xử lý.
```
models/
```
→ Model và Model Artifacts.
```
notebooks/
```
→ Exploration, EDA, Analysis và Experiment.
```
scripts/
```
→ Script chạy các tác vụ độc lập.
```
src/
```
→ Source Code chính.
```
src/data/
```
→ Data Collection, Validation, Cleaning.
```
src/features/
```
→ Feature Engineering và NLP.
```
src/visualization/
```
→ Visualization và EDA functions.
```
src/models/
```
→ Training, Evaluation, Prediction, Interpretation.
```
src/pipeline/
```
→ Điều phối các module thành Workflow hoàn chỉnh.
```
src/utils/
```
→ Utility dùng chung.
```
tests/
```
→ Unit Test và Integration Test.
```
============================================================
NGUYÊN TẮC PHÂN CHIA TRÁCH NHIỆM
============================================================
1. Người 1 chịu trách nhiệm:
   Data Collection
   *
   Data Understanding
   *
   Data Quality
   *
   Data Cleaning
---
2. Người 2 chịu trách nhiệm:
   EDA
   *
   Statistical Analysis
   *
   Feature ↔ Feature
   *
   Feature ↔ Target
   *
   Correlation
   *
   Multicollinearity
   *
   Skill Analysis
   *
   Text Analysis
   *
   Visualization
   *
   Business Insights
---
3. Người 3 chịu trách nhiệm:
   Feature Engineering
   *
   NLP
   *
   Skill Extraction
   *
   Feature Selection
   *
   Preprocessing
   *
   Data Leakage Prevention
---
4. Người 4 chịu trách nhiệm:
   Modeling
   *
   Loss Function
   *
   Class Imbalance
   *
   Cross Validation
   *
   Hyperparameter Tuning
   *
   Bias
   *
   Variance
   *
   Underfitting
   *
   Overfitting
   *
   Evaluation
   *
   Feature Importance
   *
   Model Interpretation
---
5. Người 5 chịu trách nhiệm:
   Integration
   *
   Pipeline
   *
   Configuration
   *
   Utils
   *
   Testing
   *
   Experiment Management
   *
   Prediction
   *
   Documentation
   *
   Git Integration
============================================================
NGUYÊN TẮC QUAN TRỌNG
============================================================
1. Không commit trực tiếp vào main.
2. Mỗi thành viên làm việc trên Branch riêng.
3. Các Branch được phát triển từ develop.
4. Không tự ý thay đổi logic của module do thành viên khác phụ trách.
5. Notebook dùng cho Exploration, EDA và Experiment.
6. Logic có thể tái sử dụng phải đưa vào src/.
7. Không hard-code các Parameter quan trọng.
8. Parameter có thể thay đổi nên đưa vào configs/.
9. Không được chỉ tập trung vào Model.
10. Project phải bao gồm:
    Data Collection
    *
    Data Cleaning
    *
    EDA
    *
    Feature ↔ Feature
    *
    Feature ↔ Target
    *
    Statistical Analysis
    *
    Feature Engineering
    *
    NLP
    *
    Skill Extraction
    *
    Modeling
    *
    Loss Function
    *
    Bias / Variance
    *
    Evaluation
    *
    Model Interpretation
    *
    Prediction
    *
    Business Insights
11. Feature ↔ Feature dùng để hiểu:
    * Correlation.
    * Redundancy.
    * Multicollinearity.
12. Feature ↔ Target dùng để hiểu:
    * Feature nào có mối quan hệ với Job Level.
    * Mức độ khác biệt giữa các nhóm Target.
    * Feature nào có khả năng hữu ích cho Modeling.
13. Skill Analysis và Skill Extraction phải được phân biệt:
    Skill Analysis:
    ```
    Phân tích Skill để tìm Business Insight.
    ```
    Skill Extraction:
    ```
    Chuyển Skill thành Feature cho Machine Learning.
    ```
14. Text Analysis và NLP phải được phân biệt:
    Text Analysis:
    ```
    Phân tích đặc điểm Text.
    ```
    NLP:
    ```
    Xử lý và chuyển Text thành Feature cho Machine Learning.
    ```
15. Correlation không đồng nghĩa với Causation.
16. Không được đánh giá Model chỉ bằng Accuracy.
17. Khi Class Imbalance đáng kể, phải chú ý Macro-F1.
18. Phải phân tích Bias và Variance.
19. Phải kiểm tra Underfitting và Overfitting.
20. Phải kiểm tra Data Leakage trước khi Training.
============================================================
MỤC TIÊU CUỐI CÙNG
============================================================
JOB POSTINGS
```
  ↓
```
DATA UNDERSTANDING
```
  ↓
```
DATA CLEANING
```
  ↓
```
EDA
```
  ↓
```
FEATURE ↔ FEATURE
```
  ↓
```
FEATURE ↔ TARGET
```
  ↓
```
STATISTICAL ANALYSIS
```
  ↓
```
FEATURE ENGINEERING
```
  ↓
```
NLP + SKILL EXTRACTION
```
  ↓
```
FEATURE SELECTION
```
  ↓
```
PREPROCESSING
```
  ↓
```
MACHINE LEARNING
```
  ↓
```
LOSS FUNCTION
```
  ↓
```
BIAS / VARIANCE
```
  ↓
```
UNDERFITTING / OVERFITTING
```
  ↓
```
MODEL EVALUATION
```
  ↓
```
MODEL INTERPRETATION
```
  ↓
```
JOB LEVEL PREDICTION
```
  ↓
```
BUSINESS INSIGHTS
============================================================
PHÂN CHIA BRANCH CUỐI CÙNG
============================================================
develop

│

├── feature/data

│   → Người 1

│   → Data Collection + Data Understanding + Data Cleaning

│

├── feature/visualization

│   → Người 2

│   → EDA + Statistical Analysis + Feature Relationship + Visualization

│

├── feature/features

│   → Người 3

│   → Feature Engineering + NLP + Skill Extraction + Preprocessing

│

├── feature/model

│   → Người 4

│   → Modeling + Loss Function + Bias/Variance + Evaluation

│

└── feature/pipeline
```
→ Người 5
→ Integration + Pipeline + Testing + Documentation
```
============================================================
# GIT TEAMWORK POLICY
## DS_Job_Recommend_Projects
============================================================
## 1. MỤC ĐÍCH
Policy này quy định cách sử dụng Git và GitHub trong quá trình phát triển project `DS_Job_Recommend_Projects`.
Mục tiêu:
* Phân chia code rõ ràng giữa các thành viên.
* Tránh xung đột code không cần thiết.
* Đảm bảo `develop` luôn ở trạng thái có thể chạy.
* Kiểm soát việc Merge thông qua Pull Request.
* Duy trì Git History rõ ràng.
* Đảm bảo mỗi thành viên chịu trách nhiệm với module của mình.
* Có thể truy xuất quá trình phát triển của từng thành viên.
============================================================
## 2. BRANCHING STRATEGY
Project sử dụng mô hình:
```text
main
  ↑
develop
  ↑
feature/*
```
Các branch chính:
```text
develop
│
├── feature/data
├── feature/visualization
├── feature/features
├── feature/model
└── feature/pipeline
```
Phân công:
```text
feature/data
→ Người 1
→ Data Collection + Data Understanding + Data Cleaning
feature/visualization
→ Người 2
→ EDA + Statistical Analysis + Feature Relationship + Visualization
feature/features
→ Người 3
→ Feature Engineering + NLP + Skill Extraction + Preprocessing
feature/model
→ Người 4
→ Modeling + Loss Function + Bias/Variance + Evaluation
feature/pipeline
→ Người 5
→ Integration + Pipeline + Testing + Documentation
```
============================================================
## 3. BRANCH RESPONSIBILITY
Mỗi thành viên chỉ phát triển code chính trên branch được phân công.
```text
Người 1 → feature/data
Người 2 → feature/visualization
Người 3 → feature/features
Người 4 → feature/model
Người 5 → feature/pipeline
```
Không tự ý sử dụng branch của thành viên khác để phát triển chức năng thuộc trách nhiệm của mình.
Không merge trực tiếp giữa các `feature/*` branch.
Ví dụ KHÔNG thực hiện:
```text
feature/data
      ↓
feature/model
```
hoặc:
```text
feature/features
      ↓
feature/data
```
Các Feature Branch chỉ được Merge vào:
```text
feature/* → develop
```
============================================================
## 4. QUY TẮC MAIN VÀ DEVELOP
### MAIN
`main` là branch ổn định của project.
Không được:
```text
git push origin main
```
trực tiếp trong quá trình phát triển thông thường.
`main` chỉ nhận code từ `develop` sau khi project đã được kiểm tra và xác nhận ổn định.
### DEVELOP
`develop` là branch tích hợp của toàn team.
Mọi Feature Branch sau khi hoàn thành phải tạo Pull Request:
```text
feature/*
    ↓
Pull Request
    ↓
develop
```
Không commit trực tiếp vào `develop`.
============================================================
## 5. WORKFLOW TRƯỚC KHI CODE
Trước khi bắt đầu làm việc, thành viên phải chuyển về đúng branch của mình:
```bash
git checkout feature/xxx
```
Sau đó cập nhật thông tin từ remote:
```bash
git fetch origin
```
Kiểm tra trạng thái:
```bash
git status
```
Đảm bảo đang ở đúng branch trước khi bắt đầu code.
Ví dụ Người 3:
```bash
git checkout feature/features
git fetch origin
git status
```
Không được bắt đầu code khi chưa kiểm tra branch hiện tại.
============================================================
## 6. QUY TẮC CODE
Mỗi thành viên chỉ thay đổi các file thuộc phạm vi task của mình nếu không có sự thống nhất với team.
Ví dụ:
```text
feature/data
→ src/data/
feature/visualization
→ src/visualization/
feature/features
→ src/features/
feature/model
→ src/models/
feature/pipeline
→ src/pipeline/
```
Nếu cần thay đổi file thuộc module của thành viên khác, phải trao đổi trước với thành viên phụ trách module đó.
Không tự ý sửa hoặc xóa code của thành viên khác.
============================================================
## 7. QUY TẮC KIỂM TRA TRƯỚC KHI COMMIT
Sau khi hoàn thành một task hoặc một phần task, phải kiểm tra:
```bash
git status
```
Sau đó kiểm tra nội dung thay đổi:
```bash
git diff
```
Mục đích:
* Kiểm tra file đã thay đổi.
* Phát hiện file không liên quan.
* Phát hiện code ngoài phạm vi task.
* Tránh commit nhầm dataset.
* Tránh commit file môi trường.
* Tránh commit file cấu hình cá nhân.
Chỉ commit những file thực sự cần thiết.
============================================================
## 8. QUY TẮC COMMIT
Commit phải có message rõ ràng và mô tả đúng nội dung thay đổi.
Cấu trúc:
```text
type: description
```
Các loại commit:
```text
feat:      Thêm chức năng mới
fix:       Sửa lỗi
refactor:  Refactor code
docs:      Documentation
test:      Unit Test / Integration Test
eda:       EDA / Data Analysis
model:     Machine Learning Model
pipeline:  Pipeline / Integration
config:    Configuration
```
Ví dụ:
```bash
git commit -m "feat: add text preprocessing"
```
```bash
git commit -m "eda: analyze salary distribution"
```
```bash
git commit -m "model: add random forest baseline"
```
```bash
git commit -m "pipeline: integrate training workflow"
```
```bash
git commit -m "test: add feature engineering tests"
```
Commit message phải:
* Ngắn gọn.
* Rõ nghĩa.
* Phản ánh đúng thay đổi.
* Không sử dụng message chung chung như:
```text
update
fix
test
abc
change
final
done
```
============================================================
## 9. QUY TẮC COMMIT
Không nên gom quá nhiều thay đổi không liên quan vào một commit.
Không nên:
```text
commit:
"update project"
```
nhưng bên trong có:
```text
EDA
+
Feature Engineering
+
Model
+
Pipeline
+
README
```
Nên chia thành các commit có ý nghĩa:
```text
eda: analyze salary distribution
feat: add text preprocessing
model: add random forest baseline
pipeline: integrate training workflow
docs: update project documentation
```
Mỗi commit nên đại diện cho một thay đổi logic tương đối rõ ràng.
============================================================
## 10. QUY TẮC PUSH
Sau khi commit:
```bash
git push origin feature/xxx
```
Ví dụ:
```bash
git push origin feature/features
```
Không push code chưa được kiểm tra.
Trước khi push phải đảm bảo:
```text
Code
 ↓
git status
 ↓
git diff
 ↓
Test
 ↓
git add
 ↓
git commit
 ↓
git push
```
============================================================
## 11. PULL REQUEST POLICY
Sau khi hoàn thành task, thành viên tạo Pull Request:
```text
feature/xxx
      ↓
Pull Request
      ↓
develop
```
Pull Request phải có:
### Title
Mô tả ngắn gọn chức năng đã thực hiện.
Ví dụ:
```text
feat: implement feature engineering
```
### Description
Mô tả:
```text
- Implement numerical features
- Implement categorical features
- Implement text features
- Implement skill extraction
- Add preprocessing
- Add tests
```
PR phải cho biết:
* Đã làm gì?
* Thay đổi file/module nào?
* Có thêm Feature nào?
* Có thay đổi Model không?
* Có thay đổi Configuration không?
* Đã test chưa?
* Có vấn đề nào cần reviewer chú ý không?
============================================================
## 12. PULL REQUEST REVIEW
Không Merge Pull Request ngay sau khi tạo.
Ít nhất một thành viên khác phải Review trước khi Merge.
Reviewer kiểm tra:
* Code có đúng task không?
* Logic có đúng không?
* Có Data Leakage không?
* Có Hard-code không?
* Có ảnh hưởng module khác không?
* Có lỗi không?
* Có Test không?
* Có commit file không cần thiết không?
* Code có tuân thủ cấu trúc project không?
Nếu có lỗi:
```text
Pull Request
      ↓
Review
      ↓
Changes Requested
      ↓
Developer sửa
      ↓
Push
      ↓
Review lại
```
Nếu đạt:
```text
Review Approved
      ↓
Merge
      ↓
develop
```
============================================================
## 13. QUY TẮC SAU KHI PR ĐƯỢC MERGE
Ví dụ Người 3 đã Merge:
```text
feature/features
        ↓
     develop
```
Lúc này `develop` đã có code mới.
Các thành viên khác cần cập nhật `develop` khi cần lấy những thay đổi mới:
```bash
git checkout develop
git pull origin develop
```
Sau đó quay lại branch của mình:
```bash
git checkout feature/data
```
Nếu branch cần lấy thay đổi mới từ `develop`:
```bash
git merge develop
```
Team sử dụng `merge` thay vì `rebase` trong giai đoạn phát triển để workflow dễ hiểu và hạn chế thao tác làm thay đổi Git History.
============================================================
## 14. CẬP NHẬT DEVELOP VÀO FEATURE BRANCH
Khi `develop` có nhiều thay đổi mới và Feature Branch của thành viên cần sử dụng các thay đổi đó:
```bash
git checkout develop
git pull origin develop
git checkout feature/xxx
git merge develop
```
Sau khi Merge cần kiểm tra:
```bash
git status
```
và chạy Test.
Nếu có Conflict:
```text
Conflict
   ↓
Xác định file
   ↓
Trao đổi với owner của module
   ↓
Resolve Conflict
   ↓
Test
   ↓
Commit
```
Không được tự ý chọn một phía của Conflict nếu không hiểu thay đổi của module đó.
============================================================
## 15. CONFLICT POLICY
Khi xảy ra Merge Conflict:
1. Không panic hoặc xóa toàn bộ code.
2. Kiểm tra:
```bash
git status
```
3. Xác định file bị Conflict.
4. Đọc cả hai phiên bản.
5. Trao đổi với owner của module nếu cần.
6. Resolve Conflict thủ công.
7. Chạy Test.
8. Kiểm tra:
```bash
git diff
```
9. Commit sau khi xác nhận code đúng.
Đặc biệt không được dùng:
```text
Accept Current Change
```
hoặc:
```text
Accept Incoming Change
```
một cách máy móc.
============================================================
## 16. .GITIGNORE POLICY
Project không commit các file / folder không cần thiết.
Các file thường phải Ignore:
```text
.venv/
venv/
env/
__pycache__/
*.pyc
.ipynb_checkpoints/
.vscode/
.env
models/*.pkl
models/*.joblib
```
Dataset raw lớn không được tùy tiện commit vào GitHub.
Nếu cần giữ cấu trúc thư mục:
```text
data/
├── raw/
│   └── .gitkeep
├── interim/
│   └── .gitkeep
└── processed/
    └── .gitkeep
```
Không commit:
* Password.
* API Key.
* Token.
* Secret.
* Environment Variable chứa thông tin nhạy cảm.
* File cá nhân.
* File tạm.
* Cache.
* Virtual Environment.
============================================================
## 17. NOTEBOOK POLICY
Notebook được sử dụng cho:
```text
Exploration
EDA
Visualization
Experiment
Model Experiment
Analysis
```
Notebook không nên trở thành nơi chứa toàn bộ Production Logic.
Logic có khả năng tái sử dụng phải đưa vào:
```text
src/
```
Ví dụ:
```text
notebooks/
└── 03_feature_engineering.ipynb
```
có thể dùng:
```text
src/features/
├── numerical.py
├── categorical.py
├── text.py
└── skill_extraction.py
```
Notebook chỉ gọi các module cần thiết.
============================================================
## 18. DATA POLICY
Dữ liệu được phân chia:
```text
data/raw/
→ Original Dataset
data/interim/
→ Intermediate Dataset
data/processed/
→ Clean / Processed Dataset
```
Không chỉnh sửa trực tiếp dữ liệu trong:
```text
data/raw/
```
Mọi thay đổi dữ liệu phải thông qua code hoặc notebook có thể tái hiện.
Nếu Dataset được xử lý:
```text
Raw
 ↓
Cleaning
 ↓
Processed
```
phải có khả năng giải thích:
* Dữ liệu thay đổi như thế nào?
* Vì sao thay đổi?
* Bao nhiêu record bị ảnh hưởng?
============================================================
## 19. MODULE OWNERSHIP POLICY
Mỗi module có một owner chính.
```text
src/data/
→ Người 1
src/visualization/
→ Người 2
src/features/
→ Người 3
src/models/
→ Người 4
src/pipeline/
→ Người 5
```
Owner chịu trách nhiệm:
* Code chính.
* Review thay đổi liên quan module.
* Test module.
* Documentation module.
* Giải quyết Conflict liên quan module.
Các thành viên khác có thể đóng góp nhưng phải trao đổi với owner.
============================================================
## 20. TESTING POLICY
Trước khi tạo Pull Request phải chạy Test liên quan.
Cấu trúc:
```text
tests/
├── test_data.py
├── test_features.py
├── test_model.py
└── test_pipeline.py
```
Phân công:
```text
Người 1 → test_data.py
Người 3 → test_features.py
Người 4 → test_model.py
Người 5 → test_pipeline.py
```
Người 2 bổ sung test cho các Visualization / Analysis Function nếu có logic cần kiểm thử.
Người 5 chịu trách nhiệm kiểm tra Integration.
============================================================
## 21. GIT HISTORY POLICY
Git History phải thể hiện được quá trình phát triển thực tế của team.
Ví dụ:
```text
feat: clean raw dataset
eda: analyze feature target relationship
feat: implement tfidf features
model: train random forest baseline
pipeline: integrate training workflow
```
Không để một thành viên commit toàn bộ code của các thành viên khác.
Không dùng:
```text
git config user.name
```
để giả danh thành viên khác.
Mỗi thành viên phải sử dụng Git identity của chính mình.
============================================================
## 22. CODE OWNERSHIP VÀ RESPONSIBILITY
```text
Người 1
→ Data
Người 2
→ EDA / Statistics / Visualization
Người 3
→ Features / NLP
Người 4
→ Machine Learning
Người 5
→ Pipeline / Integration
```
Git History phải phản ánh trách nhiệm này.
Mục tiêu là khi nhìn vào Repository có thể xác định:
```text
Ai làm?
↓
Làm module nào?
↓
Thay đổi gì?
↓
Commit nào?
↓
Pull Request nào?
```
============================================================
## 23. QUY TRÌNH TEAMWORK CHUẨN
Quy trình đầy đủ:
```text
START
  ↓
Checkout đúng feature branch
  ↓
Fetch remote
  ↓
Code
  ↓
Test
  ↓
git status
  ↓
git diff
  ↓
git add
  ↓
git commit
  ↓
git push
  ↓
Create Pull Request
  ↓
Code Review
  ↓
Changes Required?
  ├── YES → Fix → Push → Review lại
  │
  └── NO
       ↓
     Merge
       ↓
    develop
       ↓
 Integration Test
       ↓
    Stable?
       ├── NO → Fix
       │
       └── YES
            ↓
           main
```
============================================================
## 24. QUY TẮC KHÔNG ĐƯỢC VI PHẠM
1. Không push trực tiếp vào `main`.
2. Không commit trực tiếp vào `develop`.
3. Không merge Feature Branch trực tiếp vào Feature Branch khác.
4. Không tự ý sửa module của thành viên khác.
5. Không commit Dataset lớn nếu chưa thống nhất.
6. Không commit `.venv/`.
7. Không commit API Key / Password / Token.
8. Không commit file `.pkl` / `.joblib` nếu đã được quy định Ignore.
9. Không commit code chưa kiểm tra.
10. Không tạo Pull Request mà chưa Test.
11. Không Merge Pull Request khi chưa Review.
12. Không Resolve Conflict một cách máy móc.
13. Không sử dụng Commit Message chung chung.
14. Không squash hoặc rewrite History của branch người khác nếu chưa thống nhất.
15. Không sử dụng Git để giả danh thành viên khác.
============================================================
## 25. QUICK REFERENCE
### BẮT ĐẦU LÀM VIỆC
```bash
git checkout feature/xxx
git fetch origin
git status
```
### SAU KHI CODE
```bash
git status
git diff
```
### COMMIT
```bash
git add .
git commit -m "type: description"
```
### PUSH
```bash
git push origin feature/xxx
```
### SAU KHI PR MERGE
```bash
git checkout develop
git pull origin develop
git checkout feature/xxx
git merge develop
```
### KIỂM TRA BRANCH
```bash
git branch
```
### KIỂM TRA REMOTE
```bash
git branch -a
```
### KIỂM TRA LỊCH SỬ
```bash
git log --oneline --graph --all --decorate
```
============================================================
## 26. NGUYÊN TẮC CỐT LÕI
Team thống nhất 5 nguyên tắc:
```text
1. Mỗi người một Feature Branch.
2. Feature Branch → Pull Request → develop.
3. Develop → Test → main.
4. Code của ai người đó chịu trách nhiệm.
5. Không Merge khi chưa Review và Test.
```
Workflow chính thức của Project:
```text
             MAIN
               ↑
          Stable Release
               ↑
            DEVELOP
               ↑
        Pull Request + Review
               ↑
          FEATURE BRANCH
               ↑
       Code → Test → Commit
               ↑
            Developer
```
============================================================
## 27. TEAM BRANCH ASSIGNMENT
```text
============================================================
BRANCH                    OWNER
============================================================
feature/data              Người 1
                          Data Collection
                          Data Understanding
                          Data Cleaning
feature/visualization     Người 2
                          EDA
                          Statistical Analysis
                          Feature Relationship
                          Visualization
feature/features          Người 3
                          Feature Engineering
                          NLP
                          Skill Extraction
                          Preprocessing
feature/model             Người 4
                          Modeling
                          Loss Function
                          Bias / Variance
                          Evaluation
feature/pipeline          Người 5
                          Integration
                          Pipeline
                          Testing
                          Documentation
============================================================
```
