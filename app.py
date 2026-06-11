import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------------------------------------------------
# STEP 1: PAGE CONFIGURATION (Lệnh Streamlit đầu tiên)
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# -----------------------------------------------------------------------------
# STEP 2: CACHED DATA LOADING & PROCESS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

# Định nghĩa danh sách biến đặc trưng cố định theo Notebook
FEATURE_COLS = [f"X_{i}" for i in range(1, 15)]
TARGET_COL = "default"

# -----------------------------------------------------------------------------
# STEP 3: SIDEBAR - CONTROL PANEL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Chọn file dữ liệu chứa các biến từ X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    
    # 2. Cấu hình siêu tham số mô hình
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Random Forest Classifier")
    
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=2, 
        max_value=30, 
        value=10, 
        step=1,
        help="Độ sâu tối đa của từng cây quyết định."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        value=42, 
        step=1,
        help="Giá trị số nguyên để cố định tính ngẫu nhiên khi huấn luyện."
    )
    
    test_size = st.slider(
        "Tỷ lệ dữ liệu kiểm định (Test Size)", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.3, 
        step=0.05,
        help="Tỷ lệ phân chia tập dữ liệu để đánh giá mô hình."
    )

    st.divider()
    
    # 3. Nút kích hoạt huấn luyện mô hình
    train_button = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu quy trình trích xuất dữ liệu và huấn luyện mô hình."
    )

# -----------------------------------------------------------------------------
# STEP 4: HEADER & DATA INITIALIZATION
# -----------------------------------------------------------------------------
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng học máy hỗ trợ tự động phân tích và nhận diện rủi ro tín dụng/gian lận dựa trên các đặc trưng giao dịch tổng hợp.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên file dữ liệu mẫu (.csv hoặc .xlsx) tại thanh Sidebar bên trái để bắt đầu.")
    st.stop()

# Đọc dữ liệu từ cache khi file đã được upload
df_main = load_data(uploaded_file, uploaded_file.name)

if df_main is not None:
    st.caption(f"📁 Đang sử dụng tệp: `{uploaded_file.name}` | Tổng số dòng: {df_main.shape[0]:,} | Tổng số cột: {df_main.shape[1]}")
else:
    st.error("Không thể xử lý tệp dữ liệu được tải lên.")
    st.stop()

st.divider()

# -----------------------------------------------------------------------------
# STEP 5: MODEL TRAINING CORE ENGINE (Chạy một lần khi click nút)
# -----------------------------------------------------------------------------
if train_button:
    # Kiểm tra sự tồn tại của các cột bắt buộc
    missing_features = [col for col in FEATURE_COLS if col not in df_main.columns]
    
    if missing_features:
        st.error(f"❌ Dữ liệu thiếu các cột đặc trưng bắt buộc: {missing_features}")
    elif TARGET_COL not in df_main.columns:
        st.error(f"❌ Dữ liệu thiếu cột mục tiêu quyết định `{TARGET_COL}`.")
    else:
        with st.spinner("⏳ Đang huấn luyện mô hình Random Forest... Xin vui lòng đợi."):
            # Chuẩn bị tập dữ liệu
            X = df_main[FEATURE_COLS]
            y = df_main[TARGET_COL]
            
            # Phân tách dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Khởi tạo và khớp mô hình
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Dự đoán kiểm định
            y_pred = model.predict(X_test)
            
            # Lưu trữ trạng thái mô hình và kết quả
            st.session_state['trained'] = True
            st.session_state['model'] = model
            st.session_state['metrics'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            st.session_state['cm'] = confusion_matrix(y_test, y_pred)
            st.session_state['report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Ghi nhận thông tin độ quan trọng của biến
            st.session_state['feature_importances'] = pd.DataFrame({
                'Đặc trưng': FEATURE_COLS,
                'Mức độ quan trọng': model.feature_importances_
            }).sort_values(by='Mức độ quan trọng', ascending=False)
            
            st.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các Tab để xem chi tiết.")

# -----------------------------------------------------------------------------
# STEP 6: APP TABS SECTION
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Sử dụng mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    st.subheader("Phân tích cấu trúc dữ liệu thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng dòng (Records)", f"{df_main.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng cột (Features)", f"{df_main.shape[1]}")
    with col_m3:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.metric("Dung lượng File", f"{file_size_mb:.2f} MB")
        
    st.write("### Xem trước 5 dòng dữ liệu đầu tiên")
    st.dataframe(df_main.head(5), use_container_width=True)
    
    # Kiểm tra và chỉ hiển thị thống kê mô tả cho các biến cấu thành mô hình
    available_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df_main.columns]
    if available_cols:
        st.write("### Bảng thống kê mô tả các biến đặc trưng")
        st.dataframe(df_main[available_cols].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    st.subheader("Phân tích trực quan các biến")
    
    # Ưu tiên biến mục tiêu "default" nếu tồn tại
    if TARGET_COL in df_main.columns:
        fig_target = px.histogram(
            df_main, 
            x=TARGET_COL, 
            color=TARGET_COL,
            title="Phân phối của Biến Mục Tiêu (0: Bình thường, 1: Gian lận/Rủi ro)",
            category_orders={TARGET_COL: [0, 1]},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_target.update_layout(height=350, bargap=0.2)
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("### Trực quan hóa các biến đầu vào dữ liệu liên tục")
    
    # Cho phép người dùng tùy chọn 4 biến để hiển thị dạng lưới 2x2 linh hoạt
    selected_features = st.multiselect(
        "Chọn tối đa các đặc trưng để xem phân phối dữ liệu:",
        options=FEATURE_COLS,
        default=FEATURE_COLS[:4]
    )
    
    if selected_features:
        cols_grid = st.columns(2)
        for idx, feat in enumerate(selected_features):
            if feat in df_main.columns:
                target_exist = TARGET_COL in df_main.columns
                fig_feat = px.histogram(
                    df_main,
                    x=feat,
                    color=TARGET_COL if target_exist else None,
                    marginal="box",
                    title=f"Phân phối tần suất đặc trưng {feat}",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    barmode="overlay"
                )
                fig_feat.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                cols_grid[idx % 2].plotly_chart(fig_feat, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến để hiển thị đồ thị.")

# --- TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH ---
with tab3:
    st.subheader("Chỉ số đánh giá năng lực mô hình AI")
    
    if 'trained' not in st.session_state:
        st.info("ℹ️ Vui lòng bấm nút **'Huấn luyện mô hình'** tại thanh điều hướng bên trái để kích hoạt tính toán số liệu.")
    else:
        metrics = st.session_state['metrics']
        
        # Hiển thị Metric Thống kê tổng hợp
        c_a, c_p, c_r, c_f = st.columns(4)
        c_a.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy']:.4f}")
        c_p.metric("Độ chính xác mô hình (Precision)", f"{metrics['precision']:.4f}")
        c_r.metric("Tỷ lệ bắt sót mục tiêu (Recall)", f"{metrics['recall']:.4f}")
        c_f.metric("Điểm F1-Score", f"{metrics['f1']:.4f}")
        
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write("#### Ma trận nhầm lẫn (Confusion Matrix)")
            cm = st.session_state['cm']
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                color_continuous_scale="Blues"
            )
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_res2:
            st.write("#### Đồ thị mức độ quan trọng các biến (Feature Importance)")
            fi_df = st.session_state['feature_importances']
            fig_fi = px.bar(
                fi_df.head(10),
                x='Mức độ quan trọng',
                y='Đặc trưng',
                orientation='h',
                color='Mức độ quan trọng',
                color_continuous_scale="Viridis",
                title="Top 10 đặc trưng ảnh hưởng lớn nhất"
            )
            fig_fi.update_layout(height=350, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_fi, use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO ---
with tab4:
    st.subheader("Vận hành mô hình dự đoán rủi ro thực tế")
    
    if 'trained' not in st.session_state:
        st.info("ℹ️ Vui lòng huấn luyện mô hình thành công trước khi thực hiện chức năng dự báo.")
    else:
        model = st.session_state['model']
        
        mode = st.radio(
            "Phương thức kiểm tra thông tin:",
            options=["Nhập chỉ số trực tiếp từ form", "Tải tệp danh sách cần chấm điểm hàng loạt (X_new)"],
            horizontal=True
        )
        
        if mode == "Nhập chỉ số trực tiếp từ form":
            st.write("### Nhập tham số cấu thành giao dịch")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(4)
                input_data = {}
                
                # Tạo nhanh các ô nhập liệu số tự động căn cứ trên khoảng dữ liệu gốc
                for idx, feat in enumerate(FEATURE_COLS):
                    col_target = form_cols[idx % 4]
                    default_val = float(df_main[feat].median()) if feat in df_main.columns else 0.0
                    min_val = float(df_main[feat].min()) if feat in df_main.columns else -10.0
                    max_val = float(df_main[feat].max()) if feat in df_main.columns else 10.0
                    
                    input_data[feat] = col_target.number_input(
                        f"Giá trị {feat}",
                        value=default_val,
                        min_value=min_val,
                        max_value=max_val,
                        format="%.5f"
                    )
                
                submit_pred = st.form_submit_button("Tiến hành phân tích rủi ro", type="primary")
                
                if submit_pred:
                    # Chuyển đổi định dạng phù hợp mô hình đầu vào
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
                    st.divider()
                    if prediction == 1:
                        st.error(f"🚨 **Cảnh báo: Giao dịch có nguy cơ GIAN LẬN / RỦI RO CAO!**")
                        st.metric("Xác suất rủi ro", f"{probabilities[1]*100:.2f}%")
                    else:
                        st.success(f"✅ **An toàn: Giao dịch được phân loại BÌNH THƯỜNG.**")
                        st.metric("Xác suất an toàn", f"{probabilities[0]*100:.2f}%")
                        
        elif mode == "Tải tệp danh sách cần chấm điểm hàng loạt (X_new)":
            st.write("### Tải lên tệp danh sách mới")
            batch_file = st.file_uploader(
                "Chọn file dữ liệu mới cần chấm điểm rủi ro (CSV/XLSX)", 
                type=["csv", "xlsx"],
                key="batch_file_uploader"
            )
            
            if batch_file is not None:
                df_batch = load_data(batch_file, batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra cấu trúc cột
                    missing_batch_cols = [c for c in FEATURE_COLS if c not in df_batch.columns]
                    if missing_batch_cols:
                        st.error(f"❌ File tải lên sai định dạng cấu trúc mẫu. Thiếu các cột đặc trưng sau: {missing_batch_cols}")
                    else:
                        # Thực hiện dự báo hàng loạt
                        X_batch = df_batch[FEATURE_COLS]
                        batch_preds = model.predict(X_batch)
                        batch_probs = model.predict_proba(X_batch)[:, 1]
                        
                        # Gán kết quả đầu ra trực tiếp vào DataFrame mới
                        df_res = df_batch.copy()
                        df_res['Dự_báo_default'] = batch_preds
                        df_res['Xác_suất_rủi_ro'] = batch_probs
                        
                        st.write("### Kết quả phân tích phân lớp danh sách mới")
                        
                        # Thống kê nhanh số ca phát hiện
                        total_fraud = int(np.sum(batch_preds))
                        st.warning(f"🔎 Phát hiện hệ thống: **{total_fraud}** trường hợp giao dịch có dấu hiệu rủi ro trên tổng số **{len(df_res)}** dòng.")
                        
                        st.dataframe(df_res, use_container_width=True)
                        
                        # Tạo nút download xuất dữ liệu trả về cho vận hành
                        csv_data = df_res.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả phân tích (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_phat_hien_gian_lan.csv",
                            mime="text/csv"
                        )
