# 機器學習特徵選擇與模型優化：5大重點流程總結

這份專案的核心是一套基於 **CRISP-DM 方法論** 所打造的「機器學習特徵選擇與模型優化」標準流程。

## 1. 資料前處理與擴增 (Data Engineering)
- 將原始 50 筆 Startup 資料擴增至 1000 筆，增加模型穩定度。
- 透過 **PCA 降維** 與 **K-Means 分群**，將所有新創公司依照獲利特徵自動劃分為「高、中、低」三個利潤等級。

## 2. 多維度特徵篩選 (Multi-Scheme Feature Selection)
為了找出真正影響獲利的關鍵，我們不依賴單一演算法，而是同時採用了三大類、共 5 種篩選機制：
- **Filter (過濾法)**：使用 Pearson 相關係數、VIF 共線性檢定、SelectKBest。
- **Wrapper (包裝法)**：使用 RFE (遞迴特徵消除)。
- **Embedded (嵌入法)**：使用 LASSO 懲罰收縮、Random Forest 決策樹重要性。

## 3. 投票矩陣決策 (Voting Matrix)
- 整合上述 5 種機制的結果進行綜合投票。
- **分析結論**：`R&D Spend (研發支出)` 與 `Marketing Spend (行銷支出)` 獲得全票通過，確立為核心特徵；而 `Administration (行政費用)` 與 `State (州別)` 則被判定為雜訊特徵而剔除。

## 4. 10 大演算法基準測試 (Algorithm Benchmarking)
- 使用包含 Linear, Ridge, LASSO, Decision Tree, Random Forest, SVR, Gradient Boosting, KNN 等 10 種主流演算法進行交叉驗證 (Cross-Validation)。
- **結果證實**：使用優化後的 2 個特徵進行訓練，其 **MSE (均方誤差)** 收斂表現與泛化能力，優於使用全部 4 個特徵的基準模型。

## 5. 自動化與互動化部署 (Deployment)
- 將最終最好的 Regression 回歸模型與 Classifier 分類模型進行打包 (Joblib)。
- 部署為現代化、具視覺衝擊力的 Streamlit 互動儀表板，支援即時的 AI 獲利預測與分類，並將分析報表與資訊圖表無縫整合。

---
*專案產出與資訊圖表詳見本資料夾內之 `one_page_crispdm_feature_selection_infographic.png` 以及完整的 `feature_selection_technical_report.pdf`。*
