# Kaggle 50 Startups V1/V2 專家討論與最終建議

## 1. 專案背景

本專案使用 Kaggle 50 Startups 資料集，目標是依據 `R&D Spend`、`Administration`、`Marketing Spend` 與 `State` 預測 `Profit`。問題類型是監督式學習中的迴歸問題。

V1 版本重點在於完整呈現 CRISP-DM 六步驟與四組線性迴歸模型比較。V2 版本進一步加入多專家觀點、模型穩定性、資料限制、商業解讀與部署提醒，較適合用於正式報告。

重要提醒：資料只有 50 筆，所有結論應解讀為「預測關聯」，不能宣稱因果關係。

## 2. V1 與 V2 差異

| 項目 | V1 | V2 |
|---|---|---|
| 核心目的 | 建立 CRISP-DM sklearn 基礎流程 | 加入跨領域商業分析與更完整解讀 |
| 資料理解 | 有基本 EDA 與相關係數 | 有 EDA、特徵意義、專家解讀 |
| 建模 | 四組 Linear Regression | 四組 Linear Regression，附專家問題 |
| 評估 | Train-test + 5-fold CV | Train-test + 5-fold CV + 穩定性與複雜度討論 |
| 商業說明 | 基礎商業理解 | 行銷、CEO、州長、R&D、ML 觀點 |
| 部署 | 儲存 v1 模型 | 儲存 v2 模型並說明這是學習型部署 |
| 適合用途 | 教學入門與程式展示 | 作業報告、簡報與商業決策說明 |

## 3. 先前技術結果更新摘要

### 3.1 特徵選擇技術結果

| 方法 | 主要發現 | 解讀 |
|---|---|---|
| Pearson Correlation | R&D Spend = 0.9729，Marketing Spend = 0.7478，Administration = 0.2007 | R&D 與 Profit 的線性關聯最強，Marketing 次之，Administration 較弱 |
| SelectKBest F-Regression | R&D F-score = 849.789，Marketing = 60.882，Administration = 2.015 | 統計檢定支持 R&D 為最強預測特徵 |
| RFE | 選出 R&D Spend 與 Marketing Spend | 模型式特徵選擇支持雙特徵精簡模型 |
| LASSO | R&D 係數最大，Marketing 次之，State 被壓到 0 | 正則化後 State 幾乎沒有額外貢獻 |
| Random Forest Importance | R&D = 0.9228，Marketing = 0.0670，Administration = 0.0072，State 約 0.0030 | 非線性模型也支持 R&D 主導預測 |

### 3.2 模型子集比較

| 模型 | 特徵 | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| R&D only | R&D Spend | 6077.36 | 7714.33 | 0.9265 |
| R&D + Marketing | R&D Spend, Marketing Spend | 6469.18 | 8206.33 | 0.9168 |
| Numeric all | R&D Spend, Marketing Spend, Administration | 6979.15 | 8995.91 | 0.9001 |
| R&D + Marketing + State | R&D Spend, Marketing Spend, State | 6454.51 | 8254.69 | 0.9159 |
| All predictors | R&D Spend, Marketing Spend, Administration, State | 6961.48 | 9055.96 | 0.8987 |

### 3.3 目前技術判斷

單一 train-test split 中，`R&D only` 的 R2 與 RMSE 最佳；但先前完整技術報告推薦 `R&D Spend + Marketing Spend` 作為優化模型，原因是它保留了市場擴張資訊，且 5-fold CV R2 mean 約為 0.9389，具備不錯的泛化表現。

因此，本專案不應簡單說「只保留 R&D」。比較穩健的建議是：以 `R&D Spend + Marketing Spend` 作為商業可解釋的主模型，同時保留 `R&D only` 作為最簡潔基準模型。

## 4. 四位專家 5 回合討論

### 第 1 回合：如何看待 R&D Spend？

市場行銷專家：R&D 是產品價值的起點。如果產品本身沒有競爭力，行銷預算可能只是放大缺點。

頂級 CEO：從資源配置角度看，R&D 是本資料集中最值得優先保留的變數。它與 Profit 的關聯最強，也最容易向董事會解釋。

專業州長：R&D 高的公司可能也集中在人才、資金與供應鏈較成熟的地區，但本資料的州別樣本太少，不能把地區因素講得太重。

R&D 專家：R&D Spend 代表產品開發、技術能力與長期競爭力。它應該是模型核心，不能移除。

本回合結論：R&D Spend 是最核心預測特徵，應固定保留。

### 第 2 回合：Marketing Spend 是否必要？

市場行銷專家：Marketing Spend 反映品牌曝光、客戶取得與市場擴張。即使 R&D 很強，沒有市場觸達也不一定能轉成營收。

頂級 CEO：Marketing 的商業價值取決於是否能帶來有效轉換。模型若加入 Marketing，雖然單次測試 R2 略低，但商業語意更完整。

專業州長：Marketing 成效也可能受到各州市場規模、消費者密度與競爭環境影響；但目前 State 貢獻弱，所以不能過度解讀區域差異。

R&D 專家：Marketing 可能與 R&D 同時反映公司規模，因此應檢查共線性，不要把 Marketing 解釋為獨立因果因素。

本回合結論：Marketing Spend 建議保留在主模型中，但解讀時應避免因果化。

### 第 3 回合：Administration 是否該移除？

市場行銷專家：Administration 對市場成長沒有直接訊號，通常不如 R&D 與 Marketing 有解釋力。

頂級 CEO：Administration 可能代表管理規模與營運成熟度，但從結果看，它加入後模型表現下降，不適合作為主模型必要變數。

專業州長：Administration 成本也可能受當地薪資、租金與法規成本影響，但資料只含三個州，樣本不足以支持深入政策解讀。

R&D 專家：Administration 若沒有連到產品研發效率，對 Profit 預測幫助有限。

本回合結論：Administration 不應一開始刪除，但經模型比較後，可不放入最終主模型。

### 第 4 回合：State 是否有商業價值？

市場行銷專家：State 可能反映不同地區的市場差異，但本資料中 State 的特徵重要性很低。

頂級 CEO：如果加入 State 沒有明顯提升 CV 表現，就不應為了完整性犧牲模型簡潔性。

專業州長：州別可能包含稅務、人才密度、投資環境與勞動成本，但每州樣本數太少。政策分析需要更多地區資料，不能只靠這 50 筆。

R&D 專家：R&D 團隊品質比州別標籤更接近技術競爭力本身。

本回合結論：State 可作為輔助特徵測試，但不建議放入最終主模型。

### 第 5 回合：V1/V2 最終模型與報告策略

市場行銷專家：報告中應強調「產品創新 + 市場擴張」的組合，而不是只講研發預算。

頂級 CEO：最終建議要能落地：主模型用 `R&D Spend + Marketing Spend`，基準模型保留 `R&D only`，並用 CV 說明穩定性。

專業州長：地區政策不能被忽略，但這份資料不足以支持州別策略。建議未來補充稅率、人才密度、薪資、產業聚落等外部資料。

R&D 專家：模型決策應以 R&D 為主軸，Marketing 為輔助，Administration 與 State 為候選但非核心。

本回合結論：V2 是較適合提交的版本；主模型建議採用 `R&D Spend + Marketing Spend`，並以 `R&D only` 作為最簡潔比較基準。

## 5. 最終建議

### 5.1 模型建議

最終主模型建議：

```text
Profit = f(R&D Spend, Marketing Spend)
```

理由：

1. R&D Spend 在所有技術方法中都是最強特徵。
2. Marketing Spend 提供市場擴張與客戶取得的商業語意。
3. Administration 與 State 經測試後增益有限，不適合放入主模型。
4. 模型比全特徵模型更簡潔、可解釋、容易部署。
5. 50 筆資料太少，因此應用 5-fold CV 補強模型穩定性評估。

### 5.2 商業建議

對創業者：優先確保 R&D 投入能形成可銷售的產品能力，再配置 Marketing 擴大市場觸達。

對 CEO/管理者：不要把更多特徵視為更好決策。應採用簡潔、穩定、可解釋的模型作為預算討論工具。

對投資人：R&D Spend 是最值得優先觀察的量化訊號，但仍需搭配產品、市場、團隊與財務基本面。

對州政府/政策單位：State 在此資料集中貢獻有限，不代表地區政策不重要，而是資料粒度不足。若要分析政策效果，需要更多外部變數與更大樣本。

對 R&D 團隊：R&D 是核心競爭力訊號，但預算投入不等於結果保證。應搭配產品里程碑、研發效率、專利/技術壁壘與上市速度一起評估。

### 5.3 報告用一句話結論

本資料顯示，新創 Profit 的預測主要由 `R&D Spend` 主導，`Marketing Spend` 提供次要但有商業意義的補充；`Administration` 與 `State` 在目前 50 筆資料中增益有限，因此最終建議使用 `R&D Spend + Marketing Spend` 作為簡潔且可解釋的主模型，並將結果視為預測關聯而非因果結論。

## 6. 技術圖表與資料更新索引

既有圖表：

| 圖表 | 檔案 |
|---|---|
| CRISP-DM 一頁式資訊圖 | `one_page_crispdm_feature_selection_infographic.png` |
| 技術報告 PDF | `feature_selection_technical_report.pdf` |
| Pearson Correlation | `outputs/algorithm_charts/01_pearson_correlation.png` |
| SelectKBest F Scores | `outputs/algorithm_charts/02_selectkbest_f_scores.png` |
| RFE Selection Strength | `outputs/algorithm_charts/03_rfe_selection_strength.png` |
| LASSO Coefficient Path | `outputs/algorithm_charts/04_lasso_coefficient_path.png` |
| Random Forest Importance | `outputs/algorithm_charts/05_random_forest_importance.png` |
| Model Subset Curves | `outputs/algorithm_charts/06_model_subset_curves.png` |
| Actual vs Predicted | `outputs/algorithm_charts/07_actual_vs_predicted.png` |
| Residual Analysis | `outputs/algorithm_charts/08_residual_analysis.png` |

本次新增/更新資料：

| 資料 | 檔案 |
|---|---|
| V1/V2 專家討論與最終建議 | `outputs/v1_v2_expert_discussion_final_recommendation.md` |
| 最終建議精簡版 | `outputs/final_recommendation_updated.md` |
| 技術結果摘要資料表 | `outputs/updated_technical_results_summary.csv` |
| 圖表索引資料表 | `outputs/updated_chart_index.csv` |
