# ML Algorithms - Methods, Application Scope, and One-Page Technical Summary

## Project Context

- Dataset: 50 Startups
- Records: 50
- Target: Profit
- Missing values: 0
- Duplicates: 0
- Optimized model: Linear Regression with R&D Spend + Marketing Spend
- Test R2: 0.9168
- Test RMSE: $8,206.33
- 5-fold CV mean R2: 0.9389

## Algorithm Details

### 1. Pearson Correlation

**Category:** Filter / 統計關聯

**How it works:** 計算每個數值特徵與 Profit 的線性相關係數，越接近 +1 表示正向線性關係越強。

**Practical application scope:** EDA、快速特徵篩選、線性迴歸前置分析、報告中解釋變數與目標的直接關係。

**Limitations:** 只能看線性關係，無法判斷非線性、交互作用，也不等於因果關係。

**Project result:** Top feature: R&D Spend (0.973)

### 2. VIF

**Category:** Filter / 共線性診斷

**How it works:** 以每個特徵被其他特徵解釋的程度計算膨脹因子，判斷線性迴歸係數是否受共線性影響。

**Practical application scope:** 線性迴歸、統計建模、風險模型、需要解釋係數的商業報告。

**Limitations:** 主要用於數值特徵與線性模型，不直接衡量預測力。

**Project result:** Highest VIF: R&D Spend (2.47)

### 3. SelectKBest F-Regression

**Category:** Filter / 單變量檢定

**How it works:** 使用 F-test 評估每個特徵與連續目標的統計關聯強度，依分數排名。

**Practical application scope:** 快速篩選大量候選特徵、建立模型前的統計排名、資料科學報告。

**Limitations:** 逐一檢查特徵，無法直接考慮特徵之間的組合效果。

**Project result:** Top feature: R&D Spend (F=849.8)

### 4. Recursive Feature Elimination

**Category:** Wrapper / 模型迭代篩選

**How it works:** 用指定模型反覆訓練並移除較弱特徵，直到保留指定數量的特徵。

**Practical application scope:** 特徵數中等、需要模型導向選擇、希望比較不同特徵組合的情境。

**Limitations:** 計算成本較高，結果會受 estimator 與資料切分影響。

**Project result:** Selected: Marketing Spend, R&D Spend

### 5. LASSO Regression

**Category:** Embedded / 正則化選擇

**How it works:** 在迴歸損失中加入 L1 penalty，使弱特徵係數趨近 0，達成自動化特徵收縮。

**Practical application scope:** 高維表格資料、稀疏模型、需要兼顧預測與可解釋性的線性模型。

**Limitations:** 特徵高度相關時可能任意選其中一個；alpha 設定需交叉驗證。

**Project result:** CV alpha: 337.63

### 6. Random Forest Importance

**Category:** Embedded / 樹模型重要度

**How it works:** 建立多棵決策樹，根據特徵降低誤差或分裂貢獻估計重要度。

**Practical application scope:** 非線性資料、特徵互動、表格資料基準模型、特徵重要度交叉驗證。

**Limitations:** 重要度可能偏好高基數或連續特徵；需搭配 permutation 或 SHAP 更穩健。

**Project result:** Top feature: R&D Spend (92.33%)

### 7. Multiple Linear Regression

**Category:** Supervised / Regression

**How it works:** 以特徵的線性組合預測連續目標，透過最小化誤差學習係數。

**Practical application scope:** 房價、銷售額、成本、利潤、能耗等可解釋性需求高的連續數值預測。

**Limitations:** 假設關係近似線性，對離群值與遺漏變數敏感。

**Project result:** Optimized R2=0.9168, RMSE=$8,206

### 8. Random Forest Regressor

**Category:** Supervised / Ensemble Regression

**How it works:** 透過多棵決策樹平均預測，降低單棵樹不穩定性並捕捉非線性。

**Practical application scope:** 表格資料、非線性預測、特徵重要度、基準模型比較。

**Limitations:** 可解釋性低於線性迴歸，模型較大，外插能力有限。

**Project result:** Benchmark R2=0.9066, RMSE=$8,699

### 9. K-Fold Cross Validation

**Category:** Evaluation / 泛化驗證

**How it works:** 將資料切成 K 份輪流訓練與驗證，降低單次 train/test split 的偶然性。

**Practical application scope:** 小資料集、模型選擇、調參、作業報告可信度提升。

**Limitations:** 資料有時間序時不可隨機切分；計算成本高於單次切分。

**Project result:** 5-fold CV mean R2=0.9389

### 10. Residual Analysis

**Category:** Diagnostic / 模型診斷

**How it works:** 分析 Actual - Predicted 的殘差分布，檢查線性模型是否有系統性偏誤。

**Practical application scope:** 迴歸模型驗證、錯誤分析、商業風險說明、模型改善方向。

**Limitations:** 不是特徵選擇方法，但能指出模型假設或遺漏變數問題。

**Project result:** Residuals mostly centered around zero; low-profit cases need attention.

## Feature Evidence Table

| Feature         | Correlation         | VIF                |   RFE_Rank |   LASSO_Coefficient |   RF_Importance |
|:----------------|:--------------------|:-------------------|-----------:|--------------------:|----------------:|
| R&D Spend       | 0.9729004656594828  | 2.4689030699947017 |          1 |           36173.3   |      0.92334    |
| Administration  | 0.20071656826872147 | 1.1750910070550453 |          2 |            -300.143 |      0.0071917  |
| Marketing Spend | 0.7477657217414767  | 2.326773290530878  |          1 |            3290.92  |      0.0663687  |
| State_Florida   | N/A                 | N/A                |          3 |               0     |      0.00149974 |
| State_New York  | N/A                 | N/A                |          4 |              -0     |      0.0016     |

## Recommended Conclusion

The strongest ML recommendation is to use a compact, explainable Linear Regression pipeline with `R&D Spend` and `Marketing Spend`. The feature-selection evidence consistently shows that R&D is the strongest profit driver, while Marketing is the secondary useful predictor. Administration and State can remain in a baseline comparison model, but they are not essential for the optimized pipeline.
