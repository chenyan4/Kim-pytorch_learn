### pandas 常用操作速查表（含例子）

下面默认环境：

```python
import pandas as pd
import numpy as np
```

示例数据：

```python
data = {
    "State": ["CA", "CA", "NY", "TX"],
    "City": ["LA", "SF", "NYC", "Dallas"],
    "Price": [100, 200, 300, 400],
    "Rooms": [3, 2, 4, 5],
}
df = pd.DataFrame(data)
```

---

## 一、基础操作

- **读取 / 保存 CSV**

```python
df = pd.read_csv("train.csv")                # 读
df.to_csv("out.csv", index=False)            # 写
```

- **查看基本信息**

```python
df.head(5)           # 前5行
df.tail(3)           # 后3行
df.shape             # (行数, 列数)
df.columns           # 列名
df.dtypes            # 每列类型
df.info()            # 概况（包含缺失、内存等）
df.describe()        # 数值列统计信息
df.describe(include="all")  # 包含类别列
```

---

## 二、选行选列

- **按列名选择**

```python
df["Price"]                  # Series
df[["State", "Price"]]       # DataFrame
```

- **按位置选择（iloc）**

```python
df.iloc[0]           # 第1行
df.iloc[0:2]         # 第1~2行
df.iloc[:, 1:3]      # 所有行，第2~3列
```

- **按标签选择（loc）**

```python
df.loc[0, "Price"]               # 行索引=0，列名=Price
df.loc[0:2, ["State", "Price"]]  # 0~2行，State 和 Price 列
```

---

## 三、条件过滤

- **按条件筛选行**

```python
df[df["Price"] > 200]
df[(df["Price"] > 100) & (df["State"] == "CA")]
df[df["State"].isin(["CA", "TX"])]
```

- **缺失值过滤**

```python
df[df["Price"].isna()]      # Price 为空
df[df["Price"].notna()]     # Price 非空
```

---

## 四、新增 / 修改列

- **简单计算列**

```python
df["Price_per_room"] = df["Price"] / df["Rooms"]
```

- **使用 apply**

```python
def price_level(x):
    return "high" if x > 200 else "low"

df["Level"] = df["Price"].apply(price_level)
```

- **批量标准化数值列（类似 `california_house_predict.py` 中的做法）**

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(
    lambda x: (x - x.mean()) / x.std()
)
```

---

## 五、缺失值处理

示例：

```python
df2 = pd.DataFrame({
    "A": [1, 2, np.nan, 4],
    "B": ["x", None, "y", "z"],
})
```

- **检查缺失**

```python
df2.isna().sum()          # 每列缺失个数
```

- **填充缺失**

```python
df2["A"].fillna(df2["A"].mean(), inplace=True)   # 用平均值填
df2["B"].fillna("unknown", inplace=True)         # 用固定值填
```

- **丢弃含缺失的行**

```python
df2.dropna()                                  # 任意列缺失就丢
df2.dropna(subset=["A", "B"])                 # 只关心 A、B 两列
```

---

## 六、字符串列处理

```python
df_str = pd.DataFrame({
    "address": ["LA, CA", "SF, CA", "NYC, NY"]
})
```

- **大小写、包含**

```python
df_str["address"].str.lower()
df_str["has_CA"] = df_str["address"].str.contains("CA")
```

- **拆分字符串**

```python
# 拆成两列
df_str[["city", "state"]] = df_str["address"].str.split(", ", expand=True)
```

---

## 七、排序与重命名

- **排序**

```python
df.sort_values("Price", ascending=False)          # 按价格降序
df.sort_values(["State", "Price"])               # 先 State 再 Price
```

- **重命名列**

```python
df_renamed = df.rename(columns={"Price": "SoldPrice"})
```

---

## 八、分组统计（groupby）

```python
# 每个州的平均价格
df.groupby("State")["Price"].mean()

# 多个聚合
df.groupby("State").agg(
    mean_price=("Price", "mean"),
    max_price=("Price", "max"),
    count=("Price", "count"),
)
```

---

## 九、拼接与合并

- **按行拼接（堆数据）**

```python
df1 = df.iloc[:2]
df2 = df.iloc[2:]
df_all = pd.concat([df1, df2], axis=0, ignore_index=True)
```

- **按列拼接**

```python
extra = pd.DataFrame({"Extra": [10, 20, 30, 40]})
df_concat = pd.concat([df, extra], axis=1)
```

- **类似 SQL 的 join（merge）**

```python
left = pd.DataFrame({"Id": [1, 2], "Price": [100, 200]})
right = pd.DataFrame({"Id": [1, 2], "State": ["CA", "NY"]})

pd.merge(left, right, on="Id", how="inner")   # inner / left / right / outer
```

---

## 十、类别变量与 one-hot 编码

```python
df_cat = pd.DataFrame({
    "State": ["CA", "NY", "TX", np.nan]
})
pd.get_dummies(df_cat, dummy_na=True)
# 会生成 State_CA, State_NY, State_TX, State_nan 四列
```

常配合数值列一起用：

```python
numeric = df.select_dtypes(include=[np.number])
cat = df[["State"]]
df_all = pd.concat([numeric, cat], axis=1)
df_all = pd.get_dummies(df_all, dummy_na=True)
```

---

## 十一、索引与重置

```python
df_indexed = df.set_index("City")        # 以 City 为行索引
df_reset = df_indexed.reset_index()      # 恢复普通列
df_reset2 = df_indexed.reset_index(drop=True)  # 丢弃索引，不保留列
```

---

## 十二、与 NumPy / PyTorch 联动

- **DataFrame → NumPy**

```python
arr = df.values          # 或 df.to_numpy()
```

- **DataFrame → torch.tensor（与 `california_house_predict.py` 中一致）**

```python
import torch

features = torch.tensor(df.values.astype(float), dtype=torch.float32)
```

