import pandas as pd
df =pd.read_excel(r"D:/datasets/Telco_customer_churn.xlsx")
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["Churn Value"].value_counts())
print(pd.crosstab(df["Contract"],df["Churn Value"],normalize="index")*100)
df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins= [0,12,24,48,72],
    labels= ["0-12","13-24","25-48","49-72"])
print(pd.crosstab(df["Tenure Group"],df["Churn Value"],normalize="index")*100)
df["Monthly Group"] = pd.qcut(df["Monthly Charges"],
                              q=4,
                              labels=["Low","Medium","High","Very High"])
print(pd.crosstab(df["Monthly Group"],df["Churn Value"],normalize="index")*100)
categorical_col = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method"
]

for col in categorical_col:
    print("\n",col)
    print(pd.crosstab(df[col],df["Churn Value"],normalize="index")*100)
    
numeric_col = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV"
]    
for col in numeric_col:
    df[col] = pd.to_numeric(df[col],errors="coerce")
    print("\n",col)
    print(df.groupby("Churn Value")[col].agg(["mean","median","std"]))
    
import matplotlib.pyplot as plt

pd.crosstab(
    df["Contract"],
    df["Churn Value"],
    normalize="index"
).plot(kind="bar")

plt.title("Churn Rate by Contract")
plt.xlabel("Contract")
plt.ylabel("Churn Rate")
plt.legend(title="Churn")
plt.show()
df.boxplot(
    column="Tenure Months",
    by="Churn Value"
)

plt.title("Tenure vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure Months")
plt.show()
df.boxplot(
    column="Monthly Charges",
    by="Churn Value"
)

plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()
pd.crosstab(
    df["Payment Method"],
    df["Churn Value"],
    normalize="index"
).plot(kind="bar")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate")
plt.show()

df["Avg Monthly Spend"] = df["Total Charges"] / df["Tenure Months"].replace(0,1)
df["Is_New_Customer"] = (df["Tenure Months"] <=12).astype(int)
services = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support"
]
df["Num_Support_Services"] = (df[services] =="Yes").sum(axis=1)

x = df.drop(columns=["Churn Value","CustomerID","City","Zip Code","Lat Long","Churn Label","Churn Reason","Churn Score"])
y = df["Churn Value"]
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from sklearn.metrics import classification_report,roc_auc_score
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
x_train = pd.get_dummies(x_train,drop_first=True)
x_test = pd.get_dummies(x_test,drop_first=True)
x_test = x_test.reindex(columns=x_train.columns,fill_value=0)
print(x_train.shape)
print(x_test.shape)
print(x_train.columns.tolist())
x_train = x_train.astype(float)
x_test = x_test.astype(float)
scale_weight = (len(y_train)-sum(y_train))/sum(y_train)
model = xgb.XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.05,
    max_depth = 5 ,
    scale_pos_weight = scale_weight,
    random_state =42,
    eval_metric = "logloss"
)
model.fit(x_train,y_train)
y_proba = model.predict_proba(x_test)[:,1]
customer_threshold = 0.45
y_predict = (y_proba>=customer_threshold).astype(int)
print(classification_report(y_test,y_predict))
print(f"ROC-AUC Score: {roc_auc_score(y_test,y_proba):.4f}")
explainer = shap.TreeExplainer(model,feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test)