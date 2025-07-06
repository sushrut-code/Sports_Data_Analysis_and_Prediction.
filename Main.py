#Imports
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro,f_oneway,pearsonr
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

#Data Loading
df=pd.read_excel("C:\\Users\\LENOVO\\Downloads\\Strikers_performance.xlsx")
#500 rows,19 columns
# Index(['Striker_ID', 'Nationality', 'Footedness', 'Marital Status',
#        'Goals Scored', 'Assists', 'Shots on Target', 'Shot Accuracy',
#        'Conversion Rate', 'Dribbling Success', 'Movement off the Ball',
#        'Hold-up Play', 'Aerial Duels Won', 'Defensive Contribution',
#        'Big Game Performance', 'Consistency', 'Penalty Success Rate',
#        'Impact on Team Performance', 'Off-field Conduct'],
#       dtype='object')

#
#Movement off the Ball-6,Big Game Performance-2,Penalty Success Rate -5
#we will impute the values using the Median Values.

#Imputation
imputer=SimpleImputer(strategy="median")
imputer.fit(df[['Movement off the Ball']])
df[['Movement off the Ball']]=imputer.transform(df[['Movement off the Ball']])
imputer.fit(df[['Big Game Performance']])
df[['Big Game Performance']]=imputer.transform(df[['Big Game Performance']])
imputer.fit(df[['Penalty Success Rate']])
df[['Penalty Success Rate']]=imputer.transform(df[['Penalty Success Rate']])

#checking null values!!
null_values=df.isnull().sum()

#checking duplicates
duplicate_values=df.duplicated()
#no duplicates values in thee dataframe

#Footedness distribution pie chart
Footedness=df['Footedness'].value_counts()
Footedness=Footedness/len(df['Footedness'])*100
Footedness.plot(kind="pie",autopct="%1.2f%%")
plt.title("Footedness Distribution")
plt.ylabel("")

#Clustered bar chart showing nationality vs footedness
sns.countplot(x='Nationality',hue='Footedness',data=df)
plt.title("Distribution of Footedness across various Nationality")
plt.legend()

#Group-wise analysis
GoalScorer_based_on_nationality=df.groupby("Nationality")['Goals Scored'].mean()

#Highest avg Goal Scored based on Nationality:Brazil     15.804927

ConversionRate_based_on_footedness=df.groupby("Footedness")['Conversion Rate'].mean()

#the average conversion rate for players based on their footedness: Left-footed     19.8, Right-footed    20.0.

#Find whether there is any significant difference in consistency rates among strikers from various nationalities.
#we need to use one-way annova

#First test normality of "Conversion".We will plot kde
# sns.kdeplot(df['Consistency'])
# plt.title("Distribution of Consistency")
# plt.show()
#Looks like normal distribution...
#Let's do Shapiro Wilk Test

numeric_column=['Consistency']
shapiro_result={}
for column in numeric_column:
    stat,p_value=shapiro(df[column])
    shapiro_result[column]=round(p_value,3)

#our significance level was 0.05 but the p_value is 0.451.Since the p_value is greater than alpha,
#it follows normal distribution...

#Separate consistency values for each nationality
#['Spain' 'France' 'Germany' 'Brazil' 'England']
Spain_consitency=df.query('Nationality=="Spain"')['Consistency']
France_consitency=df.query('Nationality=="France"')['Consistency']
Germany_consitency=df.query('Nationality=="Germany"')['Consistency']
Brazil_consitency=df.query('Nationality=="Brazil"')['Consistency']
England_consitency=df.query('Nationality=="England"')['Consistency']

#One-way ANOVA test
tstats,p_value=f_oneway(Spain_consitency,France_consitency,Germany_consitency,Brazil_consitency,England_consitency)

#p_value=0.19278675901599154 which is greater than significance level.Hence null hypothesis accepted.
#there is no significant difference in consistency rates among strikers from various nationalities.

# if there is any significant correlation between strikers' Hold-up play and consistency rate?
#linearity,normality mandatory

#Distribution plot
sns.kdeplot(df['Hold-up Play'])
plt.title("Distribution of Consistency")

#Shapiro test for hold-up play
numeric_column=['Hold-up Play']
shapiro_result={}
for column in numeric_column:
    stat,p_value=shapiro(df[column])
    shapiro_result[column]=round(p_value,3)

#hold-up play is also normally distributed.

#Pearson correlation test
consistency=df['Consistency']
hold_up_play=df['Hold-up Play']
corr,p_value=pearsonr(consistency,hold_up_play)

#weakly correlated p_value=0.0011443972418055694, corr_coef=0.14504436542869956

#if strikers' hold-up play significantly influences their consistency rate.

x=df['Hold-up Play']
y=df['Consistency']

#Linear regression model
x_and_const=sm.add_constant(x)
model=sm.OLS(y,x_and_const).fit()

#R-squared: 0.021, p_value:  0.001
#✔️ Yes, Hold-up Play significantly influences Consistency — the p-value confirms that.
# ❗️But the influence is weak — it only explains a small portion (2.1%) of consistency variation.

# List of columns to sum
contribution_columns = [
    'Goals Scored',
    'Assists',
    'Shots on Target',
    'Dribbling Success',
    'Aerial Duels Won',
    'Defensive Contribution',
    'Big Game Performance',
    'Consistency'
]

# Create the new feature: total contribution score
df['Total Contribution Score'] = round(df[contribution_columns].sum(axis=1),2)

#Encoding categorical columns
encoder=LabelEncoder()
df['Footedness']=encoder.fit_transform(df['Footedness'])
df['Marital Status']=encoder.fit_transform(df['Marital Status'])

#One-hot encoding Nationality
dummies=pd.get_dummies(df[['Nationality']])
df=pd.concat([df,dummies],axis=1)

#Clustering - drop irrelevant features
x_clust=df.drop(['Striker_ID','Nationality'],axis=1)
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(x_clust)
    wcss_values=kmeans.inertia_
    wcss.append(wcss_values)

#Elbow curve
plt.plot(range(1,11),wcss,marker="o")
plt.title("Elbow Method")
plt.xlabel("No of Clusters")
plt.ylabel("wcss")

#Fitting KMeans with 3 clusters
kmeans=KMeans(n_clusters=3)
kmeans.fit(x_clust)
labels=kmeans.labels_
df['Clusters']=labels

#0    130.998571
# 1    101.649497
# 2    117.856471

#Mapping clusters to striker types
mapping={0:'Regular Strikers',1:"Good Striker",2:"Best Striker"}
df['Striker Type']=df['Clusters'].map(mapping)

# print(df.columns)
'''['Striker_ID', 'Nationality', 'Footedness', 'Marital Status',
       'Goals Scored', 'Assists', 'Shots on Target', 'Shot Accuracy',
       'Conversion Rate', 'Dribbling Success', 'Movement off the Ball',
       'Hold-up Play', 'Aerial Duels Won', 'Defensive Contribution',
       'Big Game Performance', 'Consistency', 'Penalty Success Rate',
       'Impact on Team Performance', 'Off-field Conduct',
       'Total Contribution Score', 'Nationality_Brazil', 'Nationality_England',
       'Nationality_France', 'Nationality_Germany', 'Nationality_Spain',
       'Clusters', 'Striker Type']'''

#Preparing data for classification
x_class=df.drop(['Striker_ID','Nationality','Striker Type','Clusters'],axis=1)
y_class=df['Striker Type']

#Scaling features
scaler=StandardScaler()
scaled_class=scaler.fit_transform(x_class)

#Train-test split
x_train,x_test,y_train,y_test=train_test_split(scaled_class,y_class,test_size=0.2,random_state=42)

#Logistics regression machine Learning Model
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

#Classification report
print(classification_report(y_test, y_pred))

#Confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
print(conf_matrix)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

#Prediction for a new player
new_player = {
    'Footedness':1,
    'Marital Status':1,
    'Goals Scored': 20,
    'Assists': 10,
    'Shots on Target': 35,
    'Shot Accuracy': 0.6,
    'Conversion Rate': 0.2,
    'Dribbling Success': 0.8,
    'Movement off the Ball': 60,
    'Hold-up Play': 70,
    'Aerial Duels Won': 15,
    'Defensive Contribution': 25,
    'Big Game Performance': 7,
    'Consistency': 0.85,
    'Penalty Success Rate': 0.9,
    'Impact on Team Performance': 9,
    'Off-field Conduct': 8,
    'Total Contribution Score': 150,  # Must be consistent with training
    'Nationality_Brazil': 0,
    'Nationality_England': 0,
    'Nationality_France': 1,
    'Nationality_Germany': 0,
    'Nationality_Spain': 0
}

#Convert to DataFrame and scale
new_df = pd.DataFrame([new_player])
new_scaled = scaler.transform(new_df)

#Predict
predicted_type = model.predict(new_scaled)
print("Predicted Striker Type:", predicted_type[0])
