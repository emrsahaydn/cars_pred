import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import streamlit as st
#pipeline verideki boyut farkını hafizasında tutard
df=pd.read_excel('cars.xls')
x=df.drop('Price',axis=1)
y=df['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

preprocessror=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),['Mileage','Cylinder','Liter','Doors']),  #Rakamlar normalize olacak 
        ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])    #getdummies görevi görür(Yazılar sayılara dönüştürülecek)
    ]
)
model=LinearRegression()

pipeline=Pipeline(steps=[('preprocesssor',preprocessror),('regressor',model)])
pipeline.fit(x_train,y_train)
pred=pipeline.predict(x_test)

rmse=mean_squared_error(pred,y_test)*0.5
r2=r2_score(pred,y_test)

def price_pred(make,model,trim,mileage,type,cylinder,liter,doors,cruise,sound,leather):  #sound ve leather veride 1 veya 0 olduğu için onlarla ilgili yukarda bir işlem yapmadık.
    input_data=pd.DataFrame({
        'Make':[make],
        'Model':[model],
        'Trim':[trim],
        'Mileage':[mileage],
        'Type':[type],
        'Cylinder':[cylinder],
        'Liter':[liter],
        'Doors':[doors],
        'Cruise':[cruise],
        'Sound':[sound],
        'Leather':[leather]
    })

    prediction=pipeline.predict(input_data)[0]
    return prediction

def main():
    st.title('MLOps Cars Price Prediction:red_car:')
    st.write('Enter Cars Details to Predict the Price')
    make=st.selectbox('Make',df['Make'].unique())   #unique markaları getirir.
    model=st.selectbox('Model',df[(df['Make']==make)]['Model'].unique())

    trim=st.selectbox('Trim',df[(df['Make']==make)&(df['Model']==model)]['Trim'].unique())

    mileage=st.number_input('Mileage',200,60000)
    car_type=st.selectbox('Type',df['Type'].unique())
    cylinder=st.selectbox('Cylinder',df['Cylinder'].unique())
    liter=st.number_input('Mileage',1,6)
    doors=st.selectbox('Doors',df['Doors'].unique())
    cruise=st.radio('Cruise',[0,1])
    sound=st.radio('Sound',[0,1])
    leather=st.radio('Leather',[0,1])
    if st.button('Predict'):
        price=price_pred(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)
        st.write(f'The Predicted Price is: {price:.1f}$')

if __name__=='__main__':
        main()

