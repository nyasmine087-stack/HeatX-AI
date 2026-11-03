import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from engine import simulate, batch

st.set_page_config(page_title='HeatX AI',page_icon='🌡️',layout='wide')
st.markdown('''<style>.block-container{padding-top:1.6rem;max-width:1250px} h1{letter-spacing:-.04em} [data-testid="stMetric"]{background:#f0f5f8;padding:14px;border-radius:12px}</style>''',unsafe_allow_html=True)
st.title('🌡️ HeatX AI')
st.caption('Intelligent Heat Exchanger Lab · Engineering simulation · Machine learning · Optimization')
with st.sidebar:
    st.header('⚙️ Operating conditions')
    th=st.slider('Hot inlet (°C)',45.,95.,80.,1.)
    tc=st.slider('Cold inlet (°C)',5.,40.,20.,1.)
    mh=st.slider('Hot flow (kg/s)',.2,2.,1.,.1)
    mc=st.slider('Cold flow (kg/s)',.2,2.,1.,.1)
    ua=st.slider('Clean conductance UA (W/K)',500,6000,2000,100)
    fouling=st.slider('Conductance loss from fouling (%)',0,70,0,5)
    st.info('Educational model: steady-state counter-current water/water exchanger, constant Cp=4180 J/(kg·K), no phase change or heat loss. Pressure drop is not modeled.')
actual_ua=ua*(1-fouling/100)
r=simulate(th,tc,mh,mc,actual_ua)
tabs=st.tabs(['📊 Dashboard','📈 Sensitivity','🧠 Machine learning','🎯 Optimization','🧪 Fouling & validation'])
with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric('Hot outlet',f"{r['hot_out']:.2f} °C")
    b.metric('Cold outlet',f"{r['cold_out']:.2f} °C")
    c.metric('Heat transfer',f"{r['heat_kw']:.2f} kW")
    d.metric('Effectiveness',f"{100*r['effectiveness']:.1f}%")
    st.subheader('Temperature comparison')
    fig=go.Figure()
    fig.add_trace(go.Bar(name='Inlet',x=['Hot fluid','Cold fluid'],y=[th,tc]))
    fig.add_trace(go.Bar(name='Outlet',x=['Hot fluid','Cold fluid'],y=[r['hot_out'],r['cold_out']]))
    fig.update_layout(barmode='group',yaxis_title='Temperature (°C)',height=340,margin=dict(t=20,b=20))
    st.plotly_chart(fig,use_container_width=True)
    st.write(f"**NTU:** {r['ntu']:.3f}  ·  **Effective UA:** {actual_ua:,.0f} W/K")
    st.download_button('⬇️ Download current result',pd.DataFrame([{**dict(hot_in=th,cold_in=tc,hot_flow=mh,cold_flow=mc,ua_effective=actual_ua),**r}]).to_csv(index=False),'heatx_result.csv','text/csv')
with tabs[1]:
    st.subheader('How do conditions affect performance?')
    var=st.selectbox('Change one parameter',['Hot inlet temperature','Cold inlet temperature','Hot mass flow','Cold mass flow','UA'])
    specs={'Hot inlet temperature':(np.linspace(45,95,45),0,'°C'),'Cold inlet temperature':(np.linspace(5,40,45),1,'°C'),'Hot mass flow':(np.linspace(.2,2,45),2,'kg/s'),'Cold mass flow':(np.linspace(.2,2,45),3,'kg/s'),'UA':(np.linspace(500,6000,45),4,'W/K')}
    values,idx,unit=specs[var]
    rows=[]
    for x in values:
        params=[th,tc,mh,mc,actual_ua]
        params[idx]=float(x)
        if params[0]<=params[1]: continue
        v=simulate(*params)
        rows.append({'Input':x,'Hot outlet (°C)':v['hot_out'],'Cold outlet (°C)':v['cold_out'],'Heat (kW)':v['heat_kw']})
    df=pd.DataFrame(rows)
    y=st.selectbox('Output',['Heat (kW)','Hot outlet (°C)','Cold outlet (°C)'])
    fig=go.Figure(go.Scatter(x=df['Input'],y=df[y],mode='lines',line=dict(width=3)))
    fig.update_layout(xaxis_title=f'{var} ({unit})',yaxis_title=y,height=350)
    st.plotly_chart(fig,use_container_width=True)
    st.download_button('⬇️ Download sensitivity data',df.to_csv(index=False),'sensitivity.csv','text/csv')
@st.cache_resource(show_spinner=False)
def train_model():
    rng=np.random.default_rng(42)
    X=np.column_stack([rng.uniform(45,95,2200),rng.uniform(5,40,2200),rng.uniform(.2,2,2200),rng.uniform(.2,2,2200),rng.uniform(500,6000,2200)])
    y=batch(X)
    xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=.2,random_state=42)
    model=RandomForestRegressor(n_estimators=90,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    return model,mean_absolute_error(ytest,pred,multioutput='raw_values'),r2_score(ytest,pred,multioutput='raw_values'),xtest,ytest,pred
with tabs[2]:
    st.subheader('Machine-learning temperature predictor')
    st.write('A random forest is trained on synthetic data from the engineering model. Its accuracy is measured on a held-out test set; it is **not validated against real plant measurements**.')
    if st.button('Train / load AI model',type='primary'):
        with st.spinner('Training and evaluating...'):
            model,mae,r2,xt,yt,yp=train_model()
        p=model.predict(np.array([[th,tc,mh,mc,actual_ua]]))[0]
        a,b=st.columns(2)
        a.metric('AI hot outlet',f'{p[0]:.2f} °C',f'{p[0]-r["hot_out"]:+.2f} °C vs physics')
        b.metric('AI cold outlet',f'{p[1]:.2f} °C',f'{p[1]-r["cold_out"]:+.2f} °C vs physics')
        st.write(f'**Held-out test MAE:** hot {mae[0]:.3f} °C, cold {mae[1]:.3f} °C.  **R²:** hot {r2[0]:.3f}, cold {r2[1]:.3f}.')
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=yt[:,0],y=yp[:,0],mode='markers',name='Hot outlet',marker=dict(opacity=.4)))
        fig.add_trace(go.Scatter(x=yt[:,1],y=yp[:,1],mode='markers',name='Cold outlet',marker=dict(opacity=.4)))
        fig.add_trace(go.Scatter(x=[0,100],y=[0,100],mode='lines',name='Perfect prediction',line=dict(dash='dash')))
        fig.update_layout(xaxis_title='Physics result (°C)',yaxis_title='AI prediction (°C)',height=380)
        st.plotly_chart(fig,use_container_width=True)
    else: st.info('Click the button to train the model and see its predictions.')
with tabs[3]:
    st.subheader('Find an operating point for a desired cold outlet')
    target=st.slider('Target cold outlet temperature (°C)',float(tc+1),float(th-1),float(min(th-1,max(tc+1,co if (co:=r['cold_out']) else tc+1))),1.)
    st.caption('Searches cold-water mass flow from 0.2 to 2.0 kg/s, holding other parameters fixed. This minimizes target-temperature error, not energy consumption or pumping cost.')
    candidates=[]
    for flow in np.linspace(.2,2,361):
        v=simulate(th,tc,mh,float(flow),actual_ua)
        candidates.append((abs(v['cold_out']-target),flow,v))
    error,bestflow,best=min(candidates,key=lambda x:x[0])
    a,b,c=st.columns(3)
    a.metric('Suggested cold flow',f'{bestflow:.3f} kg/s')
    b.metric('Predicted cold outlet',f'{best["cold_out"]:.2f} °C')
    c.metric('Target error',f'{error:.2f} °C')
    if error>1: st.warning('The target may be infeasible within the chosen flow limits and fixed exchanger conditions.')
with tabs[4]:
    st.subheader('Illustrative fouling impact')
    clean=simulate(th,tc,mh,mc,ua)
    st.metric('Heat duty reduction vs clean exchanger',f'{100*(1-r["heat_kw"]/clean["heat_kw"]):.1f}%')
    frows=[]
    for loss in range(0,71,5):
        v=simulate(th,tc,mh,mc,ua*(1-loss/100))
        frows.append({'Conductance loss (%)':loss,'Heat transfer (kW)':v['heat_kw'],'Cold outlet (°C)':v['cold_out']})
    fdf=pd.DataFrame(frows)
    fig=go.Figure(go.Scatter(x=fdf['Conductance loss (%)'],y=fdf['Heat transfer (kW)'],mode='lines+markers'))
    fig.update_layout(height=330,xaxis_title='Conductance loss (%)',yaxis_title='Heat transfer (kW)')
    st.plotly_chart(fig,use_container_width=True)
    st.caption('This changes UA directly to illustrate performance deterioration. It cannot diagnose real fouling without sensor measurements and a calibrated model.')
    st.write(f"Energy balance numerical residual: {r['balance_error_w']:.2e} W")
    st.download_button('⬇️ Download fouling scenario data',fdf.to_csv(index=False),'fouling_scenarios.csv','text/csv')
st.divider()
st.caption('HeatX AI · Student educational prototype · Do not use for industrial equipment design, safety decisions, or operational control without independent engineering validation.')
