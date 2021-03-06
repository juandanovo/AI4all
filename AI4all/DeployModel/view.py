  
import django
from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import json
from django.core.files.storage import FileSystemStorage
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt; plt.rcdefaults()
from plotly.offline import plot
from plotly.graph_objs import Scatter,bar,Pie,Histogram 
import plotly.graph_objects as go
import pandas as pd
#import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
fig_size = plt.rcParams["figure.figsize"]

# Set figure width to 12 and height to 9
fig_size[0] = 15
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size

# Set font size
plt.rc("font", size=14)


global df
global data
global features
global target
global column


def upload(request): 
    
    #return render(request, ['index1.html', 'home.html'],column)
    #return render(request, ['index1.html', 'home.html'])
    return render(request, 'home.html')

#new
# def clean_dataset(df):

#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

#     df.dropna(inplace=True)

#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

#     return df[indices_to_keep].astype(np.float64)
 
def index1(request):
    global df
    global fig
    
          
    # request.session.clear()
    if bool(request.FILES.get('document', False)) == True:
        uploaded_file = request.FILES['document']
        name = uploaded_file.name
        request.session['name'] = name
        #df = pd.read_csv(uploaded_file)
        #new
        #df = clean_dataset(df)
        #new MMVV

         
        sep = request.POST['separator']
        df = pd.read_csv(uploaded_file, sep = sep)
        
        dataFrame = df.to_json()
        request.session['df'] = dataFrame
        
        rows = len(df.index)
        request.session['rows'] = rows
        header = df.axes[1].values.tolist()
        request.session['header'] = header
        
        attributes = len(header)
        types = []
        maxs = []
        mins = []
        means = []
        # statistic attribut
        for i in range(len(header)):
            types.append(df[header[i]].dtypes)
            if df[header[i]].dtypes != 'object':
                maxs.append(df[header[i]].max())
                mins.append(df[header[i]].min())
                means.append(round(df[header[i]].mean(),2))
            else:
                maxs.append(0)
                mins.append(0)
                means.append(0)

        zipped_data = zip(header, types, maxs, mins, means)
        print(maxs)
        datas = df.values.tolist()
        data ={  
                "header": header,
                "headers": json.dumps(header),
                "name": name,
                "attributes": attributes,
                "rows": rows,
                "zipped_data": zipped_data,
                'df': datas,
                "type": types,
                "maxs": maxs,
                "mins": mins,
                "means": means,
            }
    else:
        name = 'None'
        attributes = 'None'
        rows = 'None'
        data ={
                "name": name,
                "attributes": attributes,
                "rows": rows,
            }
    return render(request, 'upload.html', data) 

            
def index2(request):
 
 
          
   
       return render(request, "upload.html", { 'plt':plt})

def home(request):
    return render(request, "home.html")

def base2(request):
    return render(request, "base2.html")

# method type
def methodtype(request):
    if 'classification' in request.POST:
        tmodel = ['knn','DTC','LOG']
    if 'regretion' in request.POST:
        tmodel = ['LIN','MLP','SVC']
        return render(request,"upload.html", tmodel)

# Models swichs
def result(request):
    if 'knn' in request.POST:
        return render(request, "knn.html")
    if 'DTC' in request.POST:
        return render(request, "Decision.html")
    if 'LOG' in request.POST:
        return render(request, "logisticreg.html")
    if 'LIN' in request.POST:
        return render(request, "linearreg.html")
    if 'MLP' in request.POST:
        return render(request, "MLPClassifier.html")
    if 'SVC' in request.POST:
        return render(request, "SVC_Classifier.html")
    if 'MULNB' in request.POST:
        return render(request, "naivebayes.html")
    if 'DTR' in request.POST:
        return render(request, "DecisionTreeREG.html")
    if 'km' in request.POST:
        return render(request,"kmeans.html")


def result1(request):
    global df
    global features
    global target

    lis = []
    lis.append(request.GET['SL'])

    s = lis[0].split(",")  

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.cluster import KMeans

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    from sklearn.preprocessing import StandardScaler

    x = df.iloc[:,1:-1].values
    y = df.iloc[:,-1].values
    
    x_data = df.iloc[:,-1]
    y_data = df.iloc[:,-1]

    # KNN
    if 'submit' in request.GET:

        s = lis[0].split(",")
        
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['neighbour'])
        hyp.append(request.GET['weights'])
        hyp.append(request.GET['algorithm'])
        hyp.append(request.GET['graph'])

         

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))
 
        cls = KNeighborsClassifier(n_neighbors=int(hyp[1]), weights=hyp[2], algorithm=hyp[3])
        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = KNeighborsClassifier(n_neighbors=int(hyp[1]), weights=hyp[2], algorithm=hyp[3])
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)


        u=hyp[4]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(x=y_data,y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25])],   output_type='div')
           #plot_div=plot([go.Box(y = [0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
    # 8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],output_type='div')

        return render(request, "knn.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})
       # return render(request, "knn.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div, 'confu_mx':confu_mx, 'classifi':classifi})
   # DECISION
    if 'submit1' in request.GET:

        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['criterion'])
        hyp.append(request.GET['splitter'])
        hyp.append(request.GET['max'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = DecisionTreeClassifier(criterion=hyp[1], splitter=hyp[2], max_features=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])
 # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = DecisionTreeClassifier(criterion=hyp[1], splitter=hyp[2], max_features=hyp[3])
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[4]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(x=y_data,y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25])],   output_type='div')
        return render(request, "Decision.html",{'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})
# NAIVE BAYES
    if 'submit2' in request.GET:

        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['alpha'])
        hyp.append(request.GET['fit'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = MultinomialNB(alpha=int(hyp[1]), fit_prior=bool(hyp[2]))

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])
 # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = SVC(C=2.0, kernel='linear')

        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[3]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
        return render(request, "naivebayes.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})

# LOGISTIC REGRESSION
    if 'submit3' in request.GET:
        
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['solver'])
        hyp.append(request.GET['penalty'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = float(hyp[0]))

        #penalty{???l1???, ???l2???, ???elasticnet???, ???none???}, default=???l2???
        cls = LogisticRegression(solver=hyp[1],penalty=hyp[2])

        cls.fit(x_train, np.ravel(y_train))
        
        y_pred = cls.predict([s])

      # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = LogisticRegression(solver=hyp[1],penalty=hyp[2])
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[3]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
        return render(request, "logisticreg.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})
# LINEAR REGRESSION
    if 'submit4' in request.GET:

        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['fit_intercept'])
        hyp.append(request.GET['normalize'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = LinearRegression(fit_intercept=bool(hyp[1]),normalize=bool(hyp[2]))

       
        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])
 # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = SVC(C=2.0, kernel='linear')
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[3]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
        return render(request, "linearreg.html",{'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})
# MPL CLASSIFIER
    if 'submit5' in request.GET:

        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['activation'])
        hyp.append(request.GET['solver'])
        hyp.append(request.GET['learning_rate'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = MLPClassifier(activation=hyp[1],solver=hyp[2],learning_rate=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

       # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = MLPClassifier(activation=hyp[1],solver=hyp[2],learning_rate=hyp[3])
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[4]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
        return render(request, "MLPClassifier.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})

# SVC CLASSIFIER
    if 'submit6' in request.GET:
        
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['C'])
        hyp.append(request.GET['kernel'])
        hyp.append(request.GET['gamma'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = SVC(C=float(hyp[1]),kernel=hyp[2],gamma=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])
 # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = SVC(C=float(hyp[1]),kernel=hyp[2],gamma=hyp[3])
 
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)

        u=hyp[4]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
        return render(request, "SVC_Classifier.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})

# DECIION TREE REG
    if 'submit7' in request.GET:
        
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['criterion'])
        hyp.append(request.GET['splitter'])
        hyp.append(request.GET['max'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = DecisionTreeRegressor(criterion=hyp[1],splitter=hyp[2],max_features=hyp[3])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

        # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model = DecisionTreeRegressor(criterion=hyp[1],splitter=hyp[2],max_features=hyp[3])

        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[4]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
        return render(request, "DecisionTreeREG.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})

# K MEANS
    if 'submit8' in request.GET:
        
        hyp = []
        hyp.append(request.GET['Split'])
        hyp.append(request.GET['n_clusters'])
        hyp.append(request.GET['algorithm'])
        hyp.append(request.GET['graph'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=float(hyp[0]))

        cls = KMeans(n_clusters=int(hyp[1]),algorithm=hyp[2])

        cls.fit(x_train, np.ravel(y_train))
        y_pred = cls.predict([s])

    # Finalize Model
        
        # prepare the model
        scaler = StandardScaler().fit(x_train)
        rescaledX = scaler.transform(x_train)
        model =KMeans(n_clusters=int(hyp[1]),algorithm=hyp[2])
        model.fit(rescaledX, y_train)
        # estimate accuracy on validation dataset
        rescaledValidationX = scaler.transform(x_test)
        predictions = model.predict(rescaledValidationX)
        print(accuracy_score(y_test, predictions)*100.0)
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


        print('Test ACCURACY is ', cls.score(x_test, y_test) * 100, '%')
        print('Train ACCURACY is ', cls.score(x_train, y_train) * 100, '%')
        acc = cls.score(x_test, y_test) * 100
        acc1 = cls.score(x_train,y_train)*100


        acc_score = accuracy_score(y_test, predictions)*100
        confu_mx = confusion_matrix(y_test, predictions)
        classifi = classification_report(y_test, predictions)
        u=hyp[3]

        if u == 'scatter':
            plot_div = plot([Scatter(x=x_data, y=y_data,marker_color='green',mode='markers')],output_type='div')
        if u == 'line':
            plot_div=plot([Scatter(x=x_data, y=y_data,marker_color='green')],output_type='div')
        if u == 'pie':
            plot_div=plot([Pie(labels=x_data, values=y_data)],output_type='div')
        if u == 'hist':
            plot_div=plot([Histogram(x=x_data)],output_type='div')
        if u == 'corr':
            plot_div=plot([go.Box(y=[0.75, 5.25, 5.5, 6, 6.2, 6.6, 6.80, 7.0, 7.2, 7.5, 7.5, 7.75, 8.15,
     8.15, 8.65, 8.93, 9.2, 9.5, 10, 10.25, 11.5, 12, 16, 20.90, 22.3, 23.25],x=y_data)],   output_type='div')
            
        return render(request, "kmeans.html", {'y_pred': y_pred, 'acc': acc,'acc1':acc1, 'plot_div': plot_div,  'acc_score':acc_score, 'confu_mx':confu_mx, 'classifi':classifi})

    if 'save_model' in request.POST:
        from sklearn.externals import joblib
        #filename = request.POST['filename']
        model = request.session['model']
        joblib.dump(model, 'AI4ALL_model')
        return render (request)


def Decision(request):
    return render(request, "Decision.html")
def naivebayes(request):
    return render(request, "naivebayes.html")
def logisticreg(request):
    return render(request, "logisticreg.html")
def linearreg(request):
    return render(request, "linearreg.html")
def MLPClassifier(request):
    return render(request, "MLPClassifier.html")
def SVC_Classifier(request):
    return render(request, "SVC_Classifier.html")
def DecisionTreeREG(request):
    return render(request, "DecisionTreeREG.html")
def kmeans(request):
    return render(request, "kmeans.html")


""" def download_model(request):
    if 'savemodel' in request.POST:
        # Save model
        from pickle import dump
        # save the model to disk
        filename = 'AI4ALL_model.sav'
        #filename = request.POST['filename']+'.sav'
        model = request.session['model']
        saved_model= dump(model, open(filename, 'wb'))
    return render (request, ["Decision.html","knn.thml", "kmeans.html"] ,saved_model) """

def save_model(request):
        if 'savemodel1' in request.POST:
            from pickle import dump
            model = request.session['model']
            filehandler = open('AI4ALL_model.sav', 'wb')
            dump(model, filehandler)
        #return render (request)
        return None