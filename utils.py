import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
np.random.seed(seed=4)


def circle(radius, num_points = 1000):
    x1 = np.linspace(0, radius, num_points)
    x2 = np.sqrt(1 - x1**2)
    return(x1, x2)


def convert_to_training_data(x_in, y_in, x_out, y_out):
    num_samples=len(x_in)+len(x_out)
    inputs_x=np.array(x_in.tolist()+x_out.tolist()).reshape((num_samples,1))
    inputs_y=np.array(y_in.tolist()+y_out.tolist()).reshape((num_samples,1))
    inputs=np.column_stack((np.ones((num_samples)),inputs_x,inputs_y))
    #inputs=np.column_stack((np.ones((num)),inputs))
    labels=np.array([0.0 for xx in x_in]+[1.0 for xy in x_out]).reshape((num_samples,1))

    return(inputs, labels)



def test_model(theta, inputs, predictor):
    x=inputs.T[1]
    y=inputs.T[2]

    # test the predictor on input
    trues=0
    falses=0
    for idx,xx in enumerate(x):
        yy=y[idx]
        p = predictor(theta, np.array([1.0, xx, yy]))
        if p<0.5 and xx**2+yy**2<=1.0:
            trues+=1
        elif p>0.5 and xx**2+yy**2>1.0:
            trues+=1
        else:
            falses+=1

    accuracy_training=float(trues)/float(trues+falses)*100.0
    print("Training accuracy: %.3f %% (%i of %i points are correct)"%(accuracy_training,trues,trues+falses))


    trues=0
    falses=0
    for xx in np.linspace(0, 1, 100):
        for yy in np.linspace(0, 1, 100):
            p = predictor(theta, np.array([1.0, xx,yy]))
            if p<0.5 and xx**2+yy**2<=1.0:
                trues+=1
            elif p>0.5 and xx**2+yy**2>1.0:
                trues+=1
            else:
                falses+=1

    accuracy_test=float(trues)/float(trues+falses)*100.0
    print("Test accuracy:     %.3f %% (%i of %i points are correct)"%(accuracy_test,trues,trues+falses))


    x1_test=[]
    x2_test=[]
    predictions=[]
    for xx in np.linspace(0, 1, 25):
        for yy in np.linspace(0, 1, 25):
            x1_test.append(xx)
            x2_test.append(yy)
            p=predictor(theta, [1.0, xx,yy])
            predictions.append(p)
    predictions=np.array(predictions)

    return(accuracy_training, accuracy_test, x1_test, x2_test, predictions)



def test_model_nn(inputs, net):
    x=inputs.T[1]
    y=inputs.T[2]

    # test the predictor on input
    trues=0
    falses=0
    for idx,xx in enumerate(x):
        yy=y[idx]
        p = net(torch.tensor([1.0, xx, yy], dtype=torch.float32))
        if p<0.5 and xx**2+yy**2<=1.0:
            trues+=1
        elif p>0.5 and xx**2+yy**2>1.0:
            trues+=1
        else:
            falses+=1

    accuracy_training=float(trues)/float(trues+falses)*100.0
    print("Training accuracy: %.3f %% (%i of %i points are correct)"%(accuracy_training,trues,trues+falses))


    trues=0
    falses=0
    for xx in np.linspace(0, 1, 100):
        for yy in np.linspace(0, 1, 100):
            p = net(torch.tensor([1.0, xx,yy], dtype=torch.float32))
            if p<0.5 and xx**2+yy**2<=1.0:
                trues+=1
            elif p>0.5 and xx**2+yy**2>1.0:
                trues+=1
            else:
                falses+=1

    accuracy_test=float(trues)/float(trues+falses)*100.0
    print("Test accuracy:     %.3f %% (%i of %i points are correct)"%(accuracy_test,trues,trues+falses))


    x1_test=[]
    x2_test=[]
    predictions=[]
    for xx in np.linspace(0, 1, 25):
        for yy in np.linspace(0, 1, 25):
            x1_test.append(xx)
            x2_test.append(yy)
            p=net(torch.tensor([1.0, xx,yy], dtype=torch.float32))
            predictions.append(p)
    predictions=np.array(predictions)

    return(accuracy_training, accuracy_test, x1_test, x2_test, predictions)





def plot_points(x_circ, y_circ, x_in, y_in, x_out, y_out):

    plt.figure(figsize=(8,8))
    plt.plot(x_circ, y_circ, color = 'grey', lw = 5)
    plt.plot(x_in, y_in, 'o', color = 'b', label="inside")
    plt.plot(x_out, y_out, 'o', color = 'r', label="outside")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    l=plt.legend(loc = "lower left", ncol = 1, fontsize=20)


def plot_predictions(x_circ, y_circ, x_in, y_in, x_out, y_out, x1_test, x2_test, predictions):
    min_p=np.min(predictions)
    max_p=np.max(predictions)
    if abs(0.5-min_p)>abs(max_p-0.5):
        max_p=0.5+abs(0.5-min_p)
    else:
        min_p=0.5-abs(max_p-0.5)

    plt.figure(figsize=(8,8))
    plt.plot(x_circ, y_circ, color = 'grey', lw = 5)
    plt.plot(x_in, y_in, 'o', color = 'b', label="inside")
    plt.plot(x_out, y_out, 'o', color = 'r', label="outside")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    l=plt.legend(loc = "lower left", ncol = 1, fontsize=20)
    plt.scatter(x1_test, x2_test, marker='s', c = predictions, alpha=0.5, cmap="bwr", vmin=min_p, vmax=max_p, s=200)



def plot_predictions_nn(x_circ, y_circ, x_in, y_in, x_out, y_out, x1_test, x2_test, predictions):
    min_p=np.min(predictions)
    max_p=np.max(predictions)
    if abs(0.5-min_p)>abs(max_p-0.5):
        max_p=0.5+abs(0.5-min_p)
    else:
        min_p=0.5-abs(max_p-0.5)
    min_p=0.0
    max_p=1.0

    plt.figure(figsize=(8,8))
    plt.plot(x_circ, y_circ, color = 'grey', lw = 5)
    plt.plot(x_in, y_in, 'o', color = 'b', label="inside")
    plt.plot(x_out, y_out, 'o', color = 'r', label="outside")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("$x_1$", fontsize=20)
    plt.ylabel("$x_2$", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    l=plt.legend(loc = "lower left", ncol = 1, fontsize=20)
    plt.scatter(x1_test, x2_test, marker='s', c = predictions, alpha=0.5, cmap="bwr", vmin=min_p, vmax=max_p, s=200)



