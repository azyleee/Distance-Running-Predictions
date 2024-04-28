
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats('png')

ONEDARK_THEME = {
    'red': '#e06c75',
    'blue': '#61afef',
    'orange': '#d19562',
    'green': '#8dc379',
    'bg': '#23272e',
    'txt': '#8b8d90',
    'txt_dark': '#1e2227'
    }
VSLIGHTPLUS_THEME = {
    'red': '#cd3131',
    'bg': '#ffffff',
    'txt': '#868686',
    'txt_dark': '#c9c9c9'
    }
GITHUB_THEME = {
    'red': '#ba2121',
    'bg': '#ffffff',
    'txt': '#212121',
    'txt_dark': '#b5b5b5'
    }

theme = ONEDARK_THEME

def choose_label(_name,dotname,axis):

    if _name is not None and dotname is not None:
        return _name
    elif _name is not None and dotname is None:
        return _name
    elif _name is None and dotname is not None:
        return dotname
    else:
        return axis

def my_scatter(x,y,z,
               c=theme['red'],
               cmap='viridis',
               aspectratio=dict(x=1,y=1,z=1),
               height=None,
               width=None,
               markersize=5
               ):
    '''
    Helper function that plots a scatter using plotly

    c: list or np.ndarray or pd.Series - the numerical value associated with the colour map
    cmap: string - colour map
    aspectratio: dict - aspect ratio 

    '''

    x = pd.Series(x)
    y = pd.Series(y)
    z = pd.Series(z)


    trace_data = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=markersize,
            color=c,
            colorscale=cmap,
            opacity=0.8
        )
    )

    fig = go.Figure(data = trace_data)

    fig.update_layout(
        autosize=False if not (height is None and width is None) else True,
        width=width,
        height=height,
        margin = dict(l=0,r=0,b=0,t=0),
        paper_bgcolor=theme['bg'],
        font_color = theme['txt'],
        scene = dict(
            xaxis = dict(     
                title = 'x' if x.name is None else x.name,
                backgroundcolor=theme['bg'],
                gridcolor=theme['txt'],
                showbackground=True,
                zerolinecolor=theme['txt'],
                titlefont=dict(family='Inter')
                # range=
            ),
            yaxis = dict(
                title = 'y' if y.name is None else y.name,
                backgroundcolor=theme['bg'],
                gridcolor=theme['txt'],
                showbackground=True,
                zerolinecolor=theme['txt'],
                titlefont=dict(family='Inter')
                # range=
            ),
            zaxis = dict(
                title = 'z' if z.name is None else z.name,
                backgroundcolor=theme['bg'],
                gridcolor=theme['txt'],
                showbackground=True,
                zerolinecolor=theme['txt'],
                titlefont=dict(family='Inter')
                # range=
            ),
            aspectratio=aspectratio,
            camera = dict(projection_type="orthographic")
        )
    )

    return fig

def my_interactable_barchart(x: pd.Series,                
                y: pd.Series,
                x_name=None,
                y_name=None,
                width=None,
                height=None,
                ):
    
    x=pd.Series(x)
    y=pd.Series(y)
    x_name = choose_label(x_name,x.name,'x')
    y_name = choose_label(y_name,y.name,'y')
    
    df= pd.DataFrame({x.name: x, y.name: y})

    fig = px.bar(df,x=x,y=y)

    fig.update_xaxes(gridcolor=theme['txt_dark'], title_text=x_name, title_font=dict(family='Inter'), linecolor=theme['txt'], linewidth=3)
    fig.update_yaxes(gridcolor=theme['txt_dark'], title_text=y_name, title_font=dict(family='Inter'))
    fig.update_traces(marker=dict(line=dict(color=theme['txt_dark'])))

    fig.update_layout(
        autosize=False if not (height is None and width is None) else True,
        width=width,
        height=height,
        margin = dict(l=0,r=0,b=0,t=0),
        paper_bgcolor=theme['bg'],
        plot_bgcolor=theme['bg'],
        font_color = theme['txt'],
    )

    return fig

def my_iteractable_cumulative(x, x_name = None, y_name = None):

    x = pd.Series(x)
    x_name = choose_label(x_name,x.name,'x')
    df = pd.DataFrame({x_name: x})

    y_name = 'Cumulative Distribution' if y_name is None else y_name

    fig = px.ecdf(df,x=x_name)

    fig.update_xaxes(gridcolor=theme['txt'], title_text=x_name, title_font=dict(family='Inter'), linecolor=theme['txt'], linewidth=3)
    fig.update_yaxes(gridcolor=theme['txt'], title_text=y_name, title_font=dict(family='Inter'))
    fig.update_traces(line=dict(color=theme['red'],width=5))


    fig.update_layout(
        margin = dict(l=50,r=50,b=50,t=50),
        paper_bgcolor=theme['bg'],
        plot_bgcolor=theme['bg'],
        font_color = theme['txt'],
    )

    return fig

def my_interactable_freqdist(x,x_name=None,height=None,width=None,n_bins=None):

    x=pd.Series(x)
    x_name = choose_label(x_name,x.name,'x')
    df=pd.DataFrame({x_name:x.values})

    fig = px.histogram(df,x_name,nbins=n_bins)
    
    fig.update_xaxes(gridcolor=theme['txt'], title_text=x_name, title_font=dict(family='Inter'), linecolor=theme['txt'], linewidth=3)
    fig.update_yaxes(gridcolor=theme['txt'], title_font=dict(family='Inter'))

    fig.update_traces(marker_color=theme['red'])
    

    fig.update_layout(
        margin = dict(l=50,r=50,b=50,t=50),
        autosize=True if (height is None and width is None) else False,
        width=width,
        height=height,
        paper_bgcolor=theme['bg'],
        plot_bgcolor=theme['bg'],
        font_color = theme['txt'],
    )

    return fig

def my_interactable_xyscatter(x,y,x_name=None,y_name=None,height=None,width=None):

    x=pd.Series(x)
    x_name = choose_label(x_name,x.name,'x')
    y=pd.Series(y)
    y_name = choose_label(y_name,y.name,'y')

    fig = go.Figure(data=go.Scatter(
        x=x, 
        y=y, 
        mode='markers',
        marker=dict(color=theme['red']))
    )  


    fig.update_xaxes(title_text=x_name, title_font=dict(family='Inter'), linecolor=theme['txt'])
    fig.update_yaxes(title_text=y_name, title_font=dict(family='Inter'))

    fig.update_layout(
        margin = dict(l=50,r=50,b=50,t=50),
        autosize=True if (height is None and width is None) else False,
        width=width,
        height=height,
        xaxis=dict(gridcolor=theme['txt'], zerolinecolor=theme['txt']),
        yaxis=dict(gridcolor=theme['txt'], zerolinecolor=theme['txt']),
        paper_bgcolor=theme['bg'],
        plot_bgcolor=theme['bg'],
        font_color = theme['txt'],
    )

    return fig

def my_interactable_xyline(y1, x1=None, y1_label=None, y2=None, x2=None, y2_label=None, x_label='x', y_label='y', title=''):
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name=y1_label, line=dict(color=theme['red'])))
    if y2 is not None:
        fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name=y2_label, line=dict(color=theme['blue'])))

    # Set labels
    fig.update_layout(
        title=title,
        xaxis_title=x_label, 
        yaxis_title=y_label,
        plot_bgcolor=theme['bg'],
        paper_bgcolor=theme['bg'],
        font=dict(
            family="Inter",
            color=theme['txt']
        )
    )

    fig.show()   

def my_xyscatter(x, y, x_name=None, y_name=None, height=4, width=6):
    x = pd.Series(x)
    y = pd.Series(y)
    x_name = choose_label(x_name, x.name, 'x')
    y_name = choose_label(y_name, y.name, 'y')

    fig, ax = plt.subplots(figsize=(width, height))

    ax.scatter(x, y, color=theme['red'])

    ax.set_xlabel(x_name, fontfamily='Inter', color=theme['txt'])
    ax.set_ylabel(y_name, fontfamily='Inter', color=theme['txt'])

    ax.spines['bottom'].set_color(theme['txt'])
    ax.spines['top'].set_color(theme['txt']) 
    ax.spines['right'].set_color(theme['txt'])
    ax.spines['left'].set_color(theme['txt'])

    ax.tick_params(axis='x', colors=theme['txt'])
    ax.tick_params(axis='y', colors=theme['txt'])

    ax.set_axisbelow(True)
    ax.grid(True, color=theme['txt'], linestyle='--', linewidth=0.5)

    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['bg'])

    return fig, ax

    
def my_freqdist(x, x_name=None, height=4, width=8, n_bins=20, xlim:list=None, ylim:list=None):
    '''
    Helper function that plots a histogram using matplotlib.pyplot

    x: list or np.ndarray or pd.Series - data values
    x_name: string - label for x-axis
    height: float - figure height
    width: float - figure width
    n_bins: int - number of bins for the histogram
    '''

    x = pd.Series(x)
    x_name = choose_label(x_name, x.name, 'x')

    fig, ax = plt.subplots(figsize=(width,height))

    counts, bins, _ = ax.hist(x, bins=n_bins, color=theme['red'], edgecolor=theme['txt_dark'], linewidth=1.5, zorder=2)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(x_name, fontfamily='Inter', color=theme['txt'])
    ax.set_ylabel('Frequency', fontfamily='Inter', color=theme['txt'])

    ax.grid(color=theme['txt'], linestyle='--', linewidth=0.5, zorder=1)

    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['bg'])

    for tick in ax.get_xticklabels():
        tick.set_fontfamily('Inter')
        tick.set_color(theme['txt'])
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Inter')
        tick.set_color(theme['txt'])

    for spine in ax.spines.values():
        spine.set_edgecolor(theme['txt'])

    plt.show()

def my_barchart(x, y, x_name=None, y_name=None, width=None, height=None):
    x = pd.Series(x)
    y = pd.Series(y)
    x_name = choose_label(x_name, x.name, 'x')
    y_name = choose_label(y_name, y.name, 'y')

    fig, ax = plt.subplots(figsize=(width, height))

    ax.bar(x, y, color=theme['bg'], edgecolor=theme['txt_dark'], linewidth=3)

    ax.set_xlabel(x_name, fontfamily='Inter', color=theme['txt'])
    ax.set_ylabel(y_name, fontfamily='Inter', color=theme['txt'])
    ax.set_facecolor(theme['bg'])
    fig.patch.set_facecolor(theme['bg'])

    ax.spines['bottom'].set_color(theme['txt'])
    ax.spines['top'].set_color(theme['txt']) 
    ax.spines['right'].set_color(theme['txt'])
    ax.spines['left'].set_color(theme['txt'])

    ax.tick_params(axis='x', colors=theme['txt'])
    ax.tick_params(axis='y', colors=theme['txt'])

    plt.show()

def my_cumulative(x, x_name=None, y_name=None):
    x = pd.Series(x)
    x_name = choose_label(x_name, x.name, 'x')
    y_name = 'Cumulative Distribution' if y_name is None else y_name

    fig, ax = plt.subplots()

    # Calculate the ECDF
    x_sorted = np.sort(x)
    y_values = np.arange(1, len(x_sorted)+1) / len(x_sorted)

    ax.plot(x_sorted, y_values, color=theme['red'], linewidth=5)

    ax.set_xlabel(x_name, fontfamily='Inter', color=theme['txt'])
    ax.set_ylabel(y_name, fontfamily='Inter', color=theme['txt'])

    ax.spines['bottom'].set_color(theme['txt'])
    ax.spines['top'].set_color(theme['txt']) 
    ax.spines['right'].set_color(theme['txt'])
    ax.spines['left'].set_color(theme['txt'])

    ax.tick_params(axis='x', colors=theme['txt'])
    ax.tick_params(axis='y', colors=theme['txt'])

    # Set grid lines behind other elements
    ax.set_axisbelow(True)
    ax.grid(True, color=theme['txt'], linestyle='--', linewidth=0.5)

    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['bg'])

    plt.show()

def my_xyline(y1, x1=None, y1_label=None,
              x2=None, y2=None, y2_label=None,
              x3=None, y3=None, y3_label=None,
              x_label='x', y_label='y',
              xlim=None, ylim=None):
    fig, ax = plt.subplots()
    if x1 is not None:
        plt.plot(x1,y1, label=y1_label, color=theme['red'])
    else:
        plt.plot(y1, label=y1_label, color=theme['red'])
    if y2 is not None:
        if x2 is not None:
            plt.plot(x2,y2, label=y2_label, color=theme['blue'])
        else:
            plt.plot(y2, label=y2_label, color=theme['blue'])
    if y3 is not None:
        if x3 is not None:
            plt.plot(x3,y3, label=y3_label, color=theme['green'])
        else:
            plt.plot(y3, label=y3_label, color=theme['green'])

    legend = plt.legend(facecolor=theme['txt'], prop={'family':'Inter'})

    ax.set_xlabel(x_label, fontfamily='Inter', color=theme['txt'])
    ax.set_ylabel(y_label, fontfamily='Inter', color=theme['txt'])

    plt.xlim(xlim)
    plt.ylim(ylim)

    ax.spines['bottom'].set_color(theme['txt'])
    ax.spines['top'].set_color(theme['txt']) 
    ax.spines['right'].set_color(theme['txt'])
    ax.spines['left'].set_color(theme['txt'])

    ax.tick_params(axis='x', colors=theme['txt'])
    ax.tick_params(axis='y', colors=theme['txt'])

    ax.set_axisbelow(True)
    ax.grid(True, color=theme['txt'], linestyle='--', linewidth=0.5)

    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['bg'])


def my_traintestpredictions(x_train=None, y_train=None, y_train_pred=None,
                            x_test=None, y_test=None, y_test_pred=None,
                            x_name='x-variable', y_name = 'Time'):
    '''
    Plots the training against predicted values for both the test and train sets Assuming only 1 y-varaible which is time.
    '''

    fig, ax = plt.subplots()
    if x_train is not None:
        plt.scatter(x_train, y_train, label='Train, True', color=theme['red'])
    if y_train_pred is not None:
        plt.scatter(x_train, y_train_pred, label='Train, Predicted', color=theme['blue'])
    if y_test is not None:
        plt.scatter(x_test, y_test, label='Test, True', color=theme['orange'])
    if y_test_pred is not None:
        plt.scatter(x_test, y_test_pred, label='Test, Predicted', color=theme['green'])

    legend = plt.legend(facecolor=theme['txt'], prop={'family':'Inter'})

    ax.set_xlabel(x_name, fontfamily='Inter', color=theme['txt'])
    ax.set_ylabel(y_name, fontfamily='Inter', color=theme['txt'])

    ax.spines['bottom'].set_color(theme['txt'])
    ax.spines['top'].set_color(theme['txt']) 
    ax.spines['right'].set_color(theme['txt'])
    ax.spines['left'].set_color(theme['txt'])

    ax.tick_params(axis='x', colors=theme['txt'])
    ax.tick_params(axis='y', colors=theme['txt'])

    ax.set_axisbelow(True)
    ax.grid(True, color=theme['txt'], linestyle='--', linewidth=0.5)

    fig.patch.set_facecolor(theme['bg'])
    ax.set_facecolor(theme['bg'])    