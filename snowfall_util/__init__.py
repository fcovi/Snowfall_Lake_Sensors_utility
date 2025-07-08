from io import StringIO
import json
import numpy as np
import pandas as pd
import pickle

import csv
import xcsv
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdb


def pre_process_transmitted(file_path):
    """
    Function that pre process the transmitted data archived in the SAMBA drive
    INPUT: path to the data file
    OUTPUT: pandas dataframe with the pre processed data 
    """

    def _linear_transform(x, scaling=1.0, offset=0.0):
        """ Linearly transform the given variable """
        return x * scaling + offset


    DEFAULTS = {
    'col_names_4': ['datetime', 'pressure', 'ground_temperature', 'battery_voltage'],
    'col_names_5': ['datetime', 'pressure', 'ground_temperature', 'battery_voltage', 'water_temperature'],
    'conversions': {'datetime': {'function': pd.to_datetime, 'kwargs': {'format': '%y%j%H'}}},
    'calibrations': {
        # 'pressure': {'function': _linear_transform, 'kwargs': {'scaling': 1.0e-7 * 10.1974, 'offset': 0.0}},
        'pressure': {'function': _linear_transform, 'kwargs': {'scaling': 1.0e-8 * 10.1974, 'offset': 0.0}},
        'ground_temperature': {'function': _linear_transform, 'kwargs': {'scaling': 0.1, 'offset': 0.0}},
        'water_temperature': {'function': _linear_transform, 'kwargs': {'scaling': 0.01, 'offset': 0.0}},
        'battery_voltage': {'function': _linear_transform, 'kwargs': {'scaling': 0.1, 'offset': 0.0}}
         },
    'index_col_names': ['datetime']
    }

    # read input data in pandas dataframe
    df_input = pd.read_csv(file_path, header=None, names=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21'])

    # TAKE CARE of PROBLEMS with kitka_f
    if file_path == '/run/user/1001/gvfs/smb-share:server=samba.nerc-bas.ac.uk,share=data/glacio/ltms/pub/snowfall//kitka_f/data/300534060110910/snowfall_kitka_f_300534060110910_archive.csv':
        df_input.drop(index=df_input.index[0], axis=0, inplace=True)

    # # remove the last column which is empty due to the comma
    # tmp = df_input.iloc[: , :-1]   

    # # split each row, which contains 4 sets of data, into 4 rows 
    # rows = []
    # ncols = 4
    # for row in tmp.values:
    #     n = len(row) // ncols
    #     [rows.append(row[i*ncols:(i+1)*ncols]) for i in range(n)]
        

    # # create new pandas dataframe
    # df_output = pd.DataFrame.from_records(rows, columns=DEFAULTS['col_names'])

    rows = []

    for i in range(0, len(df_input.index)):
        tmp = df_input.iloc[i].dropna()
        
        if len(tmp) == 16:
            rows.append(tmp.iloc[0:4].values)
            rows.append(tmp.iloc[4:8].values)
            rows.append(tmp.iloc[8:12].values)
            rows.append(tmp.iloc[12:16].values)
            
        elif len(tmp) == 20:
            rows.append(tmp.iloc[0:5].values)
            rows.append(tmp.iloc[5:10].values)
            rows.append(tmp.iloc[10:15].values)
            rows.append(tmp.iloc[15:20].values)
            
    if len(tmp) == 16:
        df = pd.DataFrame.from_records(rows, columns=DEFAULTS['col_names_4'])
        df['water_temperature'] = np.nan
    elif len(tmp) == 20:
        df = pd.DataFrame.from_records(rows, columns=DEFAULTS['col_names_5'])

    # SEPARATE DIFFERENT TRANSMITTED VALUES
    # 4 variable transmitted (no water temperature)
    df16 = df.loc[df['water_temperature'].isnull()]

    # 5 variable transmitted (with water temperature)
    df20 = df.loc[df['water_temperature'].notnull()]
    df20 = df20.drop(df20.loc[df20.pressure == '00NAN'].index)         # drop eventual rows with NaN in pressure
    df20 = df20.drop(df20.loc[df20.water_temperature == 'NAN'].index)  # drop eventual rows with NaN in water_temperature

    # adjust new 5 variable format
    battery_voltage = df20['water_temperature'].copy()
    water_temperature = df20['ground_temperature'].copy()
    ground_temperature = df20['battery_voltage'].copy()

    df20['water_temperature'] = water_temperature.copy()
    df20['ground_temperature'] = ground_temperature.copy()
    df20['battery_voltage'] = battery_voltage.copy()

    for ii in range(0, len(df20)):
        # pdb.set_trace()
        df20.loc[df20.index[ii], 'datetime'] = '2'+str(df20.loc[df20.index[ii], 'datetime'])
        df20.loc[df20.index[ii], 'pressure'] = int(df20.loc[df20.index[ii], 'pressure'])*100
        if float(df20.loc[df20.index[ii], 'battery_voltage']) <= 50:
            df20.loc[df20.index[ii], 'battery_voltage'] = '1'+str(df20.loc[df20.index[ii], 'battery_voltage'])

    # remerge dataframe
    frames = [df16, df20]
    df_output = pd.concat(frames)

    # convert date to datetime
    conversions = DEFAULTS['conversions']
    for key in conversions:
        df_output[key] = conversions[key]['function'](df_output[key], **conversions[key]['kwargs'])

    # calibrate data (to proper units)
    calibrations = DEFAULTS['calibrations']
    for key in calibrations:
        df_output[key] = pd.to_numeric(df_output[key], errors='coerce')
        df_output[key] = calibrations[key]['function'](df_output[key].astype(float), **calibrations[key]['kwargs'])

    # set, sort, deduplicate dataframe index
    df_output.set_index(DEFAULTS['index_col_names'], inplace=True)
    df_output.sort_index(inplace=True)
    df_output = df_output[~df_output.index.duplicated(keep='first')]

    # resample, to introduce NaN when no measurement is available
    df_output = df_output.resample('60min').asfreq()

    # write dataframe to output file
    df_output['pressure'] = df_output['pressure'].map('{:.7f}'.format)
    df_output['pressure'] = pd.to_numeric(df_output['pressure'], errors='coerce')
    df_output['ground_temperature'] = df_output['ground_temperature'].map('{:.2f}'.format)
    df_output['ground_temperature'] = pd.to_numeric(df_output['ground_temperature'], errors='coerce')
    df_output['battery_voltage'] = df_output['battery_voltage'].map('{:.2f}'.format)
    df_output['battery_voltage'] = pd.to_numeric(df_output['battery_voltage'], errors='coerce')
    df_output['water_temperature'] = df_output['water_temperature'].map('{:.2f}'.format)
    df_output['water_temperature'] = pd.to_numeric(df_output['water_temperature'], errors='coerce')
    # df_output.to_csv(out_file_path)  

    return df_output


def plot_raw(df, name='', show=True, saveHTML=False):
    """
    Plot the raw data of pressure and, if available, ground and water temperature as well as battery voltage
    It also save the raw data plotted into a .csv file in the input format used in the WebApp

    :param df: input data
    :type: pandas DataFrame

    :returns: a plotly figure object
    :rtype: plotly figure object
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])
  
    # plot pressure 
    fig.add_trace(go.Scatter(x=df.index, y=df['pressure'], name='pressure', showlegend=False, 
                                mode='lines', line = dict(color='black', width=1, dash='solid'), 
                                hovertemplate='P: %{y:.3f} m w.e.<extra></extra>'), secondary_y=True)
    
    # plot 0C temperature line
    if 'water_temperature' in df.keys() or 'ground_temperature' in df.keys(): 
        x = [df.index[0], df.index[-1]]
        y = [0, 0]
        fig.add_trace(go.Scatter(x=x, y=y, showlegend=False, mode='lines', line = dict(color='black', width=1, dash='dash'), 
                                    hoverinfo='none'), secondary_y=False)   
    # plot water temperature 
    if 'water_temperature' in df.keys(): 
        fig.add_trace(go.Scatter(x=df.index, y=df['water_temperature'], name='T water', showlegend=True, 
                                    mode='lines', line = dict(color='blue', width=1, dash='solid'), 
                                    hovertemplate='T water: %{y:.2f} (\u00B0C)'), secondary_y=False)
    # plot ground temperature 
    if 'ground_temperature' in df.keys(): 
        fig.add_trace(go.Scatter(x=df.index, y=df['ground_temperature'], name='T ground', showlegend=True, 
                                    mode='lines', line = dict(color='red', width=1, dash='solid'), 
                                    hovertemplate='%{y:.2f} (\u00B0C)'), secondary_y=False)
    # plot battery voltage 
    if 'battery_voltage' in df.keys(): 
        fig.add_trace(go.Scatter(x=df.index, y=df['battery_voltage'], name='battery voltage', showlegend=True, 
                                    mode='lines+markers', line = dict(color='grey', width=1, dash='solid'), marker=dict(symbol='cross', size=3), 
                                    hovertemplate='%{y:.2f} (V)'), secondary_y=False)
    
    # update axes
    fig.update_xaxes(hoverformat = "%d %b %y <br> %H:%M")
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='pressure (m w.e.)', secondary_y=True)
    if 'water_temperature' in df.keys() or 'ground_temperature' in df.keys(): 
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='temperature (\u00B0C)', secondary_y=False)
    
    # update figure layout
    fig.update_layout(
        title={'text': '%s' % (name), 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis2={'anchor': 'x', 'overlaying': 'y', 'side': 'left'},
        yaxis={'anchor': 'x', 'domain': [0.0, 1.0], 'side':'right'},
        xaxis_title='',
        legend=dict(yanchor="top", y=1., xanchor="left", x=0, orientation="h", traceorder='reversed'),
        legend_title='',
        font=dict(size=18),   
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x',
        margin=dict(l=20, r=20, t=40, b=20))
    
    if show:
        fig.show()
    if saveHTML == True:
        fig.write_html('%s_raw.html' % (name))
    elif type(saveHTML) == str:
        fig.write_html('%s/%s_raw.html' % (saveHTML, name))

    # # save the whole raw transmitted data series formatted to use as input in the WebApp
    # # df.pressure *= 0.0980665  # convert water pressure from m w.e. to bar 
    # # currently need this exact format
    # dfout = df.copy()
    # dfout['RECORD'] = np.nan
    # # dfout['datetime'] = dfout.index
    # # dfout['datetime'] = dfout.index.strftime(('"%Y-%m-%d %H:%M:%S"'))
    # dfout['timestamp'] = dfout.index.strftime(('"%Y-%m-%d %H:%M:%S"')).astype(pd.StringDtype())
    # # dfout['datetime'] = dfout['datetime'].apply(lambda x: '"' + str(x) + '"')
    # dfout['pressure'] /= 1.0e-8 * 10.1974
    # dfout['ground_temperature'] /= 0.1
    # dfout['water_temperature'] /= 0.01
    # dfout['battery_voltage'] /= 0.1
    # dfout = dfout.reindex(columns=['timestamp', 'RECORD', 'pressure', 'water_temperature', 'ground_temperature', 'battery_voltage'])
    # dfout.columns = pd.MultiIndex.from_tuples(zip(['"', '', '', '', '', ''], ['', '', '', '', '', ''], ['', '', '', '', '', ''], dfout.columns))

    # if saveHTML == True:
    #     print('Saving WebApp input formatted transmitted data to .csv')
    #     dfout.to_csv('%s_transmitted_data.csv' % (name), index=False, quoting=csv.QUOTE_NONE)
    # elif type(saveHTML) == str:
    #     print('Saving WebApp input formatted transmitted data to .csv')
    #     dfout.to_csv('%s/transmitted_data/%s_transmitted_data.csv' % (saveHTML, name), index=False, quoting=csv.QUOTE_NONE)

    return fig


def plot_colors():
    """
    Define colors (and transparencies) use in the plots

    :returns: a dictionary with colors
    :rtype: dict
    """

    # T_alpha = 0.75
    # Tg_c = 'rgba(255,128,0,%.2f)' % T_alpha
    # Tw_c = 'rgba(0,0,255,%.2f)' % T_alpha

    alpha = 0.1
    green = 'rgba(0,128,0,1)'
    green_a = 'rgba(0,128,0,%.2f)' % alpha
    red = 'rgba(224,22,61,1)'
    red_a = 'rgba(224,22,61,%.2f)' % alpha
    purple = 'rgba(162,0,255,1)'
    purple_a = 'rgba(162,0,255,%.2f)' % alpha
    orange = 'rgba(252,186,3,1)'
    orange_a = 'rgba(252,186,3,%.2f)' % alpha
    grey = 'rgba(3, 140, 252,1)'
    grey_a = 'rgba(3, 140, 252,0.5)'

    colors = {
        'green': green,
        'green_a': green_a,
        'red': red,
        'red_a': red_a,
        'purple': purple,
        'purple_a': purple_a,
        'orange': orange,
        'orange_a': orange_a,
        'grey': grey,
        'grey_a': grey_a
    }

    return colors


def plot_processed_hourly(p, name='', start=None, end=None, hours=1, show=True, saveHTML=False):
    """
    Plot the processed data showing start and end of snowfall, total accumulation and 
    hourly rates, together with before, corrected before, and after gradients used 
    in the processing

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :start: datetime at which starting the plot
    :type: datetime date 
    :end: time at which stopping the plot
    :type: datetime date

    :returns: a plotly figure object
    :rtype: plotly figure object
    """
    
    # get data from snowfall processor
    data = p.input
    extrema = p.breakpoints['extrema']
    maxima = extrema.dropna(subset=['maxima']) 
    trends = p.breakpoints['trends']
    differences = p.breakpoints['differences']
    hourly = p.breakpoints['differences_hourly']

    # subset data
    if start:
        data = data[start:]
    if end:
        data = data[:end]

    # get colors
    colors = plot_colors()

    # define config lists with legends and colors
    extrema_conf = [{'name': 'start', 'l': 'minima', 'c': colors['red']}, 
                    {'name': 'end', 'l': 'maxima', 'c': colors['green']}]
    trends_conf = [{'l': 'before', 'c': colors['orange'], 'ls': 'dot', 'label': 'before'}, 
                    {'l': 'p_corrected_before', 'c': colors['red'], 'ls': 'dash', 'label': 'corrected before'}, 
                    {'l': 'after', 'c': colors['green'], 'ls': 'dash', 'label': 'after'}]
    trends_conf_fill = [{'l': 'before', 'c': colors['orange_a'], 'ls': 'dot'},
                        {'l': 'p_corrected_before', 'c': colors['red_a'], 'ls': 'dash'}, 
                        {'l': 'after', 'c': colors['green_a'], 'ls': 'dash'}]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # plot hourly rates in the background
    if hours == 1:
        for i in range(0, len(hourly)):
            bars = hourly[i].resample('%sH' % str(hours), label='right', closed='right').sum(numeric_only=True)
            bars[bars < 0] = 0  # reset negative precipitation
            if i == 0:
                fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                        width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                        name='hourly rates', marker_color=colors['grey_a'], showlegend=True, 
                                        hovertemplate='hourly rate: %{y:.1f} mm w.e.<extra></extra>'), secondary_y=False)
            else:
                fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                        width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                        name='hourly rates', marker_color=colors['grey_a'], showlegend=False, 
                                        hovertemplate='hourly rate: %{y:.1f} mm w.e.<extra></extra>'), secondary_y=False)
    else:
        label_str = '%.f hourly rate' % hours
        for i in range(0, len(hourly)):
            bars = hourly[i].resample('%sH' % str(hours), label='right', closed='right').sum(numeric_only=True)
            bars[bars < 0] = 0  # reset negative precipitation
            if i == 0:
                fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                        width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                        name=label_str, marker_color=colors['grey_a'], showlegend=True, 
                                        hovertemplate='%{y:.1f} mm w.e.<extra></extra>'), secondary_y=False)
            else:
                fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                        width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                        name=label_str, marker_color=colors['grey_a'], showlegend=False, 
                                        hovertemplate='%{y:.1f} mm w.e.<extra></extra>'), secondary_y=False)
  
    # plot pressure 
    fig.add_trace(go.Scatter(x=data.index, y=data['pressure'], name='pressure', showlegend=False, 
                                mode='lines', line = dict(color='black', width=1, dash='solid'), 
                                hovertemplate='P: %{y:.3f} m w.e.<extra></extra>'), secondary_y=True)
    try:
        p.input_raw
        fig.add_trace(go.Scatter(x=p.input_raw.index, y=p.input_raw['pressure'], name='pressure', showlegend=False, 
                        mode='lines', line = dict(color='rgba(0,0,0,0.3)', width=1, dash='solid'), 
                        hoverinfo='skip'), secondary_y=True)
    except AttributeError:
        pass
        
    # plot start and end of snowfall events
    for conf in extrema_conf:
        fig.add_trace(go.Scatter(x=extrema.index, y=extrema[conf['l']], name=conf['name'], showlegend=True, 
                                    mode='markers', marker_symbol='circle', marker_color=conf['c'], marker_size=7, 
                                    hoverinfo='skip'), secondary_y=True)
            
    # display total snowfall accumulation in hoverinfo
    snow = p.breakpoints['status'][p.breakpoints['status']['type'] == 'end']['corrected_delta']
    unc = p.breakpoints['status'][p.breakpoints['status']['type'] == 'end']['uncertainty']
    customsnow = p.breakpoints['extrema']['maxima'].copy()
    customsnow[snow.index] = snow.values*1000.
    customunc = p.breakpoints['extrema']['maxima'].copy()
    customunc[unc.index] = unc.values*1000.
    fig.add_trace(go.Scatter(x=p.breakpoints['extrema']['maxima'].index, y=p.breakpoints['extrema']['maxima'], 
                                customdata=np.stack((customsnow, customunc), axis=-1), 
                                showlegend=False, mode='markers', marker_symbol='circle', marker_color=colors['green'], marker_size=1, 
                                hovertemplate = '<b>Total snowfall</b>: %{customdata[0]:.2f} &plusmn; %{customdata[1]:.2f} mm w.e.<extra></extra>'), 
                                secondary_y=True)
    
    # plot before/corrected_before/after trends and fill in the uncertainty spread
    for conf in trends_conf:
        for i, regression in enumerate(trends[conf['l']]):
            if i == 0:
                fig.add_trace(go.Scatter(x=regression.index, y=regression['fit'], name=conf['label'], showlegend=True, 
                                         mode='lines', line = dict(color=conf['c'], width=1, dash=conf['ls']), hoverinfo='skip'), secondary_y=True)
            else:   
                fig.add_trace(go.Scatter(x=regression.index, y=regression['fit'], name=conf['label'], showlegend=False, 
                                         mode='lines', line = dict(color=conf['c'], width=1, dash=conf['ls']), hoverinfo='skip'), secondary_y=True)        
    for conf in trends_conf_fill:
        for i, regression in enumerate(trends[conf['l']]):     
            fig.add_trace(go.Scatter(x=regression.index, y=regression['err_upper'], showlegend=False,
                                     fill=None, mode='lines', line=dict(color=conf['c'], width=0.1,), hoverinfo='skip'), secondary_y=True)
            fig.add_trace(go.Scatter(x=regression.index, y=regression['err_lower'], showlegend=False,  fill='tonexty', fillcolor=conf['c'], 
                                     mode='lines', line=dict(color=conf['c'], width=0.1), hoverinfo='skip'), secondary_y=True)

    # plot the differences as vertical lines      
    x = pd.to_datetime(differences['t2'].values)
    conf = next(i for i in trends_conf if i['l'] == 'p_corrected_before')    
    for xx in range(0, len(x)):
        ymin = trends['p_corrected_before'][xx].fit[-1]
        delta = differences['corrected_delta'][xx]
        fig.add_trace(go.Scatter(x=[x[xx], x[xx]], y=[ymin, ymin+delta], showlegend=False, 
                                    mode='lines', line=dict(color=conf['c'], width=1, dash='solid'), 
                                    hoverinfo='skip'), secondary_y=True)
        
    # plot flagged events
    small = p.breakpoints['events_windows'][p.breakpoints['events_windows'].flag == 'small and short event']
    for i in range(0, len(small)):
        if i == 0:
            fig.add_vrect(x0=small.iloc[i].start, x1=small.iloc[i].end, showlegend=True, name='flagged events', line_width=0, 
                          fillcolor="orange", opacity=0.75, layer='below', row=1, col=1)
        else:
            fig.add_vrect(x0=small.iloc[i].start, x1=small.iloc[i].end, line_width=0, 
                          fillcolor="orange", opacity=0.75, layer='below', row=1, col=1)
    large = p.breakpoints['events_windows'][p.breakpoints['events_windows'].flag == 'probable calving or avalanche']
    for i in range(0, len(large)):
        if i == 0:
            fig.add_vrect(x0=large.iloc[i].start, x1=large.iloc[i].end, showlegend=True, name='calving/avalanche', line_width=0, 
                          fillcolor="purple", opacity=0.75, layer='below', row=1, col=1)
        else:
            fig.add_vrect(x0=large.iloc[i].start, x1=large.iloc[i].end, line_width=0, 
                          fillcolor="purple", opacity=0.75, layer='below', row=1, col=1)


    # update axes
    fig.update_xaxes(hoverformat = "%d %b %y <br> %H:%M")
    if hours == 1:
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='hourly precipitation (mm w.e.)', secondary_y=False)
    else:
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='%s hourly precipitation (mm w.e.)' % hours, secondary_y=False)
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='pressure (m w.e.)', secondary_y=True)
    
    # update figure layout
    fig.update_layout(
        title={'text': '%s' % (name), 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis2={'anchor': 'x', 'overlaying': 'y', 'side': 'left'},
        yaxis={'anchor': 'x', 'domain': [0.0, 1.0], 'side':'right'},
        xaxis_title='',
        legend=dict(yanchor="top", y=1., xanchor="left", x=0, orientation="h", traceorder='reversed'),
        legend_title='',
        font=dict(size=18),   
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x',
        margin=dict(l=20, r=20, t=40, b=20))
    
    if show:
        fig.show()
    if saveHTML == True:
        fig.write_html('%s_processed.html' % (name))
    elif type(saveHTML) == str:
        fig.write_html('%s/%s_processed.html' % (saveHTML, name))

    return fig


def plot_processed_hourly_JSON(json_object, name='', start=None, end=None, show=True, saveHTML=False):
    """
    Plot the processed data showing start and end of snowfall, total accumulation and 
    hourly rates, together with before, corrected before, and after gradients used 
    in the processing starting from a dictionary containing the data read from the 
    exported WebApp JSON file

    :param json_object: WebApp exported JSON object
    :type: JSON object
    :start: datetime at which starting the plot
    :type: datetime date 
    :end: time at which stopping the plot
    :type: datetime date

    :returns: a plotly figure object
    :rtype: plotly figure object
    """

    # ------------- get JSON DATA ------------- #
    # input data
    data = pd.read_csv(StringIO(json_object['input']), sep=",", header=10)
    data = data.astype({"datetime": "datetime64[ns]"})
    data = data.set_index('datetime')
    # detection stage
    detection = pd.DataFrame(json_object["detection"]["data"]["data"])
    detection = detection.astype({"datetime": "datetime64[ns]", "window": "datetime64[ns]"})
    detection = detection.set_index("datetime")
    # summary data 
    summary = pd.DataFrame(json_object["analysis"]["summary"]["data"]["data"])
    summary = summary.astype({"start": "datetime64[ns]", "end": "datetime64[ns]"})   
    
    # get colors
    colors = plot_colors()

    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
   
    # ------------- PLOT VARIABLES -------------#
    # plot pressure used in analysis (hourly and smoothed)
    fig.add_trace(go.Scatter(x=detection.index, y=detection['pressure (m[H2O])'], name='pressure', showlegend=False, 
                                mode='lines', line = dict(color='black', width=1, dash='solid'), 
                                hovertemplate='P: %{y:.3f} m w.e.<extra></extra>'), secondary_y=True)
    # plot raw input pressure (not smoothed, may be < hourly)
    fig.add_trace(go.Scatter(x=data.index, y=data['pressure (bar)']*10.1974, name='pressure', showlegend=False, 
                                mode='lines', line = dict(color='rgba(0,0,0,0.3)', width=1, dash='solid'), 
                                hoverinfo='skip'), secondary_y=True)
    
    # plot start and end of snowfall events
    start = detection[detection['event'] == 'start']
    fig.add_trace(go.Scatter(x=start.index, y=start['pressure (m[H2O])'], name='start', showlegend=True, 
                                    mode='markers', marker_symbol='circle', marker_color=colors['red'], marker_size=7, 
                                    hoverinfo='skip'), secondary_y=True)
    end = detection[detection['event'] == 'end']
    fig.add_trace(go.Scatter(x=end.index, y=end['pressure (m[H2O])'], name='end', showlegend=True, 
                                    mode='markers', marker_symbol='circle', marker_color=colors['green'], marker_size=7, 
                                    hoverinfo='skip'), secondary_y=True)
    
    # display event total snowfall accumulation in hoverinfo
    snow = summary['accumulation (mm[H2O])']
    unc = summary['accumulation_uncertainty (mm[H2O])']
    customsnow = summary['end'].copy()
    customsnow[snow.index] = snow.values
    customunc = summary['end'].copy()
    customunc[unc.index] = unc.values
    fig.add_trace(go.Scatter(x=end.index, y=end['pressure (m[H2O])'], 
                                customdata=np.stack((customsnow, customunc), axis=-1), 
                                showlegend=False, mode='markers', marker_symbol='circle', marker_color=colors['green'], marker_size=1, 
                                hovertemplate = '<b>Total snowfall</b>: %{customdata[0]:.2f} &plusmn; %{customdata[1]:.2f} mm w.e.<extra></extra>'), 
                                secondary_y=True)
    
    # plot trends, uncertainty spread, and event total snowfall as vertical line
    for ee in range(0, len(summary)):
        # before
        b_trend = pd.DataFrame(json_object["analysis"]["beforeTrends"][ee]["data"]["data"])
        b_trend = b_trend.astype({"datetime": "datetime64[ns]"})
        fig.add_trace(go.Scatter(x=b_trend['datetime'], y=b_trend['fit (m[H2O])'], name='', showlegend=False, 
                      mode='lines', line = dict(color=colors['orange'], width=1, dash='dot'), hoverinfo='skip'), secondary_y=True) 
        fig.add_trace(go.Scatter(x=b_trend['datetime'], y=b_trend['err_upper (m[H2O])'], showlegend=False,
                                fill=None, mode='lines', line=dict(color=colors['orange_a'], width=0.1,), hoverinfo='skip'), secondary_y=True)
        fig.add_trace(go.Scatter(x=b_trend['datetime'], y=b_trend['err_lower (m[H2O])'], showlegend=False,  
                                fill='tonexty', fillcolor=colors['orange_a'], 
                                mode='lines', line=dict(color=colors['orange_a'], width=0.1), hoverinfo='skip'), secondary_y=True)
        # corrected before
        bc_trend = pd.DataFrame(json_object["analysis"]["correctedBeforeTrends"][ee]["data"]["data"])
        bc_trend = bc_trend.astype({"datetime": "datetime64[ns]"})
        fig.add_trace(go.Scatter(x=bc_trend['datetime'], y=bc_trend['fit (m[H2O])'], name='', showlegend=False, 
                      mode='lines', line = dict(color=colors['red'], width=1, dash='dash'), hoverinfo='skip'), secondary_y=True) 
        fig.add_trace(go.Scatter(x=bc_trend['datetime'], y=bc_trend['err_upper (m[H2O])'], showlegend=False,
                                fill=None, mode='lines', line=dict(color=colors['red_a'], width=0.1,), hoverinfo='skip'), secondary_y=True)
        fig.add_trace(go.Scatter(x=bc_trend['datetime'], y=bc_trend['err_lower (m[H2O])'], showlegend=False,  
                                fill='tonexty', fillcolor=colors['red_a'], 
                                mode='lines', line=dict(color=colors['red_a'], width=0.1), hoverinfo='skip'), secondary_y=True)
        # after
        a_trend = pd.DataFrame(json_object["analysis"]["afterTrends"][ee]["data"]["data"])
        a_trend = a_trend.astype({"datetime": "datetime64[ns]"})
        fig.add_trace(go.Scatter(x=a_trend['datetime'], y=a_trend['fit (m[H2O])'], name='', showlegend=False, 
                      mode='lines', line = dict(color=colors['green'], width=1, dash='dash'), hoverinfo='skip'), secondary_y=True)  
        fig.add_trace(go.Scatter(x=a_trend['datetime'], y=a_trend['err_upper (m[H2O])'], showlegend=False,
                                fill=None, mode='lines', line=dict(color=colors['green_a'], width=0.1,), hoverinfo='skip'), secondary_y=True)
        fig.add_trace(go.Scatter(x=a_trend['datetime'], y=a_trend['err_lower (m[H2O])'], showlegend=False,  
                                fill='tonexty', fillcolor=colors['green_a'], 
                                mode='lines', line=dict(color=colors['green_a'], width=0.1), hoverinfo='skip'), secondary_y=True)
        # event total snowfall as vertical line
        xx = summary.loc[ee, 'end']
        ymin = bc_trend.loc[bc_trend['datetime'] == xx, 'fit (m[H2O])'].values[0]
        ymax = a_trend.loc[a_trend['datetime'] == xx, 'fit (m[H2O])'].values[0]
        fig.add_trace(go.Scatter(x=[xx, xx], y=[ymin, ymax], showlegend=False, mode='lines', line=dict(color=colors['red'], 
                                    width=1, dash='solid'), hoverinfo='skip'), secondary_y=True)


    # ------------- FIGURE STYLE ------------- #
    # update axes
    fig.update_xaxes(hoverformat = "%d %b %y <br> %H:%M")
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='pressure (m w.e.)', secondary_y=True)
        
    # update figure layout
    fig.update_layout(
        title={'text': '%s' % (name), 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        yaxis2={'anchor': 'x', 'overlaying': 'y', 'side': 'left'},
        yaxis={'anchor': 'x', 'domain': [0.0, 1.0], 'side':'right'},
        xaxis_title='',
        legend=dict(yanchor="top", y=1., xanchor="left", x=0, orientation="h", traceorder='reversed'),
        legend_title='',
        font=dict(size=18),   
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x',
        margin=dict(l=20, r=20, t=40, b=20))

    if show:
        fig.show()
    if saveHTML == True:
        fig.write_html('%s_processed_JSON.html' % (name))
    elif type(saveHTML) == str:
        fig.write_html('%s/%s_processed_JSON.html' % (saveHTML, name))

    return fig


def plot_processed_hourly_temp(p, raw_data=None, name='', start=None, end=None, hours=1, show=True, saveHTML=False):
    """
    Plot the processed data showing start and end of snowfall, total accumulation and 
    hourly rates, together with before, corrected before, and after gradients used 
    in the processing

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :start: datetime at which starting the plot
    :type: datetime date 
    :end: time at which stopping the plot
    :type: datetime date

    :returns: a plotly figure object
    :rtype: plotly figure object
    """
    
    # get data from snowfall processor
    data = p.input
    extrema = p.breakpoints['extrema']
    maxima = extrema.dropna(subset=['maxima']) 
    trends = p.breakpoints['trends']
    differences = p.breakpoints['differences']
    hourly = p.breakpoints['differences_hourly']
    if raw_data is not None:
        Twater = raw_data['water_temperature']        
        Tground = raw_data['ground_temperature']
    else:
        Twater = data['water_temperature']        
        Tground = data['ground_temperature']


    # subset data
    if start:
        data = data[start:]
    if end:
        data = data[:end]

    # get colors
    colors = plot_colors()

    # define config lists with legends and colors
    extrema_conf = [{'name': 'start', 'l': 'minima', 'c': colors['red']}, 
                    {'name': 'end', 'l': 'maxima', 'c': colors['green']}]
    trends_conf = [{'l': 'before', 'c': colors['orange'], 'ls': 'dot', 'label': 'before'}, 
                    {'l': 'p_corrected_before', 'c': colors['red'], 'ls': 'dash', 'label': 'corrected before'}, 
                    {'l': 'after', 'c': colors['green'], 'ls': 'dash', 'label': 'after'}]
    trends_conf_fill = [{'l': 'before', 'c': colors['orange_a'], 'ls': 'dot'},
                        {'l': 'p_corrected_before', 'c': colors['red_a'], 'ls': 'dash'}, 
                        {'l': 'after', 'c': colors['green_a'], 'ls': 'dash'}]
    
    fig = make_subplots(rows=2, vertical_spacing = 0.05, shared_xaxes=True, 
                        row_heights=[0.6, 0.4], specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    # plot hourly rates in the background
    for i in range(0, len(hourly)):
        bars = hourly[i].resample('%sH' % str(hours), label='right', closed='right').sum(numeric_only=True)
        bars[bars < 0] = 0  # reset negative precipitation
        if i == 0:
            fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                    width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                    name='hourly rates', marker_color=colors['grey_a'], showlegend=True, 
                                    hovertemplate='hourly rate: %{y:.1f} mm w.e.<extra></extra>'), secondary_y=False, row=1, col=1)
        else:
            fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                    width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                    name='hourly rates', marker_color=colors['grey_a'], showlegend=False, 
                                    hovertemplate='hourly rate: %{y:.1f} mm w.e.<extra></extra>'), secondary_y=False, row=1, col=1)
  
    # plot pressure 
    fig.add_trace(go.Scatter(x=data.index, y=data['pressure'], name='pressure', showlegend=False, 
                                mode='lines', line = dict(color='black', width=1, dash='solid'), 
                                hovertemplate='P: %{y:.3f} m w.e.<extra></extra>'), secondary_y=True, row=1, col=1)
    try:
        p.input_raw
        fig.add_trace(go.Scatter(x=p.input_raw.index, y=p.input_raw['pressure'], name='pressure', showlegend=False, 
                        mode='lines', line = dict(color='rgba(0,0,0,0.3)', width=1, dash='solid'), 
                        hoverinfo='skip'), secondary_y=True, row=1, col=1)
    except AttributeError:
        pass
        
    # plot start and end of snowfall events
    for conf in extrema_conf:
        fig.add_trace(go.Scatter(x=extrema.index, y=extrema[conf['l']], name=conf['name'], showlegend=True, 
                                    mode='markers', marker_symbol='circle', marker_color=conf['c'], marker_size=7, 
                                    hoverinfo='skip'), secondary_y=True, row=1, col=1)
            
    # display total snowfall accumulation in hoverinfo
    snow = p.breakpoints['status'][p.breakpoints['status']['type'] == 'end']['corrected_delta']
    unc = p.breakpoints['status'][p.breakpoints['status']['type'] == 'end']['uncertainty']
    customsnow = p.breakpoints['extrema']['maxima'].copy()
    customsnow[snow.index] = snow.values*1000.
    customunc = p.breakpoints['extrema']['maxima'].copy()
    customunc[unc.index] = unc.values*1000.
    fig.add_trace(go.Scatter(x=p.breakpoints['extrema']['maxima'].index, y=p.breakpoints['extrema']['maxima'], 
                                customdata=np.stack((customsnow, customunc), axis=-1), 
                                showlegend=False, mode='markers', marker_symbol='circle', marker_color=colors['green'], marker_size=1, 
                                hovertemplate = '<b>Total snowfall</b>: %{customdata[0]:.2f} &plusmn; %{customdata[1]:.2f} mm w.e.<extra></extra>'), 
                                secondary_y=True, row=1, col=1)
    
    # plot before/corrected_before/after trends and fill in the uncertainty spread
    for conf in trends_conf:
        for i, regression in enumerate(trends[conf['l']]):
            if i == 0:
                fig.add_trace(go.Scatter(x=regression.index, y=regression['fit'], name=conf['label'], showlegend=True, 
                                         mode='lines', line = dict(color=conf['c'], width=1, dash=conf['ls']), hoverinfo='skip'), secondary_y=True, row=1, col=1)
            else:   
                fig.add_trace(go.Scatter(x=regression.index, y=regression['fit'], name=conf['label'], showlegend=False, 
                                         mode='lines', line = dict(color=conf['c'], width=1, dash=conf['ls']), hoverinfo='skip'), secondary_y=True, row=1, col=1)        
    for conf in trends_conf_fill:
        for i, regression in enumerate(trends[conf['l']]):     
            fig.add_trace(go.Scatter(x=regression.index, y=regression['err_upper'], showlegend=False,
                                     fill=None, mode='lines', line=dict(color=conf['c'], width=0.1,), hoverinfo='skip'), secondary_y=True, row=1, col=1)
            fig.add_trace(go.Scatter(x=regression.index, y=regression['err_lower'], showlegend=False,  fill='tonexty', fillcolor=conf['c'], 
                                     mode='lines', line=dict(color=conf['c'], width=0.1), hoverinfo='skip'), secondary_y=True, row=1, col=1)

    # plot the differences as vertical lines      
    x = pd.to_datetime(differences['t2'].values)
    conf = next(i for i in trends_conf if i['l'] == 'p_corrected_before')    
    for xx in range(0, len(x)):
        ymin = trends['p_corrected_before'][xx].fit[-1]
        delta = differences['corrected_delta'][xx]
        fig.add_trace(go.Scatter(x=[x[xx], x[xx]], y=[ymin, ymin+delta], showlegend=False, 
                                    mode='lines', line=dict(color=conf['c'], width=1, dash='solid'), 
                                    hoverinfo='skip'), secondary_y=True, row=1, col=1)
        
    # plot 0C temperature line
    x = [data.index[0], data.index[-1]]
    y = [0, 0]
    fig.add_trace(go.Scatter(x=x, y=y, showlegend=False, mode='lines', line = dict(color='black', width=1, dash='dash'), 
                                    hoverinfo='none'), secondary_y=False, row=2, col=1) 
    # plot water temperature 
    fig.add_trace(go.Scatter(x=Twater.index, y=Twater, name='T water', showlegend=True, 
                                mode='lines', line = dict(color='blue', width=1, dash='solid'), 
                                hovertemplate='T water: %{y:.2f} (\u00B0C)'), secondary_y=False, row=2, col=1)
    # plot ground temperature 
    fig.add_trace(go.Scatter(x=Tground.index, y=Tground, name='T ground', showlegend=True, 
                                mode='lines', line = dict(color='red', width=1, dash='solid'), 
                                hovertemplate='%{y:.2f} (\u00B0C)'), secondary_y=False, row=2, col=1)
    
    # plot flagged events
    small = p.breakpoints['events_windows'][p.breakpoints['events_windows'].flag == 'small and short event']
    for i in range(0, len(small)):
        if i == 0:
            fig.add_vrect(x0=small.iloc[i].start, x1=small.iloc[i].end, showlegend=True, name='flagged events', line_width=0, 
                          fillcolor="orange", opacity=0.75, layer='below', row=1, col=1)
        else:
            fig.add_vrect(x0=small.iloc[i].start, x1=small.iloc[i].end, line_width=0, 
                          fillcolor="orange", opacity=0.75, layer='below', row=1, col=1)
    large = p.breakpoints['events_windows'][p.breakpoints['events_windows'].flag == 'probable calving or avalanche']
    for i in range(0, len(large)):
        if i == 0:
            fig.add_vrect(x0=large.iloc[i].start, x1=large.iloc[i].end, showlegend=True, name='calving/avalanche', line_width=0, 
                          fillcolor="purple", opacity=0.75, layer='below', row=1, col=1)
        else:
            fig.add_vrect(x0=large.iloc[i].start, x1=large.iloc[i].end, line_width=0, 
                          fillcolor="purple", opacity=0.75, layer='below', row=1, col=1)
    

    # update axes
    fig.update_xaxes(hoverformat = "%d %b %y <br> %H:%M")
    if hours == 1:
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='hourly rates (mm w.e.)', secondary_y=False, row=1, col=1)
    else:
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='%s hours precipitation (mm w.e.)' % hours, secondary_y=False, row=1, col=1)
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='pressure (m w.e.)', secondary_y=True, row=1, col=1)
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='temperature (\u00B0C)', secondary_y=False, row=2, col=1)
    # fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='T water (\u00B0C)', secondary_y=False, row=2, col=1)
    # fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='T ground (\u00B0C)', secondary_y=True, row=2, col=1)
    
    # update figure layout
    fig.update_layout(
        title={'text': '%s' % (name), 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        # yaxis2={'anchor': 'x', 'overlaying': 'y', 'side': 'left'},
        # yaxis={'anchor': 'x', 'domain': [0.0, 1.0], 'side':'right'},
        xaxis_title='',
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0, orientation="h", traceorder='reversed'),
        legend_title='',
        font=dict(size=18),   
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x',
        # height=600,
        # width=1000,
        margin=dict(l=20, r=20, t=40, b=20)
        )
    
    if show:
        fig.show()
    if saveHTML == True:
        fig.write_html('%s_processed_temp.html' % (name))
    elif type(saveHTML) == str:
        fig.write_html('%s/%s_processed_temp.html' % (saveHTML, name))

    return fig


def plot_processed_hourly_temp_ground(p, raw_data=None, name='', start=None, end=None, hours=1, show=True, saveHTML=False):
    """
    Plot the processed data showing start and end of snowfall, total accumulation and 
    hourly rates, together with before, corrected before, and after gradients used 
    in the processing

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :start: datetime at which starting the plot
    :type: datetime date 
    :end: time at which stopping the plot
    :type: datetime date

    :returns: a plotly figure object
    :rtype: plotly figure object
    """
    
    # get data from snowfall processor
    data = p.input
    extrema = p.breakpoints['extrema']
    maxima = extrema.dropna(subset=['maxima']) 
    trends = p.breakpoints['trends']
    differences = p.breakpoints['differences']
    hourly = p.breakpoints['differences_hourly']  
    if 'ground_temperature' in data.keys():
        Tground = data['ground_temperature']
    if 'water_temperature' in data.keys():
        Twater = data['water_temperature']


    # subset data
    if start:
        data = data[start:]
    if end:
        data = data[:end]

    # get colors
    colors = plot_colors()

    # define config lists with legends and colors
    extrema_conf = [{'name': 'start', 'l': 'minima', 'c': colors['red']}, 
                    {'name': 'end', 'l': 'maxima', 'c': colors['green']}]
    trends_conf = [{'l': 'before', 'c': colors['orange'], 'ls': 'dot', 'label': 'before'}, 
                    {'l': 'p_corrected_before', 'c': colors['red'], 'ls': 'dash', 'label': 'corrected before'}, 
                    {'l': 'after', 'c': colors['green'], 'ls': 'dash', 'label': 'after'}]
    trends_conf_fill = [{'l': 'before', 'c': colors['orange_a'], 'ls': 'dot'},
                        {'l': 'p_corrected_before', 'c': colors['red_a'], 'ls': 'dash'}, 
                        {'l': 'after', 'c': colors['green_a'], 'ls': 'dash'}]
    
    fig = make_subplots(rows=2, vertical_spacing = 0.05, shared_xaxes=True, 
                        row_heights=[0.6, 0.4], specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    # plot hourly rates in the background
    for i in range(0, len(hourly)):
        bars = hourly[i].resample('%sH' % str(hours), label='right', closed='right').sum(numeric_only=True)
        bars[bars < 0] = 0  # reset negative precipitation
        if i == 0:
            fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                    width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                    name='hourly rates', marker_color=colors['grey_a'], showlegend=True, 
                                    hovertemplate='hourly rate: %{y:.1f} mm w.e.<extra></extra>'), secondary_y=False, row=1, col=1)
        else:
            fig.add_trace(go.Bar(x=bars.index-datetime.timedelta(hours=hours/2), y=bars['corrected_delta_hourly_rates']*1000, 
                                    width=datetime.timedelta(hours=hours).total_seconds()*1000, 
                                    name='hourly rates', marker_color=colors['grey_a'], showlegend=False, 
                                    hovertemplate='hourly rate: %{y:.1f} mm w.e.<extra></extra>'), secondary_y=False, row=1, col=1)
  
    # plot pressure 
    fig.add_trace(go.Scatter(x=data.index, y=data['pressure'], name='pressure', showlegend=False, 
                                mode='lines', line = dict(color='black', width=1, dash='solid'), 
                                hovertemplate='P: %{y:.3f} m w.e.<extra></extra>'), secondary_y=True, row=1, col=1)
    try:
        p.input_raw
        fig.add_trace(go.Scatter(x=p.input_raw.index, y=p.input_raw['pressure'], name='pressure', showlegend=False, 
                        mode='lines', line = dict(color='rgba(0,0,0,0.3)', width=1, dash='solid'), 
                        hoverinfo='skip'), secondary_y=True, row=1, col=1)
    except AttributeError:
        pass
        
    # plot start and end of snowfall events
    for conf in extrema_conf:
        fig.add_trace(go.Scatter(x=extrema.index, y=extrema[conf['l']], name=conf['name'], showlegend=True, 
                                    mode='markers', marker_symbol='circle', marker_color=conf['c'], marker_size=7, 
                                    hoverinfo='skip'), secondary_y=True, row=1, col=1)
            
    # display total snowfall accumulation in hoverinfo
    snow = p.breakpoints['status'][p.breakpoints['status']['type'] == 'end']['corrected_delta']
    unc = p.breakpoints['status'][p.breakpoints['status']['type'] == 'end']['uncertainty']
    customsnow = p.breakpoints['extrema']['maxima'].copy()
    customsnow[snow.index] = snow.values*1000.
    customunc = p.breakpoints['extrema']['maxima'].copy()
    customunc[unc.index] = unc.values*1000.
    fig.add_trace(go.Scatter(x=p.breakpoints['extrema']['maxima'].index, y=p.breakpoints['extrema']['maxima'], 
                                customdata=np.stack((customsnow, customunc), axis=-1), 
                                showlegend=False, mode='markers', marker_symbol='circle', marker_color=colors['green'], marker_size=1, 
                                hovertemplate = '<b>Total snowfall</b>: %{customdata[0]:.2f} &plusmn; %{customdata[1]:.2f} mm w.e.<extra></extra>'), 
                                secondary_y=True, row=1, col=1)
    
    # plot before/corrected_before/after trends and fill in the uncertainty spread
    for conf in trends_conf:
        for i, regression in enumerate(trends[conf['l']]):
            if i == 0:
                fig.add_trace(go.Scatter(x=regression.index, y=regression['fit'], name=conf['label'], showlegend=True, 
                                         mode='lines', line = dict(color=conf['c'], width=1, dash=conf['ls']), hoverinfo='skip'), secondary_y=True, row=1, col=1)
            else:   
                fig.add_trace(go.Scatter(x=regression.index, y=regression['fit'], name=conf['label'], showlegend=False, 
                                         mode='lines', line = dict(color=conf['c'], width=1, dash=conf['ls']), hoverinfo='skip'), secondary_y=True, row=1, col=1)        
    for conf in trends_conf_fill:
        for i, regression in enumerate(trends[conf['l']]):     
            fig.add_trace(go.Scatter(x=regression.index, y=regression['err_upper'], showlegend=False,
                                     fill=None, mode='lines', line=dict(color=conf['c'], width=0.1,), hoverinfo='skip'), secondary_y=True, row=1, col=1)
            fig.add_trace(go.Scatter(x=regression.index, y=regression['err_lower'], showlegend=False,  fill='tonexty', fillcolor=conf['c'], 
                                     mode='lines', line=dict(color=conf['c'], width=0.1), hoverinfo='skip'), secondary_y=True, row=1, col=1)

    # plot the differences as vertical lines      
    x = pd.to_datetime(differences['t2'].values)
    conf = next(i for i in trends_conf if i['l'] == 'p_corrected_before')    
    for xx in range(0, len(x)):
        ymin = trends['p_corrected_before'][xx].fit[-1]
        delta = differences['corrected_delta'][xx]
        fig.add_trace(go.Scatter(x=[x[xx], x[xx]], y=[ymin, ymin+delta], showlegend=False, 
                                    mode='lines', line=dict(color=conf['c'], width=1, dash='solid'), 
                                    hoverinfo='skip'), secondary_y=True, row=1, col=1)
        
    # plot 0C temperature line
    x = [data.index[0], data.index[-1]]
    y = [0, 0]
    fig.add_trace(go.Scatter(x=x, y=y, showlegend=False, mode='lines', line = dict(color='black', width=1, dash='dash'), 
                                    hoverinfo='none'), secondary_y=False, row=2, col=1) 
    # plot ground temperature 
    fig.add_trace(go.Scatter(x=Tground.index, y=Tground, name='T ground', showlegend=True, 
                                mode='lines', line = dict(color='red', width=1, dash='solid'), 
                                hovertemplate='T ground: %{y:.2f} (\u00B0C)'), secondary_y=False, row=2, col=1)
    # plot water temperature 
    fig.add_trace(go.Scatter(x=Twater.index, y=Twater, name='T water', showlegend=True, 
                                mode='lines', line = dict(color='blue', width=1, dash='solid'), 
                                hovertemplate='T ground: %{y:.2f} (\u00B0C)'), secondary_y=False, row=2, col=1)
    
    # plot flagged events
    if 'flagged_events' in p.breakpoints.keys():
        flagged = p.breakpoints['flagged_events']
        for i in range(0, len(flagged)):
            if i == 0:
                fig.add_vrect(x0=flagged.loc[i].flagged_start, x1=flagged.loc[i].flagged_end, showlegend=True, name='flagged events', line_width=0, 
                              fillcolor="purple", opacity=0.75, layer='below', row=1, col=1)
                fig.add_vrect(x0=flagged.loc[i].flagged_start, x1=flagged.loc[i].flagged_end, showlegend=False, name='flagged events', line_width=0, 
                              fillcolor="purple", opacity=0.75, layer='below', row=2, col=1)
            else:
                fig.add_vrect(x0=flagged.loc[i].flagged_start, x1=flagged.loc[i].flagged_end, line_width=0, 
                              fillcolor="purple", opacity=0.75, layer='below', row=1, col=1)
                fig.add_vrect(x0=flagged.loc[i].flagged_start, x1=flagged.loc[i].flagged_end, line_width=0, 
                              fillcolor="purple", opacity=0.75, layer='below', row=2, col=1)
    # plot excluded events
    if 'excluded_events' in p.breakpoints.keys():
        excluded = p.breakpoints['excluded_events']
        for i in range(0, len(excluded)):
            if i == 0:
                fig.add_vrect(x0=excluded.loc[i].excluded_start, x1=excluded.loc[i].excluded_end, showlegend=True, name='excluded events', line_width=0, 
                              fillcolor="orange", opacity=0.75, layer='below', row=1, col=1)
                fig.add_vrect(x0=excluded.loc[i].excluded_start, x1=excluded.loc[i].excluded_end, showlegend=False, name='excluded events', line_width=0, 
                              fillcolor="orange", opacity=0.75, layer='below', row=2, col=1)
            else:
                fig.add_vrect(x0=excluded.loc[i].excluded_start, x1=excluded.loc[i].excluded_end, line_width=0, 
                              fillcolor="orange", opacity=0.75, layer='below', row=1, col=1)
                fig.add_vrect(x0=excluded.loc[i].excluded_start, x1=excluded.loc[i].excluded_end, line_width=0, 
                              fillcolor="orange", opacity=0.75, layer='below', row=2, col=1)
    

    # update axes
    fig.update_xaxes(hoverformat = "%d %b %y <br> %H:%M")
    if hours == 1:
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='hourly rates (mm w.e.)', secondary_y=False, row=1, col=1)
    else:
        fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='%s hours precipitation (mm w.e.)' % hours, secondary_y=False, row=1, col=1)
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='pressure (m w.e.)', secondary_y=True, row=1, col=1)
    fig.update_yaxes(dict(ticks='inside', showgrid=True), title_text='temperature (\u00B0C)', secondary_y=False, row=2, col=1)
    # fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='T water (\u00B0C)', secondary_y=False, row=2, col=1)
    # fig.update_yaxes(dict(ticks='inside', showgrid=False), title_text='T ground (\u00B0C)', secondary_y=True, row=2, col=1)
    
    # update figure layout
    fig.update_layout(
        title={'text': '%s' % (name), 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        # yaxis2={'anchor': 'x', 'overlaying': 'y', 'side': 'left'},
        # yaxis={'anchor': 'x', 'domain': [0.0, 1.0], 'side':'right'},
        xaxis_title='',
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0, orientation="h", traceorder='reversed'),
        legend_title='',
        font=dict(size=18),   
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x',
        # height=600,
        # width=1000,
        margin=dict(l=20, r=20, t=40, b=20)
        )
    
    if show:
        fig.show()
    if saveHTML == True:
        fig.write_html('%s_processed_temp.html' % (name))
    elif type(saveHTML) == str:
        fig.write_html('%s/%s_processed_temp.html' % (saveHTML, name))

    return fig


def format_snowfall_output_summary(p):
    """
    Create and format a summary DataFrame with snowfall events start and end dates 
    and total accumulation and uncertainty

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :returns: a new DataFrame with the formatted output
    :rtype: DataFrame
    """

    tmp = p.breakpoints['status']
    df = tmp[tmp['type'] == 'start'].copy()
    df['start'] = df.index
    df['end'] = tmp[tmp['type'] == 'end'].index
    df['duration'] = df['end'] - df['start']
    df['middle'] = df['start'] + df['duration']/2
    df['corrected_delta'] *= 1000.
    df['uncertainty'] *= 1000.
    
    df = df[['start','end','corrected_delta','uncertainty','flag']]
    df = df.rename(columns={'corrected_delta':'Snowfall accumulation (mm w.e.)','uncertainty':'Uncertainty (mm w.e.)','flag':'Event flag'})
    df = df.reset_index(drop=True)
    df['Snowfall accumulation (mm w.e.)'] = df['Snowfall accumulation (mm w.e.)'].map('{:,.2f}'.format)
    df['Uncertainty (mm w.e.)'] = df['Uncertainty (mm w.e.)'].map('{:,.2f}'.format)
    
    return df


def format_snowfall_output_hourly(p):
    """
    Create and format a DataFrame with hourly snowfall accumulation and uncertainty 
    covering the whole analysis period

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :returns: a new DataFrame with the formatted output
    :rtype: DataFrame
    """

    empty_array =  np.zeros_like(p.input.pressure).astype(str)
    empty_array[:] = ''
    cols = {'Snowfall accumulation hourly rate (mm w.e.)': np.zeros_like(p.input.pressure), 
            'Uncertainty hourly (mm w.e.)': np.zeros_like(p.input.pressure),
            'Event flag': empty_array}
    df = pd.DataFrame(index=p.input.index, data=cols)

    for i in range(0, len(p.breakpoints['differences_hourly'])):
        tmp = p.breakpoints['differences_hourly'][i].copy()
        tmp['corrected_delta_hourly_rates'] *= 1000.
        tmp['uncertainty_hourly'] *= 1000.
        tmp = tmp[['corrected_delta_hourly_rates', 'uncertainty_hourly', 'flag']]
        tmp = tmp.rename(columns={'corrected_delta_hourly_rates':'Snowfall accumulation hourly rate (mm w.e.)','uncertainty_hourly':'Uncertainty hourly (mm w.e.)','flag':'Event flag'})

        df.loc[tmp.index, 'Snowfall accumulation hourly rate (mm w.e.)'] = tmp['Snowfall accumulation hourly rate (mm w.e.)'].map('{:,.2f}'.format).astype(float)
        df.loc[tmp.index, 'Uncertainty hourly (mm w.e.)'] = tmp['Uncertainty hourly (mm w.e.)'].map('{:,.2f}'.format).astype(float)
        df.loc[tmp.index, 'Event flag'] = tmp['Event flag'].astype(str)
    
    for i in range(0, len(df)):
            if df.loc[df.index[i], 'Snowfall accumulation hourly rate (mm w.e.)'] == 0.:
                df.loc[df.index[i], 'Event flag'] = ''
        

    return df


def unc_propagation(hourly_uncertainty):
    """
    Function to propagate the error from hourly to 3 hourly

    :param unc: an array containting the uncertainties at each hour that needs to be propagated
    :type: numpy array with len=3
    :returns: the propagate uncertainty
    :rtype: float
    """

    propagated_uncertainty = 0.0
    for i in range(0, len(hourly_uncertainty)):
        propagated_uncertainty += hourly_uncertainty[i]**2

    return np.sqrt(propagated_uncertainty)


def format_snowfall_output_3hourly(p):
    """
    Create and format a DataFrame with 3 hourly snowfall accumulation and uncertainty 
    covering the whole analysis period

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :returns: a new DataFrame with the formatted output
    :rtype: DataFrame
    """

    # create empty DataFrame at 3 hours frequency
    clone = p.input.resample('3H', label='right', closed='right').sum(numeric_only=True)
    clone = clone[p.input.index.date[1]:p.input.index.date[-2]]
    cols = {'Snowfall accumulation 3 hourly rate (mm w.e.)': np.zeros_like(clone.pressure), 
            'Uncertainty 3 hourly (mm w.e.)': np.zeros_like(clone.pressure),
            'Event flag': np.asarray(["" for x in range(len(clone.pressure))])}
    df = pd.DataFrame(index=clone.index, data=cols)

    hourly = format_snowfall_output_hourly(p)
    hourly = hourly[p.input.index.date[1]:p.input.index.date[-2]]
    # compute the 3 hourly sum 
    df['Snowfall accumulation 3 hourly rate (mm w.e.)'] = pd.to_numeric(hourly['Snowfall accumulation hourly rate (mm w.e.)']).resample('3H', label='right', closed='right').sum().astype(float)

    days = df.resample('1D').mean(numeric_only=True).index
    days = days.drop(days[-1])

    for i in range(0, len(days)):

        unc = hourly[days[i] + datetime.timedelta(hours=1):days[i] + datetime.timedelta(hours=24)]['Uncertainty hourly (mm w.e.)'].astype(float)
        
        unc_propagation(unc[unc.index.hour.isin([1, 2, 3])])
        df.loc[days[i]+datetime.timedelta(hours=3), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([1, 2, 3])])

        unc_propagation(unc[unc.index.hour.isin([4, 5, 6])])
        df.loc[days[i]+datetime.timedelta(hours=6), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([4, 5, 6])])

        unc_propagation(unc[unc.index.hour.isin([7, 8, 9])])
        df.loc[days[i]+datetime.timedelta(hours=9), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([7, 8, 9])])

        unc_propagation(unc[unc.index.hour.isin([10, 11, 12])])
        df.loc[days[i]+datetime.timedelta(hours=12), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([10, 11, 12])])

        unc_propagation(unc[unc.index.hour.isin([13, 14, 15])])
        df.loc[days[i]+datetime.timedelta(hours=15), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([13, 14, 15])])

        unc_propagation(unc[unc.index.hour.isin([16, 17, 18])])
        df.loc[days[i]+datetime.timedelta(hours=18), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([16, 17, 18])])

        unc_propagation(unc[unc.index.hour.isin([19, 20, 21])])
        df.loc[days[i]+datetime.timedelta(hours=21), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([19, 20, 21])])

        unc_propagation(unc[unc.index.hour.isin([22, 23, 0])])
        df.loc[days[i]+datetime.timedelta(hours=24), 'Uncertainty 3 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([22, 23, 0])])

    df['Snowfall accumulation 3 hourly rate (mm w.e.)'] = df['Snowfall accumulation 3 hourly rate (mm w.e.)'].map('{:,.2f}'.format)
    df['Snowfall accumulation 3 hourly rate (mm w.e.)'] = df['Snowfall accumulation 3 hourly rate (mm w.e.)'].astype(float)
    df['Uncertainty 3 hourly (mm w.e.)'] = df['Uncertainty 3 hourly (mm w.e.)'].map('{:,.2f}'.format)
    df['Uncertainty 3 hourly (mm w.e.)'] = df['Uncertainty 3 hourly (mm w.e.)'].astype(float)

    for i in range(1, len(df['Snowfall accumulation 3 hourly rate (mm w.e.)'])):
        if df.loc[df.index[i], 'Snowfall accumulation 3 hourly rate (mm w.e.)'] > 0.:
            if hourly['Event flag'].loc[df.index[i]] != '':
                df.loc[df.index[i], 'Event flag'] = hourly['Event flag'].loc[df.index[i]]
            elif hourly['Event flag'].loc[df.index[i]-datetime.timedelta(hours=1)] != '':
                df.loc[df.index[i], 'Event flag'] = hourly['Event flag'].loc[df.index[i]-datetime.timedelta(hours=1)]
            elif hourly['Event flag'].loc[df.index[i]-datetime.timedelta(hours=2)] != '':
                df.loc[df.index[i], 'Event flag'] = hourly['Event flag'].loc[df.index[i]-datetime.timedelta(hours=2)]
    df['Event flag'] = df['Event flag'].astype(str)

    return df


def format_snowfall_output_6hourly(p):
    """
    Create and format a DataFrame with 6 hourly snowfall accumulation and uncertainty 
    covering the whole analysis period

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :returns: a new DataFrame with the formatted output
    :rtype: DataFrame
    """

    # create empty DataFrame at 6 hours frequency
    clone = p.input.resample('6H', label='right', closed='right').sum(numeric_only=True)
    clone = clone[p.input.index.date[1]:p.input.index.date[-2]]
    cols = {'Snowfall accumulation 6 hourly rate (mm w.e.)': np.zeros_like(clone.pressure), 
            'Uncertainty 6 hourly (mm w.e.)': np.zeros_like(clone.pressure)}
    df = pd.DataFrame(index=clone.index, data=cols)

    hourly = format_snowfall_output_hourly(p)
    hourly = hourly[p.input.index.date[1]:p.input.index.date[-2]]
    # compute the 3 hourly sum 
    df['Snowfall accumulation 6 hourly rate (mm w.e.)'] = pd.to_numeric(hourly['Snowfall accumulation hourly rate (mm w.e.)']).resample('6H', label='right', closed='right').sum().astype(float)

    days = df.resample('1D').mean().index
    days = days.drop(days[-1])

    for i in range(0, len(days)):

        unc = hourly[days[i] + datetime.timedelta(hours=1):days[i] + datetime.timedelta(hours=24)]['Uncertainty hourly (mm w.e.)'].astype(float)
        
        unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6])])
        df.loc[days[i]+datetime.timedelta(hours=6), 'Uncertainty 6 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6])])

        unc_propagation(unc[unc.index.hour.isin([7, 8, 9, 10, 11, 12])])
        df.loc[days[i]+datetime.timedelta(hours=12), 'Uncertainty 6 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([7, 8, 9, 10, 11, 12])])

        unc_propagation(unc[unc.index.hour.isin([13, 14, 15, 16, 17, 18])])
        df.loc[days[i]+datetime.timedelta(hours=18), 'Uncertainty 6 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([13, 14, 15, 16, 17, 18])])

        unc_propagation(unc[unc.index.hour.isin([19, 20, 21, 22, 23, 0])])
        df.loc[days[i]+datetime.timedelta(hours=24), 'Uncertainty 6 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([19, 20, 21, 22, 23, 0])])

    df['Snowfall accumulation 6 hourly rate (mm w.e.)'] = df['Snowfall accumulation 6 hourly rate (mm w.e.)'].map('{:,.2f}'.format)
    df['Uncertainty 6 hourly (mm w.e.)'] = df['Uncertainty 6 hourly (mm w.e.)'].map('{:,.2f}'.format)

    return df


def format_snowfall_output_12hourly(p):
    """
    Create and format a DataFrame with 12 hourly snowfall accumulation and uncertainty 
    covering the whole analysis period

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :returns: a new DataFrame with the formatted output
    :rtype: DataFrame
    """

    # create empty DataFrame at 12 hours frequency
    clone = p.input.resample('12H', label='right', closed='right').sum(numeric_only=True)
    clone = clone[p.input.index.date[1]:p.input.index.date[-2]]
    cols = {'Snowfall accumulation 12 hourly rate (mm w.e.)': np.zeros_like(clone.pressure), 
            'Uncertainty 12 hourly (mm w.e.)': np.zeros_like(clone.pressure)}
    df = pd.DataFrame(index=clone.index, data=cols)

    hourly = format_snowfall_output_hourly(p)
    hourly = hourly[p.input.index.date[1]:p.input.index.date[-2]]
    # compute the 12 hourly sum 
    df['Snowfall accumulation 12 hourly rate (mm w.e.)'] = pd.to_numeric(hourly['Snowfall accumulation hourly rate (mm w.e.)']).resample('12H', label='right', closed='right').sum().astype(float)

    days = df.resample('1D').mean().index
    days = days.drop(days[-1])

    for i in range(0, len(days)):

        unc = hourly[days[i] + datetime.timedelta(hours=1):days[i] + datetime.timedelta(hours=24)]['Uncertainty hourly (mm w.e.)'].astype(float)
        
        unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])])
        df.loc[days[i]+datetime.timedelta(hours=12), 'Uncertainty 12 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])])

        unc_propagation(unc[unc.index.hour.isin([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0])])
        df.loc[days[i]+datetime.timedelta(hours=24), 'Uncertainty 12 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0])])

    df['Snowfall accumulation 12 hourly rate (mm w.e.)'] = df['Snowfall accumulation 12 hourly rate (mm w.e.)'].map('{:,.2f}'.format)
    df['Uncertainty 12 hourly (mm w.e.)'] = df['Uncertainty 12 hourly (mm w.e.)'].map('{:,.2f}'.format)

    return df


def format_snowfall_output_daily(p):
    """
    Create and format a DataFrame with 12 hourly snowfall accumulation and uncertainty 
    covering the whole analysis period

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :returns: a new DataFrame with the formatted output
    :rtype: DataFrame
    """

    # create empty DataFrame at 24 hours frequency
    clone = p.input.resample('24H', label='right', closed='right').sum(numeric_only=True)
    clone = clone[p.input.index.date[1]:p.input.index.date[-2]]
    cols = {'Snowfall accumulation daily rate (mm w.e.)': np.zeros_like(clone.pressure), 
            'Uncertainty daily (mm w.e.)': np.zeros_like(clone.pressure)}
    df = pd.DataFrame(index=clone.index, data=cols)

    hourly = format_snowfall_output_hourly(p)
    hourly = hourly[p.input.index.date[1]:p.input.index.date[-2]]
    # compute the 12 hourly sum 
    df['Snowfall accumulation daily rate (mm w.e.)'] = pd.to_numeric(hourly['Snowfall accumulation hourly rate (mm w.e.)']).resample('24H', label='right', closed='right').sum().astype(float)

    days = df.resample('1D').mean().index
    days = days.drop(days[-1])

    for i in range(0, len(days)):

        unc = hourly[days[i] + datetime.timedelta(hours=1):days[i] + datetime.timedelta(hours=24)]['Uncertainty hourly (mm w.e.)'].astype(float)
        
        # unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])])
        # df.loc[days[i]+datetime.timedelta(hours=12), 'Uncertainty 12 hourly (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])])

        # unc_propagation(unc[unc.index.hour.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0])])
        # df.loc[days[i]+datetime.timedelta(hours=24), 'Uncertainty daily (mm w.e.)'] = unc_propagation(unc[unc.index.hour.isin([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0])])
        
        df.loc[days[i]+datetime.timedelta(hours=24), 'Uncertainty daily (mm w.e.)'] = unc_propagation(unc)
        
    df['Snowfall accumulation daily rate (mm w.e.)'] = df['Snowfall accumulation daily rate (mm w.e.)'].map('{:,.2f}'.format)
    df['Uncertainty daily (mm w.e.)'] = df['Uncertainty daily (mm w.e.)'].map('{:,.2f}'.format)

    return df


def logger_to_xcsv(p, filename='snowfall', filepath=''):
    """
    Save the preprocessed transmitted data to a .xcsv file formatted to use as input for the WebApp

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """

    # currently need this exact format
    dfout = p.input.copy()
    # dfout = dfout.interpolate('linear')
    # dfout = dfout.dropna(subset=['pressure'])
    dfout['datetime'] = dfout.index
    dfout['datetime'] = pd.to_datetime(dfout['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    dfout = dfout.loc[:, ['datetime', 'pressure', 'ground_temperature', 'water_temperature', 'battery_voltage']]
    dfout['pressure'] *= 0.09804  # converto m w.e. to bar

    dfout = dfout.rename(columns={'pressure': 'pressure (bar)', 
                                'ground_temperature': 'ground_temperature (Cel)',
                                'water_temperature': 'water_temperature (Cel)',
                                'battery_voltage': 'battery_voltage (V)'    })

    header = {
        'id': filename,
        'authors': 'Hamish Pritchard',
        'provider': 'British Antarctic Survey'
    }
    metadata = {'header': header, 'column_headers': {}}

    dataset = xcsv.XCSV(metadata=metadata, data=dfout)
    dataset.store_column_headers()

    if filepath == '':
        with xcsv.File(filename+'_logger_data.csv', mode='w') as f:
            f.write(dataset)
    else:
        with xcsv.File(filepath+'/logger_data'+'/'+filename+'_logger_data.csv', mode='w') as f:
            f.write(dataset)
    
    return


def transmitted_to_xcsv(p, filename='snowfall', filepath=''):
    """
    Save the preprocessed transmitted data to a .xcsv file formatted to use as input for the WebApp

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """

    # currently need this exact format
    dfout = p.input.copy()
    # dfout = dfout.interpolate('linear')
    # dfout = dfout.dropna(subset=['pressure'])
    dfout['datetime'] = dfout.index
    dfout['datetime'] = pd.to_datetime(dfout['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    dfout = dfout.loc[:, ['datetime', 'pressure', 'ground_temperature', 'water_temperature', 'battery_voltage']]
    dfout['pressure'] *= 0.09804  # converto m w.e. to bar

    dfout = dfout.rename(columns={'pressure': 'pressure (bar)', 
                                'ground_temperature': 'ground_temperature (Cel)',
                                'water_temperature': 'water_temperature (Cel)',
                                'battery_voltage': 'battery_voltage (V)'    })

    header = {
        'id': filename,
        'authors': 'Hamish Pritchard',
        'provider': 'British Antarctic Survey'
    }
    metadata = {'header': header, 'column_headers': {}}

    dataset = xcsv.XCSV(metadata=metadata, data=dfout)
    dataset.store_column_headers()

    if filepath == '':
        with xcsv.File(filename+'_transmitted_data.csv', mode='w') as f:
            f.write(dataset)
    else:
        with xcsv.File(filepath+'/transmitted_data'+'/'+filename+'_transmitted_data.csv', mode='w') as f:
            f.write(dataset)
    
    return


def transmitted_to_xcsv_ALL(dfout, filename='snowfall', filepath=''):
    """
    Save the preprocessed transmitted data to a .xcsv file formatted to use as input for the WebApp

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """

    # dfout = dfout.interpolate('linear')
    # dfout = dfout.dropna(subset=['pressure'])
    dfout['datetime'] = dfout.index
    dfout['datetime'] = pd.to_datetime(dfout['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    dfout = dfout.loc[:, ['datetime', 'pressure', 'ground_temperature', 'water_temperature', 'battery_voltage']]
    dfout['pressure'] *= 0.09804  # converto m w.e. to bar

    dfout = dfout.rename(columns={'pressure': 'pressure (bar)', 
                                'ground_temperature': 'ground_temperature (Cel)',
                                'water_temperature': 'water_temperature (Cel)',
                                'battery_voltage': 'battery_voltage (V)'    })

    header = {
        'id': filename,
        'authors': 'Hamish Pritchard',
        'provider': 'British Antarctic Survey'
    }
    metadata = {'header': header, 'column_headers': {}}

    dataset = xcsv.XCSV(metadata=metadata, data=dfout)
    dataset.store_column_headers()

    if filepath == '':
        with xcsv.File(filename+'_transmitted_data.csv', mode='w') as f:
            f.write(dataset)
    else:
        with xcsv.File(filepath+'/transmitted_data'+'/'+filename+'_transmitted_data.csv', mode='w') as f:
            f.write(dataset)
    
    return


# def save_transmitted(p, filename='snowfall', filepath=''):
#     """
#     ----------------------
#     ----- DEPRECATED ----- 
#     ----------------------

#     Save the preprocessed transmitted data to a .csv file formatted to use as input for the WebApp

#     :param p: snowfall processor type object returned from the snowfall software
#     :type: snowfall processor
#     :param filename: name of the file to save the DataFrame to
#     :type: string
#     :param filepath: path to the folder where to save the file to
#     :type: string
#     :return: saved file
#     :rtype: .csv
#     """

#     # currently need this exact format
#     dfout = p.input.copy()
#     # if 'dt' in dfout:
#     #     dfout = dfout.drop(columns=['dt', 'dt1', 'dt2', 'TIMESTAMP'])
#     dfout = dfout.interpolate('linear')
#     dfout = dfout.dropna(subset=['pressure'])
#     dfout['RECORD'] = np.nan
#     dfout['timestamp'] = dfout.index.strftime(('"%Y-%m-%d %H:%M:%S"')).astype(pd.StringDtype())
#     dfout['pressure'] /= 1.0e-7 * 10.1974
#     # dfout['pressure'] = dfout['pressure'].astype(int)
#     dfout['pressure'] = dfout['pressure'].apply(lambda x: '{:.0f}'.format(x))
#     dfout['ground_temperature'] /= 0.1
#     dfout['water_temperature'] /= 0.01
#     dfout['battery_voltage'] /= 0.1
#     dfout = dfout.reindex(columns=['timestamp', 'RECORD', 'pressure', 'water_temperature', 'ground_temperature', 'battery_voltage'])

#     header = '"TOA5" \n \n \n \n'
#     if filepath == '':
#         with open(filename+'_transmitted_data.csv', 'w') as fp:
#             fp.write(header)
#         dfout.to_csv(filename+'_transmitted_data.csv', index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
#     else:
#         with open(filepath+'/'+filename+'_transmitted_data.csv', 'w') as fp:
#             fp.write(header)
#         dfout.to_csv(filepath+'/'+filename+'_transmitted_data.csv', index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
    
#     return


# def save_transmitted_smoothed(p, filename='snowfall', filepath=''):
#     """
#     ----------------------
#     ----- DEPRECATED ----- 
#     ----------------------

#     Save the preprocessed transmitted data to a .csv file formatted to use as input for the WebApp

#     :param p: snowfall processor type object returned from the snowfall software
#     :type: snowfall processor
#     :param filename: name of the file to save the DataFrame to
#     :type: string
#     :param filepath: path to the folder where to save the file to
#     :type: string
#     :return: saved file
#     :rtype: .csv
#     """

#     # currently need this exact format
#     dfout = p.input.copy()
#     dfout = dfout.drop(columns=['dt', 'dt1', 'dt2', 'TIMESTAMP'])
#     dfout = dfout.interpolate('linear')
#     dfout = dfout.dropna(subset=['pressure'])
#     dfout['RECORD'] = np.nan
#     dfout['timestamp'] = dfout.index.strftime(('"%Y-%m-%d %H:%M:%S"')).astype(pd.StringDtype())
#     dfout['pressure'] /= 1.0e-7 * 10.1974
#     # dfout['pressure'] = dfout['pressure'].astype(int)
#     dfout['pressure'] = dfout['pressure'].apply(lambda x: '{:.0f}'.format(x))
#     dfout['ground_temperature'] /= 0.1
#     dfout['water_temperature'] /= 0.01
#     dfout['battery_voltage'] /= 0.1
#     dfout = dfout.reindex(columns=['timestamp', 'RECORD', 'pressure', 'water_temperature', 'ground_temperature', 'battery_voltage'])

#     header = '"TOA5" \n \n \n \n'
#     if filepath == '':
#         with open(filename+'_transmitted_data_smoothed.csv', 'w') as fp:
#             fp.write(header)
#         dfout.to_csv(filename+'_transmitted_data_smoothed.csv', index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
#     else:
#         with open(filepath+'/'+filename+'_transmitted_data_smoothed.csv', 'w') as fp:
#             fp.write(header)
#         dfout.to_csv(filepath+'/'+filename+'_transmitted_data_smoothed.csv', index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
    
#     return


# def save_logger(p, filename='snowfall', filepath=''):
#     """
#     ----------------------
#     ----- DEPRECATED ----- 
#     ----------------------

#     Save the preprocessed transmitted data to a .csv file formatted to use as input for the WebApp

#     :param p: snowfall processor type object returned from the snowfall software
#     :type: snowfall processor
#     :param filename: name of the file to save the DataFrame to
#     :type: string
#     :param filepath: path to the folder where to save the file to
#     :type: string
#     :return: saved file
#     :rtype: .csv
#     """

#     # currently need this exact format
#     dfout = p.input.copy()
#     # if 'dt' in dfout:
#     #     dfout = dfout.drop(columns=['dt', 'dt1', 'dt2', 'TIMESTAMP'])
#     dfout = dfout.interpolate('linear')
#     dfout = dfout.dropna(subset=['pressure'])
#     dfout['RECORD'] = np.nan
#     dfout['timestamp'] = dfout.index.strftime(('"%Y-%m-%d %H:%M:%S"')).astype(pd.StringDtype())
#     dfout['pressure'] /= 1.0e-7 * 10.1974
#     # dfout['pressure'] = dfout['pressure'].astype(int)
#     dfout['pressure'] = dfout['pressure'].apply(lambda x: '{:.0f}'.format(x))
#     dfout['ground_temperature'] /= 0.1
#     dfout['water_temperature'] /= 0.01
#     dfout['battery_voltage'] /= 0.1
#     dfout = dfout.reindex(columns=['timestamp', 'RECORD', 'pressure', 'water_temperature', 'ground_temperature', 'battery_voltage'])

#     header = '"TOA5" \n \n \n \n'
#     if filepath == '':
#         with open(filename+'_logger_data.csv', 'w') as fp:
#             fp.write(header)
#         dfout.to_csv(filename+'_logger_data.csv', index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
#     else:
#         with open(filepath+'/'+filename+'_logger_data.csv', 'w') as fp:
#             fp.write(header)
#         dfout.to_csv(filepath+'/'+filename+'_logger_data.csv', index=False, header=False, quoting=csv.QUOTE_NONE, mode='a')
    
#     return


def save_config(p, filename='snowfall', filepath=''):
    """
    Save the configuration used to run the snowfall software to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """

    keys = list({k:v for k,v in p.conf.items() if k not in {'gradient_conf'}}.keys()) + list(p.conf['gradient_conf'].keys())
    vals = list({k:v for k,v in p.conf.items() if k not in {'gradient_conf'}}.values()) + list(p.conf['gradient_conf'].values())
    types = {
        'interpolate': 'boolean', 
        'smoothing': 'intH', 
        'strategy': 'str', 
        'systematic_uncertainty': 'float', 
        'standard_window': 'int', 
        'events_file': 'str',
        'gradient_correction': 'boolean',
        'gradient_smoothing': 'intH',
        'event_duration_min_h': 'int',
        'event_amount_min_mwe': 'float',
        'event_merging': 'int',
        'search_window_extension': 'int',
        'order': 'int',
        'order_adjust': 'int',
        'attempts': 'int',
        'start_time': 'datetime',
        'end_time': 'datetime'
    }
    description = {
        'interpolate': 'linear interpolation of input data to avoid gaps in the data series',
        'smoothing': 'if given input data are smoothed using a windowed median with the given hours window (e.g. 12H)',
        'strategy': 'strategy used in the detection of start and end of snowfall events',
        'order': 'order used in the argrelextrema() function (OLD STRATEGY)',
        'order_adjust': 'order adjust parameter used in the (OLD STRATEGY)',
        'attempts': 'order adjustment attempts used in the (OLD STRATEGY)',
        'systematic_uncertainty': 'systematic uncertainty used in the uncertainties calculation',
        'start_time': 'start time of the analysis period, if None uses the data series from the start',
        'end_time': 'end time of the analysis period, if None uses the data series till the end',
        'standard_window': 'time window in hours to compute the before and after event gradients',
        'events_file': 'filepath.csv to external file to read start, end of snowfall and before and after windows, if None use given strategy and standard_window',
        'gradient_correction': 'if True uses the gradient correction (e.g. for non draining lakes)',
        'gradient_smoothing': 'number of hours used to compute the moving window median of the pressure gradient used in the gradient strategy snowfall detection algorithm',
        'event_duration_min_h': 'minimum snowfall event duration in hours for detection',
        'event_amount_min_mwe': 'minimum snowfall amount in mm w.e. for detection',
        'event_merging': 'hours threshold below which too close events are merged, if None do not merge',
        'search_window_extension': 'number of hours used to extend the search window for max and min positive gradients'
    }

    df = pd.DataFrame(data=np.transpose([keys, vals]), columns=['parameter', 'value'])
    for ii in range(0, len(df.index)):
        df.loc[df.parameter == keys[ii], 'type'] = types[keys[ii]]
        df.loc[df.parameter == keys[ii], 'description'] = description[keys[ii]]
    if filepath == '':
        df.to_csv(filename+'_config.csv', na_rep='None')
    else:
        df.to_csv(filepath+'/'+filename+'_config.csv', na_rep='None')
    

def save_events_windows(p, filename='snowfall', filepath=''):
    """
    Save the events_windows DataFrame from the snowfall to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """

    if filepath == '':
        p.breakpoints['events_windows'].to_csv(filename+'_events_windows.csv') 
    else:
        p.breakpoints['events_windows'].to_csv(filepath+'/'+filename+'_events_windows.csv') 


def save_output_summary(p, filename='snowfall', filepath=''):
    """
    Save the summary DataFrame returned by format_snowfall_output_summary() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    df = format_snowfall_output_summary(p)
    if filepath == '':
        df.to_csv(filename+'_summary.csv')
    else:
        df.to_csv(filepath+'/'+filename+'_summary.csv')


def save_output_hourly(p, filename='snowfall', filepath=''):
    """
    Save the hourly DataFrame returned by format_snowfall_output_summary() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    df = format_snowfall_output_hourly(p)
    if filepath == '':
        df.to_csv(filename+'_hourly.csv')
    else:
        df.to_csv(filepath+'/'+filename+'_hourly.csv')


def save_output_3hourly(p, filename='snowfall', filepath=''):
    """
    Save the hourly DataFrame returned by format_snowfall_output_summary() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    df = format_snowfall_output_3hourly(p)
    if filepath == '':
        df.to_csv(filename+'_3hourly.csv')
    else:
        df.to_csv(filepath+'/'+filename+'_3hourly.csv')


def save_output_6hourly(p, filename='snowfall', filepath=''):
    """
    Save the hourly DataFrame returned by format_snowfall_output_summary() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    df = format_snowfall_output_6hourly(p)
    if filepath == '':
        df.to_csv(filename+'_6hourly.csv')
    else:
        df.to_csv(filepath+'/'+filename+'_6hourly.csv')


def save_output_12hourly(p, filename='snowfall', filepath=''):
    """
    Save the hourly DataFrame returned by format_snowfall_output_summary() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    df = format_snowfall_output_12hourly(p)
    if filepath == '':
        df.to_csv(filename+'_12hourly.csv')
    else:
        df.to_csv(filepath+'/'+filename+'_12hourly.csv')


def save_output_daily(p, filename='snowfall', filepath=''):
    """
    Save the hourly DataFrame returned by format_snowfall_output_summary() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    df = format_snowfall_output_daily(p)
    if filepath == '':
        df.to_csv(filename+'_daily.csv')
    else:
        df.to_csv(filepath+'/'+filename+'_daily.csv')


def save_pickled_p(p, filename='snowfall', filepath=''):
    """
    Pickle the p dictionary from the snowfall software to file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: name of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved pickled file
    :rtype: .pkl
    """

    if filepath == '':
        with open('%s.pkl' % filename, 'wb') as f:
            pickle.dump(p, f)
    else:
        with open(filepath+'/'+filename+'.pkl', 'wb') as f:
            pickle.dump(p, f)


def save_output(p, filename='snowfall', filepath=''):
    """
    Save BOTH the summary and hourly DataFrames returned by format_snowfall_output_summary()
    and format_snowfall_output_hourly() to a .csv file

    :param p: snowfall processor type object returned from the snowfall software
    :type: snowfall processor
    :param filename: names of the file to save the DataFrame to
    :type: string
    :param filepath: path to the folder where to save the file to
    :type: string
    :return: saved file
    :rtype: .csv
    """
    
    save_config(p, filename, filepath)
    save_events_windows(p, filename, filepath)
    save_output_summary(p, filename, filepath)
    save_output_hourly(p, filename, filepath)
    save_output_3hourly(p, filename, filepath)
    save_pickled_p(p, filename, filepath)
    # save_output_6hourly(p, filename, filepath)
    # save_output_12hourly(p, filename, filepath)
    # save_output_daily(p, filename, filepath)
