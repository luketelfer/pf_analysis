import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams, animation
from matplotlib import style
from matplotlib.ticker import MaxNLocator
from datetime import datetime,timedelta
matplotlib.rcParams['animation.embed_limit'] = 2**128
style.use('fivethirtyeight')

def animate_watershed_scatter(ds):
    
    # update style
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    style.use('seaborn-notebook')
    
    # set up figure
    fig = plt.figure(tight_layout=True,figsize=[10,6])
    gs = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs[:,:])
    
    # get list of dates
    dates = pd.date_range(
        start = '2005-10-01',
        periods = 365,
        freq = 'D'
    )
    
    # function to update plot
    def update(frame):
        
        # clear all
        ax1.cla()
        
        # add center line
        ax1.axhline(
            0,
            linestyle = '--',
            linewidth = 1,
            color = 'silver',
            alpha = 0.5
        )
        
        # add net runoff
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_prop',
            y = 'ctd_net_runoff',
            color = c[0],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'Cumulative Net Runoff'
        )
        
        # add soil storage
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_prop',
            y = 'ctd_soil_storage',
            color = c[1],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'Cumulative $\Delta$ Soil Storage'
        )
        
        # add et
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_prop',
            y = 'ctd_et',
            color = c[2],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'Cumulative ET'
        )
        
        # stylistic reset
        ax1.set_ylim([-8,8])
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.set_title('')
        ax1.grid(False)
        ax1.spines['bottom'].set_color('k')
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['left'].set_color('k')
        ax1.spines['left'].set_linewidth(1)
        ax1.legend(loc = 'lower center')
    
        # show date
        fig.suptitle(str(dates[frame])[:10])
        
    anim = animation.FuncAnimation(
        fig,
        update,
        frames = 365,
        interval = 100,
        repeat = False
    )
    
    plt.close()
    
    return anim
        