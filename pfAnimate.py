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
    fig = plt.figure(tight_layout=True,figsize=[12,6])
    gs = gridspec.GridSpec(4,4)
    ax1 = fig.add_subplot(gs[:,0:2])
    ax2 = fig.add_subplot(gs[0,2:])
    ax3 = fig.add_subplot(gs[1,2:])
    ax4 = fig.add_subplot(gs[2,2:])
    ax5 = fig.add_subplot(gs[3,2:])
    
    # get list of dates
    dates = pd.date_range(
        start = '2005-10-01',
        periods = 365,
        freq = 'D'
    )
    
    def update(frame):
        
        #####################
        ### SCATTER PLOTS ###
        #####################
        
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
        
        # add net runoff scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_prop',
            y = 'ctd_net_runoff',
            color = c[0],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'Net Runoff'
        )
        
        # add soil storage scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_prop',
            y = 'ctd_soil_storage',
            color = c[1],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = '$\Delta$ Soil Storage'
        )
        
        # add et scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_prop',
            y = 'ctd_et',
            color = c[2],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'ET'
        )
        
        # housekeeping
        ax1.set_ylim([-8,8])
        ax1.set_ylabel('Cumulative Normalized Depth [mm per unit area]')
        ax1.set_xlabel('Unburned Area Downstream of Fire [sq km]')
        ax1.set_title('')
        ax1.grid(False)
        ax1.spines['bottom'].set_color('k')
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['left'].set_color('k')
        ax1.spines['left'].set_linewidth(1)
        
        #######################
        ### SWE TIME SERIES ###
        #######################

        # clear all
        ax2.cla()
        
        # add timeseries
        ax2.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.burned_swe.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.1
        )
        
        ax2.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.unburned_swe.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.1
        )
        
        ax2.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).unburned_swe.mean(dim=['group','sample']),
            color = c[3],
            alpha = 0.4,
            label = 'Unburned'
        )
        
        ax2.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).burned_swe.mean(dim=['group','sample']),
            color = c[5],
            alpha = 0.3,
            label = 'Burned'
        )
        
        # housekeeping
        ax2.grid(False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylabel('SWE',rotation=0)
        ax2.legend(loc='upper right')
        ax2.spines['left'].set_visible(False)
        
        ##########################
        ### RUNOFF TIME SERIES ###
        ##########################

        # clear all
        ax3.cla()
        
        # add timeseries
        ax3.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.mean_net_runoff.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax3.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).mean_net_runoff.mean(dim=['group','sample']),
            color = c[0],
            alpha = 0.5
        )
        
        # housekeeping
        ax3.grid(False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.set_ylabel('Net\nRunoff',rotation=0)
        ax3.spines['left'].set_visible(False)
        
        ########################
        ### SOIL TIME SERIES ###
        ########################

        # clear all
        ax4.cla()
        
        # add timeseries
        ax4.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.mean_soil_storage.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax4.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).mean_soil_storage.mean(dim=['group','sample']),
            color = c[1],
            alpha = 0.5
        )
        
        # housekeeping
        ax4.grid(False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.set_ylabel('$\Delta$ Soil\nStorage',rotation=0)
        ax4.spines['left'].set_visible(False)
        
        ######################    
        ### ET TIME SERIES ###
        ###################### 

        # clear all
        ax5.cla()

        # add timeseries
        ax5.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.mean_et.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax5.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).mean_et.mean(dim=['group','sample']),
            color = c[2],
            alpha = 0.5
        )

        # housekeeping
        ax5.grid(False)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax5.set_ylabel('ET',rotation=0)
        ax5.spines['left'].set_visible(False)
        
        ##########################
        ### FINAL HOUSEKEEPING ###
        ##########################
        
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
        