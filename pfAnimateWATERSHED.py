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

def animate_watershed(ds):
    
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
        
        # add swe scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_cells',
            y = 'norm_swe_watershed',
            color = c[3],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'SWE'
        )
        
        # add et scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_cells',
            y = 'norm_et_watershed',
            color = c[2],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'ET'
        )
        
        # add net runoff scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_cells',
            y = 'norm_net_runoff_watershed',
            color = c[0],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = 'Net Runoff'
        )
        
        # add soil storage scatter
        ds.isel(t=frame).plot.scatter(
            ax = ax1,
            x = 'downstream_cells',
            y = 'norm_soil_watershed',
            color = c[1],
            marker = 'o',
            add_guide = False,
            alpha = 0.5,
            label = '$\Delta$ Soil Storage'
        )
        
        # housekeeping
        ax1.set_ylim([-12.5,12.5])
        ax1.set_ylabel('Depth [mm]')
        ax1.set_xlabel('Fire Downstream Area [sq km]')
        ax1.set_title('')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(left = False)
        ax1.tick_params(bottom = False)
        ax1.set_xticks([0,300])
        ax1.set_yticks([-12,0,12])
        ax1.legend(loc = 'lower center')
        
        #######################
        ### SWE TIME SERIES ###
        #######################

        # clear all
        ax2.cla()
        
        # add timeseries
        ax2.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.swe_burned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.05
        )
        
        ax2.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.swe_unburned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.05
        )
        
        ax2.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).swe_unburned.mean(dim=['group','sample']),
            color = 'tab:grey',
            alpha = 0.3,
            label = 'Unburned'
        )
        
        ax2.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).swe_burned.mean(dim=['group','sample']),
            color = c[3],
            alpha = 0.5,
            label = 'Burned'
        )
        
        # add peak and melt markers
        
        if frame < dates.get_loc(ds.swe_burned.attrs['peak']):
            mc = 'silver'
            alph = 0.05
        
        else:
            mc = c[3]
            alph = 0.5
            
        ax2.scatter(
            x = ds.swe_burned.attrs['peak'],
            y = ds.swe_burned.mean(dim=['group','sample']).sel(t=ds.swe_burned.attrs['peak'])+60,
            marker = 'v',
            color = mc,
            alpha = alph
        )
        
        if frame < dates.get_loc(ds.swe_unburned.attrs['peak']):
            mc = 'silver'
            alph = 0.1
        
        else:
            mc = 'tab:grey'
            alph = 0.5
        
        ax2.scatter(
            x = ds.swe_unburned.attrs['peak'],
            y = ds.swe_unburned.mean(dim=['group','sample']).sel(t=ds.swe_unburned.attrs['peak'])+60,
            marker = 'v',
            color = mc,
            alpha = alph
        )
        
        if frame < dates.get_loc(ds.swe_burned.attrs['melted']):
            mc = 'silver'
            alph = 0.05
        
        else:
            mc = c[3]
            alph = 0.5
        
        ax2.scatter(
            x = ds.swe_burned.attrs['melted'],
            y = ds.swe_burned.mean(dim=['group','sample']).sel(t=ds.swe_burned.attrs['melted'])-60,
            marker = '^',
            color = mc,
            alpha = alph
        )
        
        if frame < dates.get_loc(ds.swe_unburned.attrs['melted']):
            mc = 'silver'
            alph = 0.05
        
        else:
            mc = 'tab:grey'
            alph = 0.5
        
        ax2.scatter(
            x = ds.swe_unburned.attrs['melted'],
            y = ds.swe_unburned.mean(dim=['group','sample']).sel(t=ds.swe_unburned.attrs['melted'])-60,
            marker = '^',
            color = mc,
            alpha = alph
        )
        
        # housekeeping
        ax2.set_ylabel('SWE',rotation=0,loc='center')
        ax2.legend(loc='center right')
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(top = False)
        ax2.tick_params(bottom = False)
        ax2.tick_params(left = False)
        ax2.tick_params(right = False)
        
        ##########################
        ### RUNOFF TIME SERIES ###
        ##########################

        # clear all
        ax3.cla()
        
        # add timeseries
        ax3.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.net_runoff_watershed.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax3.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).net_runoff_watershed.mean(dim=['group','sample']),
            color = c[0],
            alpha = 0.5
        )
        
        # housekeeping
        ax3.set_ylabel('Net\nRunoff',rotation=0,loc='center')
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(top = False)
        ax3.tick_params(bottom = False)
        ax3.tick_params(left = False)
        ax3.tick_params(right = False)
        
        ########################
        ### SOIL TIME SERIES ###
        ########################

        # clear all
        ax4.cla()
        
        # add timeseries
        ax4.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.soil_watershed.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax4.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).soil_watershed.mean(dim=['group','sample']),
            color = c[1],
            alpha = 0.5
        )
        
        # housekeeping
        ax4.set_ylabel('$\Delta$ Soil\nStorage',rotation=0,loc='center')
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)
        ax4.spines['top'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(top = False)
        ax4.tick_params(bottom = False)
        ax4.tick_params(left = False)
        ax4.tick_params(right = False)
        
        ######################    
        ### ET TIME SERIES ###
        ###################### 

        # clear all
        ax5.cla()

        # add timeseries
        ax5.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.et_watershed.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax5.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).et_watershed.mean(dim=['group','sample']),
            color = c[2],
            alpha = 0.5
        )

        # housekeeping
        ax5.set_ylabel('ET',rotation=0,loc='center')
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax5.spines['top'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['left'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        ax5.tick_params(top = False)
        ax5.tick_params(bottom = True)
        ax5.tick_params(left = False)
        ax5.tick_params(right = False)
        
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