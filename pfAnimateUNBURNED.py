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

def animate_unburned(ds):
    
    # get colors
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # set style
    style.use('seaborn-notebook')
    
    # set up figure
    fig = plt.figure(
        tight_layout = True,
        figsize = [13,5]
    )
    
    # define axes
    gs = gridspec.GridSpec(8,4)
    ax0 = fig.add_subplot(gs[3:,:])
    ax1 = fig.add_subplot(gs[0:3,0])
    ax2 = fig.add_subplot(gs[0:3,1])
    ax3 = fig.add_subplot(gs[0:3,2])
    ax4 = fig.add_subplot(gs[0:3,3])
    ax5 = fig.add_subplot(gs[3:,0])
    ax6 = fig.add_subplot(gs[3:,1])
    ax7 = fig.add_subplot(gs[3:,2])
    ax8 = fig.add_subplot(gs[3:,3])
    
    # get list of dates
    dates = pd.date_range(
        start = '2005-10-01',
        periods = 365,
        freq = 'D'
    )
    
    def update(frame):
        
        ##########################
        ### RUNOFF TIME SERIES ###
        ##########################
        
        # clear all
        ax1.cla()
        
        # add runoff time series
        ax1.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.net_runoff_unburned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax1.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).net_runoff_unburned.mean(dim=['group','sample']),
            color = c[0],
            alpha = 0.5
        )
        
        # housekeeping
        ax1.set_ylim([-100,1000])
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(top = False)
        ax1.tick_params(bottom = False)
        ax1.tick_params(left = False)
        ax1.tick_params(right = False)
        
        ##########################
        ### SOIL TIME SERIES ###
        ##########################
        
        # clear all
        ax2.cla()
        
        # add soil time series
        ax2.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.soil_unburned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax2.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).soil_unburned.mean(dim=['group','sample']),
            color = c[1],
            alpha = 0.5
        )
        
        # housekeeping
        ax2.set_ylim([-100,1000])
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
        
        ######################
        ### ET TIME SERIES ###
        ######################
        
        # clear all
        ax3.cla()
        
        # add soil time series
        ax3.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.et_unburned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.2
        )
        
        ax3.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).et_unburned.mean(dim=['group','sample']),
            color = c[2],
            alpha = 0.5
        )
        
        # housekeeping
        ax3.set_ylim([-100,1000])
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
        
        #######################
        ### SWE TIME SERIES ###
        #######################

        # clear all
        ax4.cla()
        
        # add swe timeseries
        ax4.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.swe_burned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.05
        )
        
        ax4.fill_between(
            x = ds.t.values,
            y1 = 0,
            y2 = ds.swe_unburned.mean(dim=['group','sample']),
            color = 'silver',
            alpha = 0.05
        )
        
        ax4.fill_between(
            x = ds.isel(t=slice(None,frame)).t.values,
            y1 = 0,
            y2 = ds.isel(t=slice(None,frame)).swe_unburned.mean(dim=['group','sample']),
            color = 'tab:grey',
            alpha = 0.3,
            label = 'Unburned'
        )
        
        ax4.fill_between(
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
            
        ax4.scatter(
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
        
        ax4.scatter(
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
        
        ax4.scatter(
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
        
        ax4.scatter(
            x = ds.swe_unburned.attrs['melted'],
            y = ds.swe_unburned.mean(dim=['group','sample']).sel(t=ds.swe_unburned.attrs['melted'])-60,
            marker = '^',
            color = mc,
            alpha = alph
        )
        
        # housekeeping
        ax4.set_ylim([-100,1000])
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
        ### RUNOFF SCATTER ###
        ######################
        
        # clear all
        ax5.cla()
        
        # add center line
        ax5.axhline(
            0,
            linestyle = '--',
            linewidth = 1,
            color = 'silver',
            alpha = 0.5
        )
        
        # add runoff mpd scatter
        ds.isel(t=frame).sel(group='MPD').plot.scatter(
            ax = ax5,
            x = 'init_soil_burned',
            y = 'norm_net_runoff_unburned',
            color = c[0],
            marker = 'o',
            add_guide = False,
            alpha = 0.6,
            label = 'MPD'
        )
        
        # add runoff nn scatter
        ds.isel(t=frame).sel(group='NN').plot.scatter(
            ax = ax5,
            x = 'init_soil_burned',
            y = 'norm_net_runoff_unburned',
            color = c[0],
            marker = 'x',
            add_guide = False,
            alpha = 0.3,
            label = 'NN'
        )
        
        # add runoff r scatter
        ds.isel(t=frame).sel(group='R').plot.scatter(
            ax = ax5,
            x = 'init_soil_burned',
            y = 'norm_net_runoff_unburned',
            color = c[0],
            marker = 's',
            add_guide = False,
            alpha = 0.1,
            label = 'R'
        )
        
        # housekeeping
        ax5.set_xlim([278,295])
        ax5.set_ylim([-25,25])
        ax5.set_ylabel('Depth [mm]')
        ax5.set_xlabel('')
        ax5.set_title('Net Runoff')
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.tick_params(left = False)
        ax5.tick_params(bottom = False)
        ax5.set_xticks([280,293])
        ax5.set_yticks([-25,0,25])
        ax5.legend(
            loc = 'lower right',
            fontsize = 'x-small'
        )
        
        ######################
        ### SOIL SCATTER ###
        ######################
        
        # clear all
        ax6.cla()
        
        # add center line
        ax6.axhline(
            0,
            linestyle = '--',
            linewidth = 1,
            color = 'silver',
            alpha = 0.5
        )
        
        # add soil mpd scatter
        ds.isel(t=frame).sel(group='MPD').plot.scatter(
            ax = ax6,
            x = 'init_soil_burned',
            y = 'norm_soil_unburned',
            color = c[1],
            marker = 'o',
            add_guide = False,
            alpha = 0.6,
            label = 'MPD'
        )
        
        # add soil nn scatter
        ds.isel(t=frame).sel(group='NN').plot.scatter(
            ax = ax6,
            x = 'init_soil_burned',
            y = 'norm_soil_unburned',
            color = c[1],
            marker = 'x',
            add_guide = False,
            alpha = 0.3,
            label = 'NN'
        )
        
        # add soil r scatter
        ds.isel(t=frame).sel(group='R').plot.scatter(
            ax = ax6,
            x = 'init_soil_burned',
            y = 'norm_soil_unburned',
            color = c[1],
            marker = 's',
            add_guide = False,
            alpha = 0.1,
            label = 'R'
        )
        
        # housekeeping
        ax6.set_xlim([278,295])
        ax6.set_ylim([-25,25])
        ax6.set_ylabel('')
        ax6.set_xlabel('')
        ax6.set_title('$\Delta$ Soil Storage')
        ax6.spines['right'].set_visible(False)
        ax6.spines['top'].set_visible(False)
        ax6.tick_params(left = False)
        ax6.tick_params(bottom = False)
        ax6.set_xticks([280,293])
        ax6.set_yticks([-25,0,25])
        ax6.legend(
            loc = 'lower right',
            fontsize = 'x-small'
        )
        
        ######################
        ### ET SCATTER ###
        ######################
        
        # clear all
        ax7.cla()
        
        # add center line
        ax7.axhline(
            0,
            linestyle = '--',
            linewidth = 1,
            color = 'silver',
            alpha = 0.5
        )
        
        # add et mpd scatter
        ds.isel(t=frame).sel(group='MPD').plot.scatter(
            ax = ax7,
            x = 'init_soil_burned',
            y = 'norm_et_unburned',
            color = c[2],
            marker = 'o',
            add_guide = False,
            alpha = 0.6,
            label = 'MPD'
        )
        
        # add et nn scatter
        ds.isel(t=frame).sel(group='NN').plot.scatter(
            ax = ax7,
            x = 'init_soil_burned',
            y = 'norm_et_unburned',
            color = c[2],
            marker = 'x',
            add_guide = False,
            alpha = 0.3,
            label = 'NN'
        )
        
        # add et r scatter
        ds.isel(t=frame).sel(group='R').plot.scatter(
            ax = ax7,
            x = 'init_soil_burned',
            y = 'norm_et_unburned',
            color = c[2],
            marker = 's',
            add_guide = False,
            alpha = 0.1,
            label = 'R'
        )
        
        # housekeeping
        ax7.set_xlim([278,295])
        ax7.set_ylim([-25,25])
        ax7.set_ylabel('')
        ax7.set_xlabel('')
        ax7.set_title('ET')
        ax7.spines['right'].set_visible(False)
        ax7.spines['top'].set_visible(False)
        ax7.tick_params(left = False)
        ax7.tick_params(bottom = False)
        ax7.set_xticks([280,293])
        ax7.set_yticks([-25,0,25])
        ax7.legend(
            loc = 'lower right',
            fontsize = 'x-small'
        )
        
        ######################
        ### SWE SCATTER ###
        ######################
        
        # clear all
        ax8.cla()
        
        # add center line
        ax8.axhline(
            0,
            linestyle = '--',
            linewidth = 1,
            color = 'silver',
            alpha = 0.5
        )
        
        # add swe mpd scatter
        ds.isel(t=frame).sel(group='MPD').plot.scatter(
            ax = ax8,
            x = 'init_soil_burned',
            y = 'norm_swe_unburned',
            color = 'tab:grey',
            marker = 'o',
            add_guide = False,
            alpha = 0.6,
            label = 'MPD'
        )
        
        # add swe nn scatter
        ds.isel(t=frame).sel(group='NN').plot.scatter(
            ax = ax8,
            x = 'init_soil_burned',
            y = 'norm_swe_unburned',
            color = 'tab:grey',
            marker = 'x',
            add_guide = False,
            alpha = 0.3,
            label = 'NN'
        )
        
        # add swe r scatter
        ds.isel(t=frame).sel(group='R').plot.scatter(
            ax = ax8,
            x = 'init_soil_burned',
            y = 'norm_swe_unburned',
            color = 'tab:grey',
            marker = 's',
            add_guide = False,
            alpha = 0.1,
            label = 'R'
        )
        
        # housekeeping
        ax8.set_xlim([278,295])
        ax8.set_ylim([-25,25])
        ax8.set_ylabel('')
        ax8.set_xlabel('')
        ax8.set_title('SWE')
        ax8.spines['right'].set_visible(False)
        ax8.spines['top'].set_visible(False)
        ax8.tick_params(left = False)
        ax8.tick_params(bottom = False)
        ax8.set_xticks([280,293])
        ax8.set_yticks([-25,0,25])
        ax8.legend(
            loc = 'lower right',
            fontsize = 'x-small'
        )
        
        ##########################
        ### FINAL HOUSEKEEPING ###
        ##########################
        
        fig.suptitle('Unburned Areas\n' + str(dates[frame])[:10])
        
        ax0.spines['top'].set_visible(False)
        ax0.spines['bottom'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.tick_params(top = False)
        ax0.tick_params(bottom = False)
        ax0.tick_params(left = False)
        ax0.tick_params(right = False)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_xlabel('Initial Soil Storage [mm]',labelpad=25)
        
        
    anim = animation.FuncAnimation(
        fig,
        update,
        frames = 365,
        interval = 100,
        repeat = False
    )
    
    plt.close()
    
    return anim