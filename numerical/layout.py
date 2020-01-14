import numpy as np
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, HTML, clear_output
import matplotlib.pyplot as plt
from numerical.rhsfuns import RhsFuncs
from numerical.integrators import integrators

class Counter():
    def __init__(self,progress_bar):
        self.theCount = 1
        self.progress_bar = progress_bar
        self.progress_bar.value = 1

    def __call__(self):
        self.theCount += 1
        self.progress_bar.value = self.theCount
        return self.theCount

class PlotStyling(object):

    def __init__(self):
        self.tabs = []
        keys = ['dt','tend','xmax','xmin','sol','time','x','dx']
        self.data=dict.fromkeys(keys)
        self.display_objects=[]
        # self.draw_layout()
        self.resultsTab()
        self.numerical_setup()
        self.stylingTab()
        self.tabs_disiplay()

        self.__display(self.tab)

        self.on_change()
        self.generateButtonPressed = False

    def __display(self,obj):
        display(obj)

    def tabs_disiplay(self):
        self.tab = widgets.Tab(children=self.tabs)
        self.tab.set_title(0, 'results')
        self.tab.set_title(1, 'Numerical Setup')
        self.tab.set_title(2, 'style')

    def stylingTab(self):
        self.grid_button_widget = widgets.ToggleButton(
            value=True,
            description='Grid',
            icon='check'
        )
        self.color_buttons_widget = widgets.ToggleButtons(
            options=['blue', 'red', 'green'],
            description='Color:',
        )
        self.xlim_min_widget = widgets.widgets.FloatText(
            value=-4.0,
            description='min xlim:',
            disabled=False
        )
        self.xlim_max_widget = widgets.widgets.FloatText(
            value=4.0,
            description='max xlim:',
            disabled=False
        )
        self.ylim_min_widget = widgets.widgets.FloatText(
            value=-0.5,
            description='min ylim:',
            disabled=False
        )
        self.ylim_max_widget = widgets.widgets.FloatText(
            value=1.25,
            description='max ylim:',
            disabled=False
        )
        self.disable_axislim_widget = widgets.ToggleButton(
            value=True,
            description='axislim',
            icon='check'
        )
        self.xaxis_label_widget = widgets.Textarea(
            value='x',
            description='x-axis label:',
            disabled=False
        )
        self.yaxis_label_widget = widgets.Textarea(
            value='y',
            description='y-axis label:',
            disabled=False
        )
        self.styleTab= \
            VBox(children=[
                HBox(children=[
                   self.xaxis_label_widget,
                   self.yaxis_label_widget
                ]),
                HBox(children=[
                    self.color_buttons_widget]),
                HBox(children=[
                    self.xlim_min_widget,
                    self.xlim_max_widget
                ]),
                HBox(children=[
                    self.ylim_min_widget,
                    self.ylim_max_widget
                ]),
                HBox(children=[
                    self.disable_axislim_widget,
                    self.grid_button_widget
                ])

            ])
        self.tabs.append(self.styleTab)

    def resultsTab(self):
        self.dt_widget = widgets.FloatText(
            value=0.01,
            description='dt:',
            disabled=False
        )
        self.dx_widget = widgets.FloatText(
            value=0.1,
            description='dx:',
            disabled=False
        )
        self.tend_widget = widgets.FloatText(
            value=10.0,
            description='tend:',
            disabled=False
        )
        self.minXlim_widget = widgets.FloatText(
            value=-4.0,
            description='x-min:',
            disabled=False
        )
        self.maxXlim_widget = widgets.FloatText(
            value=4.0,
            description='x-max:',
            disabled=False
        )

        self.CFL_widget = widgets.FloatText(
            value=0.1,
            description='CFL:',
            disabled=True
        )

        self.Fr_widget = widgets.FloatText(
            value=0.0,
            description='Fr:',
            disabled=True
        )

        self.progressBar_widget = widgets.IntProgress(
            value=0,
            min=0,
            max=int(self.tend_widget.value/self.dt_widget.value),
            step=1,
            description='Progress:',
            bar_style='',  # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal'
        )


        self.generate_button_widget = widgets.Button(
            description='Generate'
        )

        self.edit_button_widget = widgets.Button(
            description='Edit',
            disabled=True
        )

        self.savefig_button_widget = widgets.Button(
            description='save fig',
            disabled=True
        )

        self.timestep_widget = widgets.IntSlider(
            description='Timestep',
            min=0,
            max=int(self.tend_widget.value/self.dt_widget.value)-1,
            step=1,
            value=0
        )
        self.resultTab = \
            VBox(children=[
                HBox(children=[
                    self.dt_widget,
                    self.tend_widget
                ]),
                HBox(children=[
                    self.dx_widget,
                    self.minXlim_widget,
                    self.maxXlim_widget]),
                HBox(children=[
                    self.CFL_widget,
                    self.Fr_widget
                ]),
                HBox(children=[
                    self.generate_button_widget,
                    self.edit_button_widget,
                    self.progressBar_widget
                ]),
                HBox(children=[self.timestep_widget,
                               self.savefig_button_widget])
            ])
        self.tabs.append(self.resultTab)

    def numerical_setup(self):
        self.integrator_buttons_widget = widgets.ToggleButtons(
            options=['Forward Euler', 'Backward Euler', 'Crank Nicholson'],
            value='Backward Euler',
            description='Integrator:',
        )

        self.advection_buttons_widget = widgets.ToggleButtons(
            options=['Forward Difference', 'Backward Difference', 'Central Difference'],
            value='Central Difference',
            description='Advection:',
        )
        self.diffusion_buttons_widget = widgets.ToggleButtons(
            options=['Forward Difference', 'Backward Difference', 'Central Difference'],
            description='Diffusion:',
        )

        self.phi0_widget = widgets.Textarea(
            value='np.exp(-x**2/0.25)',
            description='f(t=0):'
        )
        self.advection_vel_widget=widgets.FloatText(
            value=1.0,
            description='Advec vel'
        )

        self.diffusion_coef_widget = widgets.FloatText(
            value=0.0,
            description='Diff coef:'
        )
        self.numerical_setup = \
            VBox(children=[
                self.integrator_buttons_widget,
                self.advection_vel_widget,
                self.advection_buttons_widget,
                self.diffusion_coef_widget,
                self.diffusion_buttons_widget,
                self.phi0_widget
            ])
        self.tabs.append(self.numerical_setup)


    def on_change(self):
        self.generate_button_widget.on_click(self.generate_pressed)
        self.dx_widget.observe(self.update_CFL,names='value')
        self.dt_widget.observe(self.update_CFL, names='value')

        self.dx_widget.observe(self.update_Fr, names='value')
        self.dt_widget.observe(self.update_Fr, names='value')
        self.advection_vel_widget.observe(self.update_CFL, names='value')
        self.diffusion_coef_widget.observe(self.update_Fr, names='value')
        self.dt_widget.observe(self.update_maximum_timesteps, names='value')
        self.tend_widget.observe(self.update_maximum_timesteps, names='value')

        self.timestep_widget.observe(self.update_plot,names='value')
        self.edit_button_widget.on_click(self.edit_button_pressed)
        self.savefig_button_widget.on_click(self.savefig_button_pressed)

    def savefig_button_pressed(self,change):
        plt.savefig('./CFL_'+str(self.CFL_widget.value)+'_'+'Fr_'+str(self.Fr_widget.value)+'.pdf',transparent=True)

    def update_progress_bar(self):
        self.progressBar_widget.max= int(self.tend_widget.value/self.dt_widget.value)
        # print(self.progressBar_widget.max)

    def update_CFL(self,change):
        self.CFL_widget.value=self.advection_vel_widget.value*self.dt_widget.value/self.dx_widget.value

    def update_Fr(self,change):
        self.Fr_widget.value=self.diffusion_coef_widget.value*self.dt_widget.value/self.dx_widget.value/self.dx_widget.value

    def update_maximum_timesteps(self,change):
        self.timestep_widget.max= int(self.tend_widget.value/self.dt_widget.value) -1

    def enable_widgets(self,bool):
        self.dt_widget.disabled=bool
        self.dx_widget.disabled=bool
        self.tend_widget.disabled=bool
        self.minXlim_widget.disabled=bool
        self.maxXlim_widget.disabled=bool
        self.integrator_buttons_widget.disabled=bool
        self.advection_buttons_widget.disabled=bool
        self.diffusion_buttons_widget.disabled=bool
        self.advection_vel_widget.disabled=bool
        self.diffusion_coef_widget.disabled=bool
        self.phi0_widget.disabled=bool
        self.timestep_widget.disabled=not bool
        self.savefig_button_widget.disabled=not bool

    def collect_data(self):
        self.data['dt'] = self.dt_widget.value
        self.data['tend']=self.tend_widget.value
        self.data['dx']=self.dx_widget.value
        self.data['xmin']=self.minXlim_widget.value
        self.data['xmax']=self.maxXlim_widget.value
        self.data['time'] = np.arange(0, self.data['tend'], self.data['dt'])
        self.data['x'] = np.arange(self.data['xmin'], self.data['xmax'], self.data['dx'])
        self.myCount = Counter(self.progressBar_widget)
        integrator = integrators(self.integrator_buttons_widget.value,self.myCount)
        self.integ = integrator()
        self.data['advection_stencil']=self.advection_buttons_widget.value
        self.data['diffusion_stencil']=self.diffusion_buttons_widget.value
        self.data['advec_vel'] = self.advection_vel_widget.value
        self.data['diffusion_coef'] = self.diffusion_coef_widget.value
        rhsfun =  RhsFuncs(self.data['advection_stencil'],  self.data['diffusion_stencil'])
        self.rhsfunc =rhsfun(self.data['dx'],advection_vel=self.data['advec_vel'], diffusion_coef=self.data['diffusion_coef'])
        self.bcfun = rhsfun.periodicbcfun
        self.phi0 = lambda x : eval(self.phi0_widget.value)

        self.data['color']=self.color_buttons_widget.value
        self.data['grid']=self.grid_button_widget.value
        self.data['xlabel']=self.xaxis_label_widget.value
        self.data['ylabel']=self.yaxis_label_widget.value
        self.data['xminlim']=self.xlim_min_widget.value
        self.data['xmaxlim']=self.xlim_max_widget.value
        self.data['yminlim'] = self.ylim_min_widget.value
        self.data['ymaxlim'] = self.ylim_max_widget.value
        self.data['doaxislim']=self.disable_axislim_widget.value
        self.data['max_time_step']=int(self.tend_widget.value/self.dt_widget.value)-1

    def generate_solution(self):
        self.sol=None

        self.data['sol'] = self.integ(self.rhsfunc, self.phi0(self.data['x']), self.data['time'], self.bcfun())

    def generate_pressed(self, *args):
        self.enable_widgets(True)
        self.update_progress_bar()
        self.collect_data()
        self.generate_solution()
        self.edit_button_widget.disabled=False
        self.init()

    def init(self):
        self.l, = plt.plot(self.data['x'], self.data['sol'][0][1:-1], lw=2, color=self.data['color'])
        if self.data['doaxislim']:
            plt.xlim([self.data['xminlim'],self.data['xmaxlim']])
            plt.ylim([self.data['yminlim'],self.data['ymaxlim']])
        plt.xlabel(self.data['xlabel'])
        plt.ylabel(self.data['ylabel'])
        plt.grid(self.data['grid'])


    def update_plot(self, change):
        timestep = self.timestep_widget.value
        sol = self.data['sol'][timestep][1:-1]
        self.l.set_ydata(sol)
        plt.show()

    def edit_button_pressed(self,*args):
        plt.cla()
        self.enable_widgets(False)
