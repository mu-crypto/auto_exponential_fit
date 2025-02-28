import sys
import numpy as np
from PyQt6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QLineEdit, QLabel, QCheckBox,QTextEdit, QComboBox, QFileDialog,QDialog,QMessageBox
from PyQt6 import QtCore
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QTimer
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
from pathlib import Path
import pymc as pm
import arviz as az
from scipy.signal import savgol_filter
from scipy.signal import savgol_filter, find_peaks
import pytensor
import ruptures as rpt
import pyqtgraph as pg    
from time import sleep    
from openpyxl import Workbook, load_workbook


pytensor.config.floatX = "float32"

""""
Auto-exponential fitter (by Daniel Ohm)

Features:

1. Bayesian exponential model selection
2. Automatic falling edge finding and fitting
3. Optional savgol filter before finding peaks


Select peak indices convention

(peak index)(any string)(peak index)

"""



class Worker(QObject):
    finished = pyqtSignal()
    def __init__(self, time, counts,index_list,sweep_range, sweep_steps, raw_data,data_folder, samples, cores,chains,n_exp,exponential_label,progress_label,iteration_label,parent=None):
        super().__init__(parent)
        self.time = time
        self.counts  = counts
        self.raw_data = raw_data
        self.sweep_range = sweep_range
        self.sweep_steps = sweep_steps
        self.index_list = index_list
        self.n_exp = n_exp
        self.samples = samples
        self.cores = cores
        self.chains = chains
        self.data_folder = data_folder
        self.exponential_label = exponential_label
        self.progress_label = progress_label
        self.iteration_label = iteration_label



    
    def bayesian_inference(self):

        pytensor.config.floatX = "float32"
        for idx, indices in enumerate(self.index_list):      
                file_name = self.data_folder + "/" + Path(self.raw_data).name[:-3]+f"comparison_truncation_sweep_range_{self.sweep_range}_peak_{idx}.xlsx"
                df = pd.DataFrame()
                df.to_excel(file_name,index=False)
                df_fit = pd.DataFrame()
                df_fit.to_excel(file_name[:-5] + "_y_fit.xlsx")
                fit_columns = []
                for j in range(self.n_exp):
                    columns = ["trunc_x","trunc_y","elpd","p_loo","se","r_squared"]
                    for k in range(j):
                        columns.append(f"a{k}")
                    for k in range(j):
                        columns.append(f"a{k}_err")
                    for k in range(j):
                        columns.append(f"a{k}_rhat")
                    
                    for k in range(j+1):
                        columns.append(f"t{k}")
                    for k in range(j+1):
                        columns.append(f"t{k}_err")
                    for k in range(j+1):
                        columns.append(f"t{k}_rhat")

                    columns.append("C_val")
                    columns.append("C_err")
                    df = pd.DataFrame(columns=columns)
                    
                    with pd.ExcelWriter(file_name, engine="openpyxl") as writer:
                        df.to_excel(writer, sheet_name=f"exp_{j+1}", index=False)

                    with pd.ExcelWriter(file_name[:-5]+"_y_fit.xlsx", engine="openpyxl") as writer:
                        df_fit.to_excel(writer, sheet_name=f"y_fit_exp_{j+1}", index=False)
                    sweep_space = [int(var) for var in np.linspace(0,abs(indices[1]-indices[0])*self.sweep_range,self.sweep_steps)]
                    iters  = 1
                    self.exponential_label.setText(f"Exponential Model {j+1}")
                    r_squared_total = []
                    for idx1, sweep1 in enumerate(sweep_space):
                        for idx2, sweep2 in enumerate(sweep_space):
                            self.progress_label.setText(f"Truncating {sweep1} and Extending {sweep2}")
                            self.iteration_label.setText(f"Iteration {iters} of {len(sweep_space)**2}")
            
                            iters +=1
                            trunc_time = self.time[indices[0]+sweep1:indices[1]+sweep2]
                            trunc_counts = self.counts[indices[0]+sweep1:indices[1]+sweep2]
                            y_obs = np.log(trunc_counts / max(trunc_counts))
                            t =  trunc_time - min(trunc_time)
                            with pm.Model() as model:
                                if j == 0:
                                    tau = pm.Normal(f"tau", mu=10,sigma=0.5, shape=j+1)
                                elif j == 1:
                                    A = pm.Normal("A",mu=0,sigma=1,shape=j)
                                    tau = pm.Normal(f"tau", mu=[21,10],sigma=1, shape=j+1)
                                else:
                                    A = pm.Normal("A",mu=[0,0]+[0]*(j-2),sigma=1,shape=j)
                                    tau = pm.Normal(f"tau", mu=[21,10]+[10]*(j-1),sigma=1, shape=j+1)

                                C = pm.Normal(f"C", mu=0, sigma=1)
                                sigma = pm.HalfNormal("sigma", sigma=2)
                                if j ==0:
                                    y_model = np.log(pm.math.exp(-t/tau)+C)
                                else:
                                    y_model = np.log((pm.math.exp(-t/tau[0])+ pm.math.sum(A * pm.math.exp(-t[:,None]/tau[1:]), axis=1))/(1+pm.math.sum(A))+C)
                        
                                    

                                y_likelihood = pm.Normal("y", mu=y_model, sigma=sigma, observed=y_obs)

                                trace = pm.sample(self.samples, cores=self.cores, chains=self.chains, tune=1000, return_inferencedata=True,nuts_sampler="numpyro",init="adapt_full",target_accept=0.9,idata_kwargs={"log_likelihood": True})


                                
                                ppc = pm.sample_posterior_predictive(trace) 
                                y_pred_samples = ppc.posterior_predictive["y"].values  
                                y_pred = np.mean(y_pred_samples[0],axis=0) 
                            y_fit = np.exp(y_pred)
                            r_squared = r2_score(y_obs, y_pred)
                            rhat = az.rhat(trace)
                            pen = pg.mkPen(color=(255, 0, 0), width=1)
                            max_length = abs(indices[1]-indices[0] + sweep_space[-1])
                            blank_col = pd.DataFrame({'':['']*max_length})
                            col_time = pd.DataFrame({f"trunc_x_{sweep1}_y_{sweep2}_time": trunc_time})
                            col_raw =  pd.DataFrame({f"trunc_x_{sweep1}_y_{sweep2}_counts": trunc_counts})
                            col_fit = pd.DataFrame({f"trunc_x_{sweep1}_y_{sweep2}_fit": y_fit})

                            col_time = col_time.reindex(range(max_length))
                            col_raw = col_raw.reindex(range(max_length))
                            col_fit = col_fit.reindex(range(max_length))

                            fit_columns.append(col_time)
                            fit_columns.append(col_raw)
                            fit_columns.append(col_fit)
                            fit_columns.append(blank_col)
                            

                            if j !=0:
                                A_val = trace.posterior["A"].mean(dim=["chain", "draw"]).values
                                A_err = np.sqrt(trace.posterior["A"].var(dim=["chain", "draw"]).values)
                                A_rhat = rhat["A"].values
                            tau_val = trace.posterior["tau"].mean(dim=["chain", "draw"]).values
                            tau_err = np.sqrt(trace.posterior["tau"].var(dim=["chain", "draw"]).values)
                            C_val = trace.posterior["C"].mean(dim=["chain", "draw"]).values
                            C_err = np.sqrt(trace.posterior["C"].var(dim=["chain", "draw"]).values)
                            
                            tau_rhat = rhat["tau"].values

                            compare = az.loo(trace)   

                            elpd = compare["elpd_loo"]
                            se = compare["se"]
                            p_loo = compare["p_loo"]                    
                            r_squared_total.append(r_squared)
                            if j ==0:
                                data = [sweep1+indices[0], sweep2+indices[1],elpd,p_loo,se,r_squared] +  list(tau_val) + list(tau_err) + list(tau_rhat) + [C_val] + [C_err]
                            else:
                                data = [sweep1+indices[0], sweep2+indices[1],elpd,p_loo,se,r_squared] + list(A_val)+ list(A_err) + list(A_rhat) + list(tau_val) + list(tau_err) + list(tau_rhat)+ [C_val] + [C_err]

                            df_existing = pd.read_excel(file_name, sheet_name=f"exp_{j+1}")
                            new_row = pd.DataFrame([data],columns=columns)
                            df= pd.concat([df_existing, new_row], ignore_index=True)
                            with pd.ExcelWriter(file_name, mode="a",if_sheet_exists="replace",engine="openpyxl") as writer:
                                df.to_excel(writer, sheet_name=f"exp_{j+1}",index=False)
                    df_fit = pd.concat(fit_columns,axis=1)
                    with pd.ExcelWriter(file_name[:-5] + "_y_fit.xlsx", mode="a",if_sheet_exists="replace",engine="openpyxl") as writer:
                            df_fit.to_excel(writer, sheet_name=f"y_fit_exp_{j+1}",index=False)
                                
                 
        self.finished.emit()
        


def r2_score(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()


    def plot_peaks(self):
        data = np.loadtxt(self.raw_data, skiprows=1) 
        time = data[:, 0]*1e-3
        counts = data[:, 1]
        index_list = []
        peak_counts = counts 
        self.time = time
        self.counts = counts
              
        if self.filter_state:
            peak_counts =  savgol_filter(counts,self.window, polyorder=self.poly)
        if min(peak_counts) != 0:
            trunc_threshold = self.threshold*min(peak_counts) 
        else:
            trunc_threshold = self.threshold*np.sort(peak_counts)[1] 
        peaks = np.sort(find_peaks(peak_counts/max(peak_counts),height=0.6)[0]).tolist()
        peaks.append(np.where(counts == max(counts))[0][0])
        for peak in peaks:           
            try:
                start_peak = peak
                end_peak = np.where(peak_counts[peak:] <= trunc_threshold)[0][0] + peak
                index_list.append([start_peak,end_peak])
            except:
                continue         

        def filter_by_max_difference(pairs):
            grouped = {}
            
            for pair in pairs:
                key = pair[1]  
                diff = abs(pair[0] - pair[1])  
                
                if key not in grouped or diff < grouped[key][0]:
                    grouped[key] = (diff, pair) 
            
            result = [item[1] for item in grouped.values()]
            return result
        index_list = filter_by_max_difference(index_list)
        temp = []
        for start_idx, end_idx in index_list:
            double_diff = np.diff(counts[start_idx:end_idx],n=2)
            start_idx = np.where(double_diff == min(double_diff))[0][0] + start_idx
            temp.append([start_idx,end_idx ])
        index_list = temp
        self.index_list = index_list
        self.counter = 0
        pen = pg.mkPen(color=(255, 0, 0), width=1)

        self.raw_plot.plot(time,counts)  
        for start_idx, end_idx in self.index_list:
            self.raw_plot.plot(time[start_idx:end_idx],counts[start_idx:end_idx],pen=pen)  
           
        


    def select_regions(self):
        peak_loc = []
        start_idx = 0
        self.peak_indices += " "
        for idx, c in enumerate(self.peak_indices):
            try:
                index = int(self.peak_indices[idx])
            except:
                end_idx = idx
                index = int(self.peak_indices[start_idx:end_idx])
                peak_loc.append(index)  
                start_idx = idx+1
           
        self.index_list = [self.index_list[idx] for idx in peak_loc]
        for idx, indices in enumerate(self.index_list):
            algo = rpt.Dynp(model="l2").fit(self.counts[indices[0]:indices[1]])
            result = algo.predict(n_bkps=4)
            result.append(0)
            result = np.sort(result)
            self.index_list[idx][0] = self.index_list[idx][0] + result[1] + self.manual_trunc_start
            self.index_list[idx][1] = self.index_list[idx][0] + result[-2] + self.manual_trunc_end
       
           

    def getFileName(self):
        file_filter = 'Data File (*.xlsx *.csv *.dat *.txt)' 
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            filter=file_filter,
            initialFilter='Data File (*.xlsx *.csv *.dat *.txt)' 
        )[0]
        self.textboxes[8].setText(str(response))
    
    def run1(self):
        self.filter_state = self.filter.isChecked()
        self.threshold = int(self.textboxes[0].text())
        self.poly = int(self.textboxes[1].text())
        self.window = int(self.textboxes[2].text())
        self.manual_trunc_start = int(self.textboxes[3].text())
        self.manual_trunc_end = int(self.textboxes[4].text())
        self.peak_indices=  self.textboxes[5].text()
        self.sweep_range = float(self.textboxes[6].text())
        self.sweep_steps = int(self.textboxes[7].text())
        self.raw_data = self.textboxes[8].text()
        self.data_folder = self.textboxes[9].text()
        self.samples = int(self.textboxes[10].text())
        self.cores = int(self.textboxes[11].text())
        self.chains = int(self.textboxes[12].text())
        self.n_exp = int(self.textboxes[13].text())
        self.plot_peaks()
    
    def run2(self):
        self.filter_state = self.filter.isChecked()
        self.threshold = int(self.textboxes[0].text())
        self.poly = int(self.textboxes[1].text())
        self.window = int(self.textboxes[2].text())
        self.manual_trunc_start = int(self.textboxes[3].text())
        self.manual_trunc_end = int(self.textboxes[4].text())
        self.peak_indices=  self.textboxes[5].text()
        self.sweep_range = float(self.textboxes[6].text())
        self.sweep_steps = int(self.textboxes[7].text())
        self.raw_data = self.textboxes[8].text()
        self.data_folder = self.textboxes[9].text()
        self.samples = int(self.textboxes[10].text())
        self.cores = int(self.textboxes[11].text())
        self.chains = int(self.textboxes[12].text())
        self.n_exp = int(self.textboxes[13].text())
        self.fit_plot.clear()
        if self.counter == 0:
            self.plot_peaks()
            self.select_regions()
            self.raw_plot.clear()
            self.raw_plot.plot(self.time,self.counts)  
            pen = pg.mkPen(color=(255, 0, 0), width=1)
            for start_idx, end_idx in self.index_list:
                self.raw_plot.plot(self.time[start_idx:end_idx],self.counts[start_idx:end_idx],pen=pen)  
        
        start_idx, end_idx  = self.index_list[self.counter]
        self.fit_plot.plot(self.time[start_idx+self.manual_trunc_start:end_idx + self.manual_trunc_end],self.counts[start_idx+self.manual_trunc_start:end_idx+ self.manual_trunc_end])



        
    def run3(self):
        self.filter_state = self.filter.isChecked()
        self.threshold = int(self.textboxes[0].text())
        self.poly = int(self.textboxes[1].text())
        self.window = int(self.textboxes[2].text())
        self.manual_trunc_start = int(self.textboxes[3].text())
        self.manual_trunc_end = int(self.textboxes[4].text())
        self.peak_indices=  self.textboxes[5].text()
        self.sweep_range = float(self.textboxes[6].text())
        self.sweep_steps = int(self.textboxes[7].text())
        self.raw_data = self.textboxes[8].text()
        self.data_folder = self.textboxes[9].text()
        self.samples = int(self.textboxes[10].text())
        self.cores = int(self.textboxes[11].text())
        self.chains = int(self.textboxes[12].text())
        self.n_exp = int(self.textboxes[13].text())
        self.index_list[self.counter][0] = self.index_list[self.counter][0] + self.manual_trunc_start
        self.index_list[self.counter][1] = self.index_list[self.counter][1] + self.manual_trunc_end
        self.counter +=1
        if self.counter > len(self.index_list)-1:
            self.counter  = len(self.index_list)-1
            msg = QMessageBox()
            msg.setWindowTitle("All regions confirmed")
            msg.setText("All regions confirmed")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()

        

    def run4(self):
        self.filter_state = self.filter.isChecked()
        self.threshold = int(self.textboxes[0].text())
        self.poly = int(self.textboxes[1].text())
        self.window = int(self.textboxes[2].text())
        self.manual_trunc_start = int(self.textboxes[3].text())
        self.manual_trunc_end = int(self.textboxes[4].text())
        self.peak_indices=  self.textboxes[5].text()
        self.sweep_range = float(self.textboxes[6].text())
        self.sweep_steps = int(self.textboxes[7].text())
        self.raw_data = self.textboxes[8].text()
        self.data_folder = self.textboxes[9].text()
        self.samples = int(self.textboxes[10].text())
        self.cores = int(self.textboxes[11].text())
        self.chains = int(self.textboxes[12].text())
        self.n_exp = int(self.textboxes[13].text())
        self.fit_plot.clear()
            
        self.thread = QThread()  # Create a new thread
        self.worker = Worker(self.time, self.counts,self.index_list, self.sweep_range, self.sweep_steps, self.raw_data, self.data_folder,self.samples, self.cores, self.chains,self.n_exp,self.exponential_label,self.progress_label,self.iteration_label)  # Create the worker

        # Move the worker to the thread
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.bayesian_inference)

        # Start the thread
        self.thread.start()
        self.worker.finished.connect(self.thread.quit)  # Quit the thread after completion
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)


      

      
    def initUI(self):
        layout = QVBoxLayout()
        plot_layout = QHBoxLayout()
        
        layout.setSpacing(5) 
        layout.setContentsMargins(5, 5, 5, 5)
        # Three plots in the center
        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setMinimumSize(600, 400)   
        self.fit_plot = pg.PlotWidget()
        self.fit_plot.setMinimumSize(600, 400)  
        self.trunc_plot = pg.PlotWidget()
        self.trunc_plot.setMinimumSize(600, 400)  
        plot_layout.addWidget(self.raw_plot)
        plot_layout.addWidget(self.fit_plot)
        # plot_layout.addWidget(self.trunc_plot)
        
        layout.addLayout(plot_layout)
        # Controls and Table Layout
        control_layout = QVBoxLayout()
        button_layout = QVBoxLayout()
        bottom_layout =  QHBoxLayout()
        progress_layout = QVBoxLayout() 
        button_layout.setSpacing(1) 
        progress_layout.setSpacing(1) 
        labels = ["Threshold","Polynomial","Window Size",
            "Manual Start truncation", "Manual End truncation", "Peak indices","Auto truncation sweep range",
            "Auto truncation bins", "Raw data file", "Saved data folder", 
            "Bayesian samples", "Cores", "Markov chains", "Max exp model number"
        ]
        
        self.textboxes = []
        for label in labels:
            lbl = QLabel(label)
            txt = QLineEdit()
            txt.setSizePolicy(QLineEdit().sizePolicy())  # Ensure expandability
            txt.setMaximumSize(300, 40)  # Ensure uniform size
            control_layout.addWidget(lbl)
            control_layout.addWidget(txt)
            self.textboxes.append(txt)
        self.exponential_label = QLabel("Exponential Model 1",self)
        self.progress_label = QLabel(f"Truncating 0 and Extending 0",self)
        self.iteration_label = QLabel(f"Iteration 0 of 0",self)
        self.exponential_label.setFont(QFont("Arial", 30))  # Set font family and size
        self.progress_label.setFont(QFont("Arial", 30))  # Set font family and size
        self.iteration_label.setFont(QFont("Arial", 30))  # Set font family and size
        
        self.exponential_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.iteration_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exponential_label.setStyleSheet("line-height: 0.5;")
        self.progress_label.setStyleSheet("line-height: 0.5;")
        self.iteration_label.setStyleSheet("line-height: 0.5;")
        
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.iteration_label)
        progress_layout.addWidget(self.exponential_label)
        button_layout.addLayout(progress_layout)
        self.textboxes[0].setText("2")
        self.textboxes[1].setText("1")
        self.textboxes[2].setText("100")
        self.textboxes[3].setText("0")
        self.textboxes[4].setText("0")
        self.textboxes[5].setText("1a2")
        self.textboxes[6].setText("0.1")
        self.textboxes[7].setText("1")
        self.textboxes[9].setText("fit_data/")
        self.textboxes[10].setText("5000")
        self.textboxes[11].setText("1")
        self.textboxes[12].setText("8")
        self.textboxes[13].setText("2")

        self.filter = QCheckBox("Use filter for peak finding")

        control_layout.addWidget(self.filter)



        # Checkbox for Manual Truncation
        bottom_layout.addLayout(control_layout)
        button_names = ["Open data", "Plot all peaks", "Designate Region", "Confirm Region", "Fit curves"]
        
        self.buttons = []

        for name in button_names:
            button = QPushButton(name)
            button.setSizePolicy(QPushButton().sizePolicy())  # Expand evenly
            button.setMinimumSize(25, 40)  # Uniform button size
            self.buttons.append(button)
            button_layout.addWidget(button)    
        
        bottom_layout.addLayout(button_layout)
        
        self.buttons[0].clicked.connect(self.getFileName)
        self.buttons[1].clicked.connect(self.run1)
        self.buttons[2].clicked.connect(self.run2)
        self.buttons[3].clicked.connect(self.run3)
        self.buttons[4].clicked.connect(self.run4)
        
        
        layout.addLayout(bottom_layout)
        self.setLayout(layout)
        self.setWindowTitle("Curve Fitting")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
