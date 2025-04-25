from svg.path import parse_path # requires svg.path: pip install svg.path
from svg.path.path import Line
from xml.dom import minidom
from PyPDF2 import PdfReader # requires PyPDF2: pip install PyPDF2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os, sys
import subprocess

# sys.path.append('../database_ctu-chb/')

# sudo apt-get install inkscape

# from src.data import load_FHR_UC, load_clinical_data, load_targets 
# requires: pip install PyWavelets
#           pip install scikit-fda
#           pip install scikit-image
# from src.database import db_to_dataframe



def index_outliers(y_values, max_dev=5):
    """ Function that returns the indices of the values with possible errors or outliers
    
    Parameters:
        - y_values: values to consider
        - max_dev: maximum deviation to take into account to consider it a normal value"""
    
    array = np.array(y_values)
    mean = np.nanmean(array)
    standard_deviation = np.nanstd(array)
    distance_from_mean = abs(array - mean)
    outliers = np.where(distance_from_mean > max_dev * standard_deviation)[0]
    
    return outliers




def add_path(xs_array, ys_array, x0, x1, y0, y1, mode_diff):
    """ Function that adds path (x0,y0)-(x1,y1) in the right place of a fixed array.
    
    Parameters:
        - xs_array: x array with fixed values
        - ys_array: y array 
        - x0: x start
        - x1: x end
        - y0: y start
        - y1: y end
        - mode_diff: difference between consecutive x values """
                            
    x0_index = np.argmin(np.abs(np.array(xs_array)-x0))
    ys_array[x0_index] = y0
    x1_index = np.argmin(np.abs(np.array(xs_array)-x1))
    ys_array[x1_index] = y1

    diff_x_threshold = mode_diff + 0.05 #threshold between xs fixed values

    #non consecutive values -> interpolation
    if(abs(x1-x0)>diff_x_threshold):
        #get indices for interpolation: xs in interval (x0,x1)
        inter_index = np.arange(x0_index+1,x1_index)
        y_inter = np.interp(xs_array[inter_index], np.array([x0,x1]), np.array([y0,y1]))
        ys_array[inter_index] = y_inter

    return ys_array




def add_no_close_point(array, point, threshold):
    """ Function that adds a point to an array if it is not too close to the other points in the array.

    Parameters:
        - array: array of points
        - point: point to add
        - threshold: threshold to consider a point too close """
    
    
    if not array: #empty
        return [point]
    
    idx = np.abs(np.array(array) - point).argmin()
    closest_value = array[idx]

    if abs(closest_value - point) > threshold:
        result_array = array + [point]
    else:
        result_array = array
    
    return result_array



def plot_signal(x, y, xlim, ylim, color, title):
    """ Function that plots a signal

    Parameters:
        - x: values of x
        - y: values of y
        - xlim: range of values in x
        - ylim: range of values in y
        - color: color line
        - title: title of plot """
    
    for i,ys in enumerate(y):
        plt.plot(x,ys,color=color[i])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    plt.grid(True)




import xml.etree.ElementTree as ET

def read_SVG_part(file_name, file_path, page, remove_outliers=True, max_dev=5, plot=False):
    """ Function that parses the SVG file, extracts and returns the coordinates of the FHR and UC signals
    
    Parameters:
        - file_name: name of SVG file
        - file_path: path of SVG file
        - page: part number (for split signals)
        - remove_outliers: eliminate or not possible errors or outliers from the signal
        - max_dev : max deviation for outliers
        - plot: plot signal or not """

    #constants
    # coord_legend_red = 4304.8; coord_legend_blue = 4234.0; coord_legend_pink = 4094.8
    coord_legend_red = 234.0; coord_legend_blue = 243.5; coord_legend_pink = 262.5
    BLUE = "#0000ff" 
    # BLUE = "#8080ff"
    RED = "#ff0000" 
    BLACK = "#000000" 
    GREEN = "#a4e1a4"
    GREEN_2 = "#70ff70"
    PINK = "#ff00ff"
    #values between horizontal lines
    value_y_FHR = 10; value_y_UC = 20
    horizontal_lines_FHR = 16; horizontal_lines_UC = 5
    # 32 de alto 
    points_page = 26*60*4+1
    
    #parse svg file -> get list of paths
    doc = minidom.parse(file_path)
    width = float(doc.getElementsByTagName('svg')[0].getAttribute('width'))
    height = float(doc.getElementsByTagName('svg')[0].getAttribute('height'))
    path_strings = [[path.getAttribute('d'),path.getAttribute('style')] for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()

 
    #check limits of green grid and difference between x values
    xs_grid = []; ys_grid = [] 
    xs_FHR_diff = []; xs_UC_diff = []
    xs_FHR_aux = []
    h_ys = []
    
    for path_string in path_strings:
        path = parse_path(path_string[0])
        color = path_string[1].split("stroke:")[1].split(';')[0]
        if(color in [RED, BLUE, BLACK, GREEN, GREEN_2, PINK]):
            for e in path:
                if isinstance(e, Line):
                    #rotate images if it is necessary
                    if(width < height):
                        y0 = e.start.real; x0 = e.start.imag
                        y1 = e.end.real; x1 = e.end.imag
                    else:
                        x0 = e.start.real; y0 = e.start.imag
                        x1 = e.end.real; y1 = e.end.imag
                    
                    #FHR
                    if(color==RED or color==BLUE):
                        if(x1!=x0):
                            xs_FHR_diff.append(abs(x1-x0))
                            xs_FHR_aux = add_no_close_point(xs_FHR_aux, x0, 1.17)
                            xs_FHR_aux = add_no_close_point(xs_FHR_aux, x1, 1.17)
                    #UC
                    elif(color==BLACK):
                        if(x1!=x0):
                            xs_UC_diff.append(abs(x1-x0))
                    #grid
                    elif(color==GREEN or color==GREEN_2):
                        xs_grid += [x0,x1]
                        ys_grid += [y0,y1]

    if len(xs_grid) == 0:
        return None, None, None, None, None
    
    min_x_grid = min(xs_grid); max_x_grid = max(xs_grid)
    min_y_grid = min(ys_grid); max_y_grid = max(ys_grid)
    
    #debug
    mode_diff_x = stats.mode(xs_FHR_diff)[0]
    mean_diff_x = np.array(xs_FHR_diff).mean()
    #min_diff_x = np.nanmin(np.array(xs_FHR_diff))
    xs_FHR_aux = sorted(xs_FHR_aux)
    
    #create FHR, UC, MHR arrays with x fixed values
    const_diff_x = (max_x_grid - min_x_grid)/(points_page)
    xs_array = np.arange(min_x_grid, max_x_grid, const_diff_x)
    ys_FHR_array = np.nan* np.zeros(len(xs_array))
    ys_UC_array = np.nan* np.zeros(len(xs_array))
    ys_MHR_array = np.nan* np.zeros(len(xs_array))
    
    
    #loop over paths and set the (x,y) values
    for path_string in path_strings:
        path = parse_path(path_string[0])
        color = path_string[1].split("stroke:")[1].split(';')[0]
        if(color=="none"):
            continue
        
        stroke_width = path_string[1].split("stroke-width:")[1].split(';')[0]
        
        if(float(stroke_width) < 10. and color in [RED, BLUE, BLACK, GREEN, GREEN_2, PINK]):
            for e in path:
                if isinstance(e, Line):
                    #rotate images if it is necessary
                    if(width < height):
                        y1 = e.start.real; x1 = e.start.imag
                        y0 = e.end.real; x0 = e.end.imag
                    else:
                        x0 = e.start.real; y0 = e.start.imag
                        x1 = e.end.real; y1 = e.end.imag
                    
                    #conditions for valid (x,y) -> grid limits and not in legend
                    if((min_x_grid <= x0 <= max_x_grid) and (min_x_grid <= x1 <= max_x_grid)
                      and (min_y_grid <= y0 <= max_y_grid) and (min_y_grid <= y1 <= max_y_grid)
                        and (y0!=coord_legend_red and y1!=coord_legend_red 
                             and y0!=coord_legend_blue and y1!=coord_legend_blue
                              and y0!=coord_legend_pink and y1!=coord_legend_pink)):
                    
                        #FHR
                        if(color==RED or color==BLUE):
                            #add path (x0,y0)-(x1,y1) with xs closest fixed values
                            ys_FHR_array = add_path(xs_array, ys_FHR_array, x0, x1, y0, y1, const_diff_x)
                            

                        #UC
                        elif(color==BLACK): 
                            #add path (x0,y0)-(x1,y1) with xs closest fixed values
                            ys_UC_array = add_path(xs_array, ys_UC_array, x0, x1, y0, y1, const_diff_x)
                        
                        #MHR (mother)
                        elif(color==PINK):
                            ys_MHR_array = add_path(xs_array, ys_MHR_array, x0, x1, y0, y1, const_diff_x)
                        
                        #grid
                        elif((color==GREEN or color==GREEN_2) and (float(stroke_width) == 6. or float(stroke_width) == 7.07)): 
                            #horizontal lines
                            if(x0==min_x_grid and x1==max_x_grid and y0==y1):
                                h_ys.append(y0)
                                
    #horizontal green lines                       
    h_ys = sorted(h_ys)
    upper_limit_FHR = h_ys[-1]; 
    lower_limit_FHR = h_ys[6]
    upper_limit_UC = h_ys[5]; 
    lower_limit_UC = h_ys[0]

    
    #revert y values of rotated images
    if(width < height):
        ys_FHR_array = ys_FHR_array[::-1]
        ys_UC_array = ys_UC_array[::-1]
        ys_MHR_array = ys_MHR_array[::-1]
    
    #remove outliers or errors
    dev = max_dev 
    if(remove_outliers):
        index = index_outliers(ys_FHR_array,max_dev=dev)
        ys_FHR_array[index] = np.nan
        index = index_outliers(ys_UC_array,max_dev=dev)
        ys_UC_array[index] = np.nan
        index = index_outliers(ys_MHR_array,max_dev=dev)
        ys_MHR_array[index] = np.nan
    
    #scale y values
    min_ys_scale_FHR = 50.; max_ys_scale_FHR = min_ys_scale_FHR + horizontal_lines_FHR * value_y_FHR
    min_ys_scale_UC = 0.; max_ys_scale_UC = min_ys_scale_UC + horizontal_lines_UC * value_y_UC
    ys_FHR_array = np.interp(ys_FHR_array, (lower_limit_FHR, upper_limit_FHR), (min_ys_scale_FHR, max_ys_scale_FHR))
    ys_UC_array = np.interp(ys_UC_array, (lower_limit_UC, upper_limit_UC), (min_ys_scale_UC, max_ys_scale_UC))
    ys_MHR_array = np.interp(ys_MHR_array, (lower_limit_FHR, upper_limit_FHR), (min_ys_scale_FHR, max_ys_scale_FHR))
    
    #fixed x values
    diff_x_scale = 0.25
    xs_array = np.arange((page-1)*points_page*diff_x_scale,page*points_page*diff_x_scale,diff_x_scale)
    length_difference = len(ys_FHR_array) - len(xs_array)
    ys_FHR_array = ys_FHR_array[length_difference:]
    ys_UC_array = ys_UC_array[length_difference:]
    ys_MHR_array = ys_MHR_array[length_difference:]

    #plot FHR and UC signals
    plt.figure(figsize=(16,8))    
    plt.subplot(211)
    plot_signal(xs_array, [ys_FHR_array, ys_MHR_array], xlim=(min(xs_array), max(xs_array)), 
                ylim=(min_ys_scale_FHR, max_ys_scale_FHR), color=[RED,PINK], title=file_name+"-"+str(page)+"-FHR")
    plt.subplot(212)
    plot_signal(xs_array, [ys_UC_array], xlim=(min(xs_array), max(xs_array)), 
                ylim=(min_ys_scale_UC, max_ys_scale_UC), color=[BLACK], title=file_name+"-"+str(page)+"-UC")
    if not os.path.exists('plots/signal_parts_plots/'+file_name+''):
        os.makedirs('plots/signal_parts_plots/'+file_name+'')
    #plt.savefig('plots/signal_parts_plots/'+file_name+'/'+file_name+'_'+str(page)+'.pdf')
    
    if(plot): 
        plt.show()
    else:
        plt.close()
    
    ylim_FHR = (min_ys_scale_FHR, max_ys_scale_FHR)
    ylim_UC = (min_ys_scale_UC, max_ys_scale_UC)
    ylim = [ylim_FHR, ylim_UC]
    
    return xs_array, ys_FHR_array, ys_UC_array, ys_MHR_array, ylim




def read_PDF_complete(file_name, pdf_folder, svg_folder, pdf_to_svg=False, plot_parts=False, plot_complete=False):
    """ Function that reads all pages of pdf and returns the coordinates of the FHR and UC signals
    
    Parameters:
        - file_name: name of pdf file
        - pdf_folder: folder of pdf file
        - svg_folder: folder of svg files
        - pdf_to_svg: convert pdf to svg or not
        - plot_parts: plot each signal part of pdf or not
        - plot_complete : plot complete signals or not """

    BLUE = "#0000ff" 
    # BLUE = "#8080ff"
    RED = "#ff0000" 
    BLACK = "#000000" 
    # GREEN = "#70ff70"
    GREEN = "#a4e1a4"
    PINK = "#ff00ff"
    seconds_page = 26.* 60.
    
    #convert pdf to svg if necessary
    if(pdf_to_svg):
        #run script -> ./pdf_to_svg.sh file_name pdf_folder svg_folder
        print('PDF to SVG ...')
        subprocess.call(["./pdf_to_svg.sh", file_name, pdf_folder, svg_folder])

    pdf = PdfReader(open(pdf_folder+file_name+'.pdf','rb'))
    pages = len(pdf.pages)
    
    
    points_page = 60*26*4+1
    xs_complete = np.nan* np.zeros(points_page*pages)  
    ys_FHR_complete = np.nan* np.zeros(points_page*pages)
    ys_UC_complete = np.nan* np.zeros(points_page*pages)
    ys_MHR_complete = np.nan* np.zeros(points_page*pages)
        
    # loop over each svg page
    for page in range(1, pages+1):  
        # print('Reading page', page)
        file_path = os.path.join(svg_folder, file_name, f"{file_name}-part{page}.svg")
        
        result = read_SVG_part(file_name, file_path, page=page, remove_outliers=False, max_dev=5, plot=plot_parts)
        # If that page fails
        if result is None:
            print(f"Skipping page {page} due to empty grid or no valid data.")
            continue
        
        xs, FHR, UC, MHR, ylim = result
        
        xs_complete[(page-1)*points_page:page*points_page] = xs
        
        ys_FHR_complete[(page-1)*points_page:page*points_page] = FHR
        ys_UC_complete[(page-1)*points_page:page*points_page] = UC
        ys_MHR_complete[(page-1)*points_page:page*points_page] = MHR
        
    
    #plot FHR and UC complete signals
    plt.figure(figsize=(16,8))    
    plt.subplot(211)
    plot_signal(xs_complete, [ys_FHR_complete, ys_MHR_complete], xlim=[min(xs_complete),max(xs_complete)],
                ylim=ylim[0], color=[RED,PINK], title=file_name+"-FHR")
    plt.subplot(212)
    plot_signal(xs_complete, [ys_UC_complete], xlim=[min(xs_complete),max(xs_complete)],
                ylim=ylim[1], color=[BLACK], title=file_name+"-UC")
    if not os.path.exists('plots/signal_complete_plots/'+file_name+''):
        os.makedirs('plots/signal_complete_plots/'+file_name+'')
    plt.savefig('plots/signal_complete_plots/'+file_name+'/'+file_name+'.pdf')
    
    if(plot_complete): 
        plt.show()
    else:
        plt.close()
        
    #create and save a dataframe
    data_FHR_UC = np.array([xs_complete, ys_FHR_complete, ys_UC_complete, ys_MHR_complete]).T
    df_FHR_UC = pd.DataFrame(data=data_FHR_UC)
    names = [['Elapsed Time', 'FHR', 'UC', 'MHR'], ['seconds', 'bpm', 'nd', 'bpm']]
    df_FHR_UC.columns = pd.MultiIndex.from_arrays(names)
    df_FHR_UC.to_csv('csv/data/'+file_name+'.csv', index=False, compression='gzip')
    
    return xs_complete, ys_FHR_complete, ys_UC_complete, ys_MHR_complete



def clean_svg_folder(svg_folder):
    """ Function that cleans the svg folder """
    subprocess.call('rm -rf ' + svg_folder, shell=True)



def clean_plots_folder(plots_folder):
    """ Function that cleans the plots folder """
    subprocess.call('rm -rf ' + plots_folder, shell=True)




def pdf_signal_csv(pdf_folder = "pdf/", svg_folder = "svg/", pdf_to_svg = True, plot_parts =  False, plot_complete = True):
    """ Function that reads all pdf files in a folder, converts them to svg, and extracts the FHR and UC signals.
    
    Parameters:
        - pdf_folder: folder of pdf files
        - svg_folder: folder of svg files
        - pdf_to_svg: convert pdf to svg or not
        - plot_parts: plot each signal part of pdf or not
        - plot_complete : plot complete signals or not """
    
    list_pdfs = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_hex = file.split('.')[0]
            list_pdfs.append(file_hex)
            
    print('There are', len(list_pdfs), 'pdfs in', pdf_folder)
    file_name = list_pdfs[0]

    # Create folders to store csv files
    if not os.path.exists('csv/data'):
        os.makedirs('csv/data')

    for i, file_name in enumerate(list_pdfs):
        if file_name + '.csv' in os.listdir('csv/data/'):
            print('File '+str(i)+':', file_name, 'already read')
            continue
            
        print('File '+str(i)+':', file_name)
        xs, FHR, UC, MHR = read_PDF_complete(file_name, pdf_folder, svg_folder, pdf_to_svg, plot_parts, plot_complete)