from io import StringIO
from os.path import exists
from os import remove
import unittest
from unittest.mock import patch
from utils import raises
from bipca import command_line
from contextlib import redirect_stdout

import anndata as ad
import numpy as np

from test_data import path_to_filtered_data, filtered_adata, test_output_path,test_datadir

class Test_bipca_commandline(unittest.TestCase):
	@raises(ValueError)
	def test_parser_path_exists(self):
		bad_input_file = 'foo'
		command_line.bipca_parse_args([bad_input_file,test_output_path])
	def test_parser_good_path(self):
		args = command_line.bipca_parse_args([path_to_filtered_data,test_output_path])
		assert args.X == path_to_filtered_data
		assert args.Y == test_output_path

	def test_bipca_runs(self):
		good_input_file = path_to_filtered_data
		command_line.bipca_main([path_to_filtered_data,test_output_path,'-v','0',
			'-nsubs','2','-subsize','200','-qits','2','--no_plotting_spectrum','-njobs','1'])
		output = ad.read_h5ad(test_output_path)
		plotting_keys = list(output.uns['bipca']['plotting_spectrum'].keys())
		assert 'Y' not in plotting_keys 
		assert 'fits' in plotting_keys
		command_line.bipca_main([path_to_filtered_data,test_output_path,'-v','0',
			'-nsubs','2','-subsize','200','-qits','2','-njobs','1'])
		output = ad.read_h5ad(test_output_path)
		plotting_keys = list(output.uns['bipca']['plotting_spectrum'].keys())
		assert 'Y' in plotting_keys 
class Test_bipca_plotting(unittest.TestCase):
	@raises(ValueError)
	def test_parser_path_exists(self):
		bad_input_file = 'foo'
		command_line.bipca_plot_parse_args([bad_input_file,test_output_path])
	def test_bipca_plot(self):
		#command_line.bipca_main([path_to_filtered_data,test_output_path,'-v',1])
		if exists(test_datadir + '/histogram.jpg'):
			remove(test_datadir+'/histogram.jpg')
			remove(test_datadir+'/spectrum.jpg')
			remove(test_datadir+'/KS.jpg')
		command_line.bipca_plot([test_output_path,test_datadir+'/','-plus','5','50','-minus','5','10','-scale','linear'])
		assert exists(test_datadir+'/histogram.jpg')
		assert exists(test_datadir+'/spectrum.jpg')
		assert exists(test_datadir+'/KS.jpg')