from distutils.core import setup

setup(
	name = 'qed',
	version = '1.0.1',
	description = 'Quantitative Estimation of Drug-likeness descriptor',
	long_description = 'Qed: Bickerton et al. (2012) Nature Chemistry 4, 90-98',
	author = 'Hans De Winter',
	author_email = 'hans@silicos-it.com',
	url = 'http://www.silicos-it.com',
	packages = ['silicos_it', 'silicos_it.errors', 'silicos_it.descriptors'],
	package_data = {'silicos_it.descriptors': ['qed_test.smi']},
	data_files = [('silicos_it', ['COPYING.LESSER'])],
	license = 'LGPL v3',
	platforms = ['OS X 10.7']
	)
