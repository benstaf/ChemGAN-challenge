# Copyright 2012 by Silicos-it, a division of Imacosi BVBA
# Biscu-it is free software; you can redistribute it and/or modify it under the 
# terms of the GNU Lesser General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later version.
# Biscu-it is distributed in the hope that it will be useful, but without any warranty; 
# without even the implied warranty of merchantability or fitness for a particular 
# purpose. See the GNU Lesser General Public License for more details.
#
# Biscu-it is using the Python bindings of RDKit. RDKit is free software; you can 
# redistribute it and/or modify it under the terms of the BSD-2 license terms as published by
# the Open Source Initiative. Please refer to the RDKit license for more details.


__all__ = ['SilicosItError', 'WrongArgument']



class SilicosItError(Exception):
	"""Base class for exceptions in Silicos-it code"""
	pass



class WrongArgument(SilicosItError):
	"""
	Exception raised when argument to function is not of correct type.

	Attributes:
		function -- function in which error occurred
		msg      -- explanation of the error
	"""
	def __init__(self, function, msg):
		self.function = function
		self.msg = msg
