from functools import wraps

class TestingError(Exception):
	pass
def raises(exc, someopt=None):
	def outer(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			try:
				fn(*args,**kwargs)
				raise TestingError(f'{fn.__name__} ran without raising {exc.__name__}')
			except Exception as err:
				if isinstance(err,exc):
					return True
				elif isinstance(err, TestingError):
					raise err
				else:
					raise TestingError(f'{fn.__name__} ran without raising {exc.__name__}') from err
			return False
		return wrapper
	return outer
