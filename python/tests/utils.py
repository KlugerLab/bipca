from functools import wraps


def raises(exc, someopt=None):
	def outer(fn):
		@wraps(fn)
		def wrapper(*args, **kwargs):
			try:
				fn()
			except Exception as err:
				if isinstance(err,exc):
					return True
				else:
					raise err
			return False
		return wrapper
	return outer

